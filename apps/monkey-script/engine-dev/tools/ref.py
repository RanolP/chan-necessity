# Numpy reference of one decoder step built only from manifest.json (the layout
# the WGSL kernels read), checked against ORT CPU running decoder_step.q4f16.onnx.
# Catches: wrong nibble / zero-point order, wrong q|k|v or gate|up row split,
# wrong rope form, wrong GQA head mapping.
import json, os, sys, time
import numpy as np

S = os.environ["S"]; E = f"{S}/engine"
man = json.load(open(f"{E}/model/manifest.json")); T = man["tensors"]; C = man["config"]
data = np.memmap(f"{S}/fusion/decoder_weights.q4f16.data", dtype=np.uint8, mode="r")
qkn = np.fromfile(f"{E}/model/qknorm.bin", dtype=np.uint8)

def raw(name):
    t = T[name]; src = data if t["src"].startswith("decoder") else qkn
    return np.concatenate([np.asarray(src[o:o + l]) for o, l in t["segments"]])

def f16(name): return raw(name).view(np.float16).astype(np.float32)

def gemv(tag, x, K):
    q = raw(f"{tag}.q"); s = raw(f"{tag}.s").view(np.float16).astype(np.float32)
    nb = K // 32; N = s.size // nb
    q = q.reshape(N, nb, 16); s = s.reshape(N, nb)
    z = raw(f"{tag}.z").reshape(N, -1)
    zz = np.stack([z & 15, z >> 4], -1).reshape(N, -1)[:, :nb].astype(np.float32)
    out = np.empty(N, np.float32); xb = x.reshape(nb, 32)
    for a in range(0, N, 16384):
        qq = q[a:a + 16384]
        w = np.stack([qq & 15, qq >> 4], -1).reshape(-1, nb, 32).astype(np.float32)
        w = (w - zz[a:a + 16384, :, None]) * s[a:a + 16384, :, None]
        out[a:a + 16384] = np.einsum("nbk,bk->n", w, xb)
    return out

def rms(x, w, eps=1e-6): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def rope(x, p):
    inv = C["rope_theta"] ** (-np.arange(64) * 2 / 128)
    f = p * inv; c = np.cos(np.concatenate([f, f])); s = np.sin(np.concatenate([f, f]))
    return x * c + np.concatenate([-x[..., 64:], x[..., :64]], -1) * s

def step(e, pos, pk, pv):
    """e [2048]; pk/pv [28,8,P,128] -> logits, new k/v rows [28,8,128]"""
    h = e.astype(np.float32); nk = np.zeros((28, 8, 128), np.float32); nv = nk.copy()
    for L in range(28):
        x = rms(h, f16(f"l{L}.ln1"))
        qkv = gemv(f"l{L}.qkv", x, 2048)
        q = qkv[:2048].reshape(16, 128); k = qkv[2048:3072].reshape(8, 128); v = qkv[3072:].reshape(8, 128)
        q = rope(rms(q, f16(f"l{L}.qn")), pos); k = rope(rms(k, f16(f"l{L}.kn")), pos)
        nk[L] = k; nv[L] = v
        K = np.concatenate([pk[L], k[:, None]], 1); V = np.concatenate([pv[L], v[:, None]], 1)
        a = np.empty((16, 128), np.float32)
        for hh in range(16):
            sc = K[hh // 2] @ q[hh] * C["attn_scale_f16"]
            p = np.exp(sc - sc.max()); p /= p.sum(); a[hh] = p @ V[hh // 2]
        h = h + gemv(f"l{L}.o", a.reshape(-1), 2048)
        x = rms(h, f16(f"l{L}.ln2"))
        gu = gemv(f"l{L}.gu", x, 2048); g, u = gu[:6144], gu[6144:]
        h = h + gemv(f"l{L}.down", u * (g / (1 + np.exp(-g))), 6144)
    return gemv("lm", rms(h, f16("norm")), 2048), nk, nv

if __name__ == "__main__":
    import onnxruntime as ort
    rng = np.random.default_rng(0); P = 5; pos = P
    emb = np.fromfile(f"{E}/embed_tokens.int8.bin", np.int8); esc = np.fromfile(f"{E}/embed_scales.f32.bin", np.float32)
    tok = 151704
    e = emb[tok * 2048:(tok + 1) * 2048].astype(np.float32) * esc[tok]
    pk = (rng.standard_normal((28, 8, P, 128)) * 0.5).astype(np.float16)
    pv = (rng.standard_normal((28, 8, P, 128)) * 0.5).astype(np.float16)
    t0 = time.time()
    lg, nk, nv = step(e.astype(np.float16).astype(np.float32), pos, pk.astype(np.float32), pv.astype(np.float32))
    print("numpy step s", round(time.time() - t0, 1))
    sess = ort.InferenceSession(f"{E}/tools/refmodel/decoder_step.q4f16.onnx", providers=["CPUExecutionProvider"])
    o = sess.run(None, {"input_embeds": e.astype(np.float16)[None, None], "position_ids": np.array([[pos]], np.int64),
                        "past_keys": pk[:, None], "past_values": pv[:, None]})
    ol = o[0].reshape(-1).astype(np.float32)
    print("logits maxabs diff", float(np.abs(ol - lg).max()), "max|logit|", float(np.abs(ol).max()),
          "argmax ort/np", int(ol.argmax()), int(lg.argmax()))
    print("new k maxdiff", float(np.abs(o[1][:, 0, :, P].astype(np.float32) - nk).max()),
          "new v maxdiff", float(np.abs(o[2][:, 0, :, P].astype(np.float32) - nv).max()))
    for L in (0, 1, 2, 13, 27):
        dk = np.abs(o[1][L, 0, :, P].astype(np.float32) - nk[L]); dv = np.abs(o[2][L, 0, :, P].astype(np.float32) - nv[L])
        print("layer", L, "k diff", float(dk.max()), "v diff", float(dv.max()), "|v|max", float(np.abs(nv[L]).max()))
    rel = np.abs(ol - lg).max() / np.abs(ol).max(); print("logits rel", float(rel), "top5 ort", np.argsort(-ol)[:5], "np", np.argsort(-lg)[:5])
