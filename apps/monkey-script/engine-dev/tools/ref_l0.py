# Layer-0 intermediates for the first engine step on the CPU fixture (tok0, pos S,
# ORT-CPU prefill KV): qkv0 [4096] raw GEMV, h after layer 0 [2048]. Single-layer oracle.
exec(open('tools/ref.py').read().split('if __name__')[0])
t = json.load(open(f"{E}/test/truth.json")); Sx = t["S"]; tok = t["tok0"]
emb = np.fromfile(f"{E}/model/embed/embed_tokens.int8.bin", np.int8); esc = np.fromfile(f"{E}/model/embed/embed_scales.f32.bin", np.float32)
e = (emb[tok*2048:(tok+1)*2048].astype(np.float32) * esc[tok]).astype(np.float16).astype(np.float32)
pk = np.fromfile(f"{E}/test/prefill_k.f16", np.float16).reshape(28, 8, Sx, 128).astype(np.float32)
pv = np.fromfile(f"{E}/test/prefill_v.f16", np.float16).reshape(28, 8, Sx, 128).astype(np.float32)
h = e.copy(); L = 0
x = rms(h, f16("l0.ln1")); qkv = gemv("l0.qkv", x, 2048)
q = qkv[:2048].reshape(16, 128); k = qkv[2048:3072].reshape(8, 128); v = qkv[3072:].reshape(8, 128)
q = rope(rms(q, f16("l0.qn")), Sx); k = rope(rms(k, f16("l0.kn")), Sx)
K = np.concatenate([pk[0], k[:, None]], 1); V = np.concatenate([pv[0], v[:, None]], 1)
a = np.empty((16, 128), np.float32)
for hh in range(16):
    sc = K[hh // 2] @ q[hh] * C["attn_scale_f16"]; p = np.exp(sc - sc.max()); p /= p.sum(); a[hh] = p @ V[hh // 2]
h = h + gemv("l0.o", a.reshape(-1), 2048)
x = rms(h, f16("l0.ln2")); gu = gemv("l0.gu", x, 2048); g, u = gu[:6144], gu[6144:]
h = h + gemv("l0.down", u * (g / (1 + np.exp(-g))), 6144)
np.concatenate([qkv, a.reshape(-1), h]).astype(np.float32).tofile(f"{E}/test/ref_l0.f32")
print("ok", float(np.abs(qkv).max()), float(np.abs(h).max()))
