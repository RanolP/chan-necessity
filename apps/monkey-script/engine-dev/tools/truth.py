# Ground truth for the fixed 8 s clip with ORT CPU: mel -> encoder -> decoder_init
# -> greedy decoder_step. Writes test/ fixtures the browser engine is checked against:
# prefill KV (f16), first token, start position, ORT greedy tokens, logits of steps 1-3.
import json, os, time
import numpy as np, onnxruntime as ort

S = os.environ["S"]; E = f"{S}/engine"; R = f"{E}/tools/refmodel"; A = f"{S}/asr"
out = f"{E}/test"; os.makedirs(out, exist_ok=True)
pcm = np.fromfile(f"{S}/perf/clip.f32", np.float32)
F = np.array(json.load(open(f"{A}/mel_filters.json"))["data"], dtype=np.float64)
F = F.reshape(128, -1) if F.ndim == 1 else F
if F.shape[0] != 128: F = F.T

def mel(x):
    x = np.pad(x.astype(np.float64), 200, mode="reflect"); win = np.hanning(401)[:-1]
    T = len(pcm) // 160
    fr = np.stack([x[i * 160:i * 160 + 400] * win for i in range(T)])
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    m = np.log10(np.maximum(F @ p.T, 1e-10))
    m = np.maximum(m, m.max() - 8); return ((m + 4) / 4).astype(np.float32)

cfg = json.load(open(f"{A}/prompt_config.json")); P = cfg["prompt"]
M = mel(pcm)
so = ort.SessionOptions(); so.log_severity_level = 3
enc = ort.InferenceSession(f"{A}/encoder.onnx", so, providers=["CPUExecutionProvider"])
af = enc.run(None, {"mel": M[None]})[0][0]; del enc
emb = np.fromfile(f"{E}/model/embed/embed_tokens.int8.bin", np.int8).reshape(-1, 2048)
esc = np.fromfile(f"{E}/model/embed/embed_scales.f32.bin", np.float32)
nA = af.shape[0]
ids = P["prefix_ids"] + [P["audio_pad_id"]] * nA + P["suffix_ids"] + cfg["language_prefix_ids"]["ko"]
ie = emb[ids].astype(np.float32) * esc[ids][:, None]
at = len(P["prefix_ids"]); ie[at:at + nA] = af
init = ort.InferenceSession(f"{R}/decoder_init.q4f16.onnx", so, providers=["CPUExecutionProvider"])
names = [i.name for i in init.get_inputs()]
feed = {"input_embeds": ie[None].astype(np.float16 if "float16" in init.get_inputs()[0].type else np.float32),
        "position_ids": np.arange(len(ids), dtype=np.int64)[None]}
o = init.run(None, feed); del init
lg, pk, pv = o
step = ort.InferenceSession(f"{R}/decoder_step.q4f16.onnx", so, providers=["CPUExecutionProvider"])
Sx = len(ids); tok0 = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
pk.astype(np.float16).tofile(f"{out}/prefill_k.f16"); pv.astype(np.float16).tofile(f"{out}/prefill_v.f16")
gen = []; tok = tok0; pos = Sx; logits3 = []
t0 = time.time()
while tok not in P["eos_ids"] and len(gen) < 64:
    gen.append(tok)
    e = (emb[tok].astype(np.float32) * esc[tok]).astype(np.float16)
    n = step.run(None, {"input_embeds": e[None, None], "position_ids": np.array([[pos]], np.int64), "past_keys": pk, "past_values": pv})
    pos += 1; pk, pv = n[1], n[2]
    l = n[0].reshape(-1).astype(np.float32)
    if len(logits3) < 3: logits3.append(l)
    tok = int(l.argmax())
np.stack(logits3).tofile(f"{out}/logits3.f32")
vocab = json.load(open(f"{R}/vocab.json")); inv = {v: k for k, v in vocab.items()}
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n_ = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n_); n_ += 1
byte_of = {chr(c): b for b, c in zip(bs, cs)}
by = bytes(byte_of.get(ch, 63) for t in gen if t < 151643 and t in inv for ch in inv[t])
text = by.decode("utf-8", "replace"); text = text[text.rfind("<asr_text>") + 10:].strip() if "<asr_text>" in text else text.strip()
meta = {"S": Sx, "nAudio": nA, "tok0": tok0, "tokens": gen, "final": tok, "text": text, "kv_shape": list(pk.shape[:2]) + [8, Sx, 128],
        "init_input_dtype": init_type if (init_type := feed["input_embeds"].dtype.name) else None, "step_s": round(time.time() - t0, 1)}
json.dump(meta, open(f"{out}/truth.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(meta, ensure_ascii=False))
