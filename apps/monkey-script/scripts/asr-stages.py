# Per-stage ORT reference for the WebGPU decoder, written beside the last
# asr-parity.mjs reference so asr-parity.browser.mjs (?stage=stages) can
# compare each op boundary instead of only the final logits.
#
#   uv run --with onnx --with onnxruntime --with numpy \
#     scripts/asr-stages.py <hf model dir> [out dir (default $TMPDIR/asr-parity)]
#
# It feeds decoder_init.q4f16.onnx the exact input_embeds the harness built
# (ref.json ids + features.f32, rounded to f16), exposes the intermediate
# tensors of layers 0, 1 and the last as extra graph outputs, then runs
# decoder_step for STEPS tokens teacher-forced with ORT's own greedy tokens.
# Every tensor lands in <out>/stages/<name>.f32 as [rows, width] f32, listed
# in stages.json.
import json, os, sys, tempfile
import numpy as np, onnx, onnxruntime as ort

STEPS = 5
dir_ = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(tempfile.gettempdir(), "asr-parity")
ref = json.load(open(f"{out}/ref.json"))
ids, at, H = ref["ids"], ref["audioAt"], ref["H"]
feat = np.fromfile(f"{out}/features.f32", np.float32).reshape(-1, H)
emb = np.memmap(f"{dir_}/embed_tokens.int8.bin", np.int8, "r").reshape(-1, H)
esc = np.fromfile(f"{dir_}/embed_scales.f32.bin", np.float32)
embed = lambda toks: emb[toks].astype(np.float32) * esc[toks][:, None]
ie = embed(ids); ie[at:at + len(feat)] = feat
ie = ie.astype(np.float16)
S = len(ids)

m = onnx.load(f"{dir_}/decoder_init.q4f16.onnx", load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
def only(xs, what):
    if len(xs) != 1: raise SystemExit(f"{what}: expected one node, got {[x.name for x in xs]}")
    return xs[0]
def after(t, op): return only([n for n in cons.get(t, []) if n.op_type == op], f"{op} after {t}")
def norm_by_weight(w): return only([n for n in g.node if n.op_type == "SimplifiedLayerNormalization" and n.input[1] == w], w)
def rope_of(normed):
    # normed -> Transpose -> Mul(cos) ; Add(Mul(cos), Mul(rotate_half, sin))
    tr = after(normed, "Transpose").output[0]
    mul = only([n for n in cons[tr] if n.op_type == "Mul"], f"cos mul of {tr}").output[0]
    return after(mul, "Add").output[0]

NL = sum(1 for n in g.node if n.op_type == "SimplifiedLayerNormalization" and n.input[1].endswith("input_layernorm.weight"))
taps = {}  # name -> (tensor, layout)
for L in (0, 1, NL - 1):
    rms1 = norm_by_weight(f"layers.{L}.input_layernorm.weight").output[0]
    qn = norm_by_weight(f"layers.{L}.self_attn.q_norm.weight"); kn = norm_by_weight(f"layers.{L}.self_attn.k_norm.weight")
    lin_of = lambda nn: prod[prod[nn.input[0]].input[0]].output[0]  # Reshape <- MatMulNBits
    qlin, klin = lin_of(qn), lin_of(kn)
    vlin = only([n for n in cons[rms1] if n.op_type == "MatMulNBits" and n.output[0] not in (qlin, klin)], "v proj").output[0]
    rms2n = norm_by_weight(f"layers.{L}.post_attention_layernorm.weight")
    resid1 = rms2n.input[0]
    oproj = [i for i in prod[resid1].input if prod.get(i) is not None and prod[i].op_type == "MatMulNBits"][0]
    attn = prod[oproj].input[0]
    rms2 = rms2n.output[0]
    gate_up = [n for n in cons[rms2] if n.op_type == "MatMulNBits"]
    sig = [n for n in gate_up if any(c.op_type == "Sigmoid" for c in cons[n.output[0]])][0]
    act = after(after(sig.output[0], "Mul").output[0], "Mul").output[0]
    down = after(act, "MatMulNBits").output[0]
    resid2 = after(down, "Add").output[0]
    for k, t in dict(rms1=rms1, q=qlin, k=klin, v=vlin, qnorm=qn.output[0], knorm=kn.output[0], qrope=rope_of(qn.output[0]),
                     krope=rope_of(kn.output[0]), attn=attn, oproj=oproj, resid1=resid1, rms2=rms2, act=act, down=down, resid2=resid2).items():
        taps[f"l{L}.{k}"] = t
final = norm_by_weight("norm.weight").output[0]
taps["final_norm"] = final
have = {o.name for o in g.output}
for t in taps.values():
    if t not in have: g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT16, None)); have.add(t)
tapped = f"{dir_}/decoder_init.stages.onnx"
onnx.save(m, tapped)

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(tapped, so, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
res = dict(zip(names, sess.run(None, {"input_embeds": ie[None], "position_ids": np.arange(S, dtype=np.int64)[None]})))
os.makedirs(f"{out}/stages", exist_ok=True)
index = {}
def save(name, a):
    a = np.asarray(a, np.float32)
    a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a[None]
    a.tofile(f"{out}/stages/{name}.f32"); index[name] = list(a.shape)
save("embed", ie)
for name, t in taps.items():
    a = res[t].astype(np.float32)
    # [1, heads, S, 128] (post-transpose rope) -> [S, heads*128]; [1, S, heads, 128] -> [S, heads*128]
    if a.ndim == 4 and a.shape[2] == S: a = a[0].transpose(1, 0, 2)
    save(name, a.reshape(S, -1))
logits = res["logits"].astype(np.float32).reshape(-1, res["logits"].shape[-1])
save("logits", logits)
pk, pv = res["present_keys"], res["present_values"]
for L in (0, 1, NL - 1):
    save(f"l{L}.kcache", pk[L, 0].transpose(1, 0, 2).reshape(S, -1).astype(np.float32))
    save(f"l{L}.vcache", pv[L, 0].transpose(1, 0, 2).reshape(S, -1).astype(np.float32))
del sess

step = ort.InferenceSession(f"{dir_}/decoder_step.q4f16.onnx", so, providers=["CPUExecutionProvider"])
toks = ref["tokens"]
for i in range(min(STEPS, len(toks))):
    o = step.run(None, {"input_embeds": embed([toks[i]]).astype(np.float16)[None], "position_ids": np.array([[S + i]], np.int64), "past_keys": pk, "past_values": pv})
    pk, pv = o[1], o[2]
    save(f"step{i}.logits", o[0].astype(np.float32).reshape(-1))
json.dump({"S": S, "layers": [0, 1, NL - 1], "steps": min(STEPS, len(toks)), "shapes": index, "tensors": taps}, open(f"{out}/stages.json", "w"), indent=1)
print(json.dumps({"S": S, "n": len(index), "logitsLastArgmax": int(logits[-1].argmax()), "refTok0": toks[0]}))
