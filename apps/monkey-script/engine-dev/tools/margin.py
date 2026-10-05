# ORT-CPU top-2 margin at a given output index of the greedy sequence in test/truth.json.
import json, os, sys, numpy as np, onnxruntime as ort
S = os.environ["S"]; E = f"{S}/engine"; idx = int(sys.argv[1])
t = json.load(open(f"{E}/test/truth.json")); toks = t["tokens"] + [t["final"]]
emb = np.fromfile(f"{E}/model/embed/embed_tokens.int8.bin", np.int8).reshape(-1, 2048); esc = np.fromfile(f"{E}/model/embed/embed_scales.f32.bin", np.float32)
so = ort.SessionOptions(); so.log_severity_level = 3
st = ort.InferenceSession(f"{E}/tools/refmodel/decoder_step.q4f16.onnx", so, providers=["CPUExecutionProvider"])
pk = np.fromfile(f"{E}/test/prefill_k.f16", np.float16).reshape(28, 1, 8, t["S"], 128); pv = np.fromfile(f"{E}/test/prefill_v.f16", np.float16).reshape(pk.shape)
for i in range(idx):
    e = (emb[toks[i]].astype(np.float32) * esc[toks[i]]).astype(np.float16)
    l, pk, pv = st.run(None, {"input_embeds": e[None, None], "position_ids": np.array([[t["S"] + i]], np.int64), "past_keys": pk, "past_values": pv})
l = l.reshape(-1).astype(np.float32); o = np.argsort(-l)[:3]
print(json.dumps({"idx": idx, "top3": [(int(k), float(l[k])) for k in o], "margin12": float(l[o[0]] - l[o[1]]), "logit17": float(l[17]), "logit21": float(l[21])}))
