# Emits manifest.json mapping each fused GPU tensor to segments of the
# upstream decoder_weights.q4f16.data (the userscript already caches that file),
# plus qknorm.bin for the inline q/k norms.
# Fused per layer: qkv rows (q|k|v), gate|up rows, o, down.
# q4 layout kept from MatMulNBits: q [N, K/32, 16B], scale f16 [N, K/32], zp u8 [N, K/64].
#
# Reads only two small files of the Hugging Face export; segment offsets come
# from the external_data fields of decoder_step.q4f16.onnx, so the 0.3-1 GB
# weights file is never needed here.
# Run: uv run --with onnx --with numpy python convert.py <dir> <repo>@<rev> <out_dir>
#   <dir> holds decoder_step.q4f16.onnx and config.json of that revision.
import json, sys, os, re
import numpy as np, onnx
from onnx import numpy_helper

src_dir, source, out_dir = sys.argv[1:4]
m = onnx.load(f"{src_dir}/decoder_step.q4f16.onnx", load_external_data=False)
dc = json.load(open(f"{src_dir}/config.json"))["decoder"]
WEIGHTS = "decoder_weights.q4f16.data"

ext = {}
inline = {}
for i in m.graph.initializer:
    if i.data_location == onnx.TensorProto.EXTERNAL:
        e = {kv.key: kv.value for kv in i.external_data}
        if e["location"] != WEIGHTS: raise SystemExit(f"{i.name}: external data in {e['location']}, expected {WEIGHTS}")
        ext[i.name] = (int(e["offset"]), int(e["length"]))
    else:
        inline[i.name] = i

# Every projection is one MatMulNBits node; the exporter numbers them in
# forward order, linear_{7L+j} for j = q, k, v, o, gate, up, down, and the
# LM head is the node writing `logits`.
weight_of = {}
lm_base = None
for n in m.graph.node:
    if n.op_type != "MatMulNBits": continue
    base = n.input[1].removesuffix("_Q4G32")
    if n.output[0] == "logits": lm_base = base; continue
    g = re.fullmatch(r"linear(?:_(\d+))?", n.output[0])
    if not g: raise SystemExit(f"unexpected MatMulNBits output {n.output[0]}")
    weight_of[int(g.group(1) or 0)] = base
NL = dc["num_layers"]
if len(weight_of) != 7 * NL or lm_base is None: raise SystemExit(f"found {len(weight_of)} projections + lm={lm_base}, expected {7 * NL} + 1")

os.makedirs(out_dir, exist_ok=True)
blob = open(f"{out_dir}/qknorm.bin", "wb")
man = {"tensors": {}}
pos = 0

def seg(name):
    return list(ext[name])

def fused(name, names):
    segs = [seg(n) for n in names]
    man["tensors"][name] = {"src": WEIGHTS, "segments": segs, "length": sum(l for _, l in segs)}

def put(name, arr):  # into qknorm.bin
    global pos
    b = np.ascontiguousarray(arr).view(np.uint8).tobytes()
    pad = (-pos) % 256
    blob.write(b"\0" * pad); pos += pad
    man["tensors"][name] = {"src": "qknorm.bin", "segments": [[pos, len(b)]], "length": len(b)}
    blob.write(b); pos += len(b)

def q4(tag, bases):
    for suf, k in (("_Q4G32", "q"), ("_scale", "s"), ("_zp", "z")):
        fused(f"{tag}.{k}", [f"{b}{suf}" for b in bases])

for L in range(NL):
    q, k, v, o, gate, up, down = (weight_of[7 * L + j] for j in range(7))
    for tag, bases in (("qkv", [q, k, v]), ("o", [o]), ("gu", [gate, up]), ("down", [down])):
        q4(f"l{L}.{tag}", bases)
    fused(f"l{L}.ln1", [f"layers.{L}.input_layernorm.weight"])
    fused(f"l{L}.ln2", [f"layers.{L}.post_attention_layernorm.weight"])
    put(f"l{L}.qn", numpy_helper.to_array(inline[f"layers.{L}.self_attn.q_norm.weight"]).astype(np.float16))
    put(f"l{L}.kn", numpy_helper.to_array(inline[f"layers.{L}.self_attn.k_norm.weight"]).astype(np.float16))
fused("norm", ["norm.weight"])
q4("lm", [lm_base])
blob.close()
man["qknorm_bytes"] = pos
hd = dc["head_dim"]
man["config"] = {"layers": NL, "hidden": dc["hidden_size"], "heads": dc["num_attention_heads"], "kv_heads": dc["num_key_value_heads"], "head_dim": hd,
                 "intermediate": dc["intermediate_size"], "vocab": dc["vocab_size"], "rope_theta": float(dc["rope_theta"]), "rms_eps": float(dc["rms_norm_eps"]),
                 "attn_scale_f16": float(np.float16(1 / np.sqrt(hd))), "eos": json.load(open(f"{src_dir}/config.json"))["special_tokens"]["eos_token_ids"], "block": 32,
                 "source": source}
json.dump(man, open(f"{out_dir}/manifest.json", "w"), indent=1)
print("bytes", pos, "tensors", len(man["tensors"]))
