# Emits manifest.json mapping each fused GPU tensor to segments of the
# upstream decoder_weights.q4f16.data (the userscript already caches that file,
# and /tmp has no room for a 1 GB copy), plus qknorm.bin for the inline norms.
# Fused per layer: qkv rows (q|k|v, N=4096), gate|up rows (N=12288), o, down.
# q4 layout kept from MatMulNBits: q [N, K/32, 16B], scale f16 [N, K/32], zp u8 [N, K/64].
import json, sys, os
import numpy as np, onnx
from onnx import numpy_helper

S = os.environ["S"]
data = np.memmap(f"{S}/fusion/decoder_weights.q4f16.data", dtype=np.uint8, mode="r")
wm = json.load(open(f"{S}/fusion/weights_map.json"))
ext = wm["external"]
m = onnx.load(f"{S}/asr/decoder_step.q4f16.onnx", load_external_data=False)
inline = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer if i.name in set(wm["inline"])}
out_dir = f"{S}/engine/model"
os.makedirs(out_dir, exist_ok=True)
blob = open(f"{out_dir}/qknorm.bin", "wb")
man = {"tensors": {}}
pos = 0

def seg(name):
    e = ext[name]
    return [e["offset"], e["length"]]

def fused(name, names):
    segs = [seg(n) for n in names]
    man["tensors"][name] = {"src": "decoder_weights.q4f16.data", "segments": segs, "length": sum(l for _, l in segs)}

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

for L in range(28):
    r = wm["roles"][str(L)]
    for tag, bases in (("qkv", [r["q"], r["k"], r["v"]]), ("o", [r["o"]]), ("gu", [r["gate"], r["up"]]), ("down", [r["down"]])):
        q4(f"l{L}.{tag}", bases)
    fused(f"l{L}.ln1", [f"layers.{L}.input_layernorm.weight"])
    fused(f"l{L}.ln2", [f"layers.{L}.post_attention_layernorm.weight"])
    put(f"l{L}.qn", inline[f"layers.{L}.self_attn.q_norm.weight"].astype(np.float16))
    put(f"l{L}.kn", inline[f"layers.{L}.self_attn.k_norm.weight"].astype(np.float16))
fused("norm", ["norm.weight"])
q4("lm", ["val_4460"])
blob.close()
man["qknorm_bytes"] = pos
man["config"] = {"layers": 28, "hidden": 2048, "heads": 16, "kv_heads": 8, "head_dim": 128,
                 "intermediate": 6144, "vocab": 151936, "rope_theta": 1e6, "rms_eps": 1e-6,
                 "attn_scale_f16": 0.08837890625, "eos": [151643, 151645], "block": 32,
                 "source": "jiangzhuo9357/Qwen3-ASR-1.7B-ONNX@fcc238dfdc95cdcccaa9a7e2c7f5abc2f94f44a7"}
json.dump(man, open(f"{out_dir}/manifest.json", "w"), indent=1)
print("bytes", pos, "tensors", len(man["tensors"]))
