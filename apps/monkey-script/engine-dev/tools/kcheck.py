# Layer-0 k only: isolates qk-norm + rope from the q4 gemv (v already matches).
import sys; sys.argv=['x']
exec(open('tools/ref.py').read().split('if __name__')[0])
import onnxruntime as ort
emb = np.fromfile(f"{E}/embed_tokens.int8.bin", np.int8); esc = np.fromfile(f"{E}/embed_scales.f32.bin", np.float32)
tok=151704; e=(emb[tok*2048:(tok+1)*2048].astype(np.float32)*esc[tok]).astype(np.float16)
sess = ort.InferenceSession(f"{E}/tools/refmodel/decoder_step.q4f16.onnx", providers=["CPUExecutionProvider"])
z=np.zeros((28,1,8,0,128),np.float16)
for pos in (0, 5, 40):
    o=sess.run(None,{"input_embeds":e[None,None],"position_ids":np.array([[pos]],np.int64),"past_keys":z,"past_values":z})
    x=rms(e.astype(np.float32),f16("l0.ln1")); qkv=gemv("l0.qkv",x,2048); k=rms(qkv[2048:3072].reshape(8,128),f16("l0.kn"))
    ok=o[1][0,0,:,0].astype(np.float32)
    print(pos,"|k|",float(np.abs(k).max()),"rotate_half",float(np.abs(rope(k,pos)-ok).max()),"norope",float(np.abs(k-ok).max()))
