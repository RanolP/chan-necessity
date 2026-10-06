// Browser half of scripts/asr-parity.mjs: runs the real WGSL encoder and
// decoder engine on the page's WebGPU device and compares them with the ORT
// reference files served beside this bundle. The result lands in
// window.__parity ({ done, error?, ...report }).
import { createEncoder } from "../src/captions/engine/encoder.ts";
import { createEngine } from "../src/captions/engine/engine.ts";

const bin = async (url) => new Float32Array(await (await fetch(url)).arrayBuffer());
const diff = (a, b) => {
  let max = 0, ref = 0;
  for (let i = 0; i < b.length; i++) { max = Math.max(max, Math.abs(a[i] - b[i])); ref = Math.max(ref, Math.abs(b[i])); }
  return { maxAbsDiff: max, maxAbsRef: ref };
};

async function run() {
  const ref = await (await fetch("ref.json")).json();
  const [mel, features, logits] = await Promise.all([bin("mel.f32"), bin("features.f32"), bin("logits.f32")]);
  const { P, ids, audioAt, A, T, H, maxNew, tokens: ortTokens } = ref;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const report = { adapter: `${adapter.info.vendor} ${adapter.info.architecture}`.trim(), shaderF16: adapter.features.has("shader-f16") };
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage },
  });
  const errors = [];
  device.addEventListener("uncapturederror", (e) => errors.push(e.error.message.slice(0, 300)));
  const readF32 = async (buffer, bytes) => {
    const m = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 0, m, 0, bytes); device.queue.submit([e.finish()]);
    await m.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(m.getMappedRange().slice(0)); m.unmap(); m.destroy();
    return r;
  };

  const encoder = await createEncoder(device, await (await fetch("/m/encoder.fp16.onnx")).arrayBuffer());
  (await encoder.encode(mel, T)).buffer.destroy(); // first run pays pipeline warm-up
  let t = performance.now();
  const ours = await encoder.encode(mel, T);
  report.gpuEncoderMs = Math.round(performance.now() - t);
  if (ours.frames !== A) throw new Error(`encoder frames: ORT ${A}, ours ${ours.frames}`);
  report.encoder = { frames: A, ...diff(await readF32(ours.buffer, A * H * 4), features) };

  const engine = await createEngine(device, {
    manifestUrl: "/e/manifest.json",
    urls: { "decoder_weights.q4f16.data": "/m/decoder_weights.q4f16.data", "qknorm.bin": "/e/qknorm.bin", embed: "/m/embed_tokens.int8.bin", embedScales: "/m/embed_scales.f32.bin" },
  });
  let pre = await engine.prefill(ids, features, { audioStart: audioAt, audioPadId: P.audio_pad_id });
  report.prefill = { ...diff(await readF32(pre.logits, logits.length * 4), logits), token: pre.token, ortToken: ortTokens[0] ?? null };

  t = performance.now();
  pre = await engine.prefill(ids, ours.buffer, { audioStart: audioAt, audioTokens: A, audioPadId: P.audio_pad_id });
  report.gpuPrefillMs = Math.round(performance.now() - t);
  const gpuTokens = [];
  if (!pre.done) {
    t = performance.now();
    const d = await engine.decode(pre.token, pre.position, { cap: maxNew });
    report.gpuDecodeMsPerToken = +((performance.now() - t) / Math.max(1, d.tokens.length)).toFixed(1);
    gpuTokens.push(pre.token, ...d.tokens.filter((x) => !P.eos_ids.includes(x)));
  }
  ours.buffer.destroy();
  report.tokens = { match: gpuTokens.length === ortTokens.length && gpuTokens.every((x, i) => x === ortTokens[i]), ort: ortTokens, gpu: gpuTokens };

  // One streaming hop: 1 s of audio (100 mel frames) through encoder,
  // prefill and a 16-token decode, as the worker runs it.
  const hopT = 100;
  const hopMel = new Float32Array(128 * hopT);
  for (let m = 0; m < 128; m++) hopMel.set(mel.subarray(m * T, m * T + hopT), m * hopT);
  const hopIds = [...ids.slice(0, audioAt), ...ids.slice(audioAt + A)];
  t = performance.now();
  const hop = await encoder.encode(hopMel, hopT);
  const tEnc = performance.now();
  hopIds.splice(audioAt, 0, ...Array(hop.frames).fill(P.audio_pad_id));
  const hp = await engine.prefill(hopIds, hop.buffer, { audioStart: audioAt, audioTokens: hop.frames, audioPadId: P.audio_pad_id });
  const tPre = performance.now();
  const hd = hp.done ? { tokens: [] } : await engine.decode(hp.token, hp.position, { cap: 16 });
  const tEnd = performance.now();
  hop.buffer.destroy();
  report.hop1s = { encodeMs: Math.round(tEnc - t), prefillMs: Math.round(tPre - tEnc), decodeMs: Math.round(tEnd - tPre), tokens: hd.tokens.length + 1, totalMs: Math.round(tEnd - t) };
  report.webgpuErrors = errors;
  return report;
}

window.__parity = { done: false };
run().then(
  (r) => { window.__parity = { done: true, ...r }; },
  (e) => { window.__parity = { done: true, error: String(e?.stack || e) }; },
);
