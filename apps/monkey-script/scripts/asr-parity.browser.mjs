// Browser half of scripts/asr-parity.mjs: runs the real WGSL encoder and
// decoder engine on the page's WebGPU device and compares them with the frozen
// ORT reference files served beside this bundle. The result lands in
// window.__parity ({ done, error?, ...report }).
//
// ?stage=encoder  encoder only, against ORT's audio features
// ?stage=decoder  prefill + greedy decode fed ORT's audio features, so the
//                 tokens compare the decoder alone (the encoder is never loaded)
// (none)          the full pipeline: encoder diff, prefill diff, end-to-end
//                 tokens and one 1 s streaming hop
// Each stage holds only its own weights, so a device that dies under memory
// pressure names the stage in window.__parityLog, together with the bytes
// allocated and uploaded so far and the device-lost reason.
import { createEncoder } from "../src/captions/engine/encoder.ts";
import { createEngine } from "../src/captions/engine/engine.ts";

const log = (window.__parityLog = []);
const note = (msg) => log.push(`[${(performance.now() / 1000).toFixed(1)}s] ${msg}`);
const bin = async (url) => new Float32Array(await (await fetch(url)).arrayBuffer());
const diff = (a, b) => {
  let max = 0, sum = 0, ref = 0;
  for (let i = 0; i < b.length; i++) { const d = Math.abs(a[i] - b[i]); max = Math.max(max, d); sum += d; ref = Math.max(ref, Math.abs(b[i])); }
  return { maxAbsDiff: max, meanAbsDiff: sum / b.length, maxAbsRef: ref };
};

async function run() {
  const stage = new URLSearchParams(location.search).get("stage") ?? "all";
  const ref = await (await fetch("ref.json")).json();
  const [mel, features, logits] = await Promise.all([bin("mel.f32"), bin("features.f32"), bin("logits.f32")]);
  const { P, ids, audioAt, A, T, H, maxNew, tokens: ortTokens } = ref;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  // A software adapter (SwiftShader, lavapipe) runs the model on the CPU: minutes per load,
  // every core pegged, and timings that say nothing about the GPU path. Refuse it outright.
  const gpuId = `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.device} ${adapter.info.description}`;
  if (adapter.info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test(gpuId))
    throw new Error(`software WebGPU adapter (${gpuId.trim()}); run on the Windows Chrome GPU over CDP 9333`);
  const report = { stage, adapter: `${adapter.info.vendor} ${adapter.info.architecture}`.trim(), shaderF16: adapter.features.has("shader-f16") };
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const bytes = { allocated: 0, uploaded: 0 };
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (d) => { bytes.allocated += d.size; return createBuffer(d); };
  const writeBuffer = device.queue.writeBuffer.bind(device.queue);
  device.queue.writeBuffer = (b, off, data, dataOff = 0, size) => {
    bytes.uploaded += size ?? ((data.byteLength ?? data.length * (data.BYTES_PER_ELEMENT ?? 1)) - dataOff * (data.BYTES_PER_ELEMENT ?? 1));
    return writeBuffer(b, off, data, dataOff, size);
  };
  const mib = (n) => `${(n / 2 ** 20).toFixed(0)} MiB`;
  const step = (name) => note(`stage ${name}: allocated ${mib(bytes.allocated)}, uploaded ${mib(bytes.uploaded)}`);
  let lost = null;
  device.lost.then((info) => { lost = { reason: info.reason, message: info.message }; note(`device lost: reason=${info.reason} message=${info.message}`); });
  const errors = [];
  device.addEventListener("uncapturederror", (e) => errors.push(e.error.message.slice(0, 300)));
  const readF32 = async (buffer, n) => {
    const m = device.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 0, m, 0, n); device.queue.submit([e.finish()]);
    await m.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(m.getMappedRange().slice(0)); m.unmap(); m.destroy();
    return r;
  };
  const guard = async (name, f) => {
    step(name);
    try { return await f(); }
    catch (e) { throw new Error(`stage ${name} failed (allocated ${mib(bytes.allocated)}, uploaded ${mib(bytes.uploaded)}, lost ${JSON.stringify(lost)}): ${e?.stack || e}`); }
  };
  const finish = () => { step("done"); report.bytes = bytes; report.deviceLost = lost; report.webgpuErrors = errors; return report; };

  let encoder = null, ours = null;
  if (stage !== "decoder") {
    encoder = await guard("encoder-load", async () => createEncoder(device, await (await fetch("/m/encoder.fp16.onnx")).arrayBuffer()));
    await guard("encoder-run", async () => {
      (await encoder.encode(mel, T)).buffer.destroy(); // first run pays pipeline warm-up
      const t = performance.now();
      ours = await encoder.encode(mel, T);
      report.gpuEncoderMs = Math.round(performance.now() - t);
      if (ours.frames !== A) throw new Error(`encoder frames: ORT ${A}, ours ${ours.frames}`);
      report.encoder = { frames: A, ...diff(await readF32(ours.buffer, A * H * 4), features) };
    });
    if (stage === "encoder") { ours.buffer.destroy(); return finish(); }
  }

  const engine = await guard("decoder-load", () => createEngine(device, {
    manifestUrl: "/e/manifest.json",
    urls: { "decoder_weights.q4f16.data": "/m/decoder_weights.q4f16.data", "qknorm.bin": "/e/qknorm.bin", embed: "/m/embed_tokens.int8.bin", embedScales: "/m/embed_scales.f32.bin" },
  }));
  let pre = await guard("prefill-ort-features", async () => {
    const t = performance.now();
    const p = await engine.prefill(ids, features, { audioStart: audioAt, audioPadId: P.audio_pad_id });
    report.prefill = { ...diff(await readF32(p.logits, logits.length * 4), logits), token: p.token, ortToken: ortTokens[0] ?? null, ms: Math.round(performance.now() - t) };
    return p;
  });

  if (stage === "decoder") {
    // ORT's greedy loop also started from ORT's features, so this compares decoders only.
    const tokens = await guard("decode", async () => {
      const out = [];
      if (!pre.done) {
        const t = performance.now();
        const d = await engine.decode(pre.token, pre.position, { cap: maxNew });
        report.gpuDecodeMsPerToken = +((performance.now() - t) / Math.max(1, d.tokens.length)).toFixed(1);
        out.push(pre.token, ...d.tokens.filter((x) => !P.eos_ids.includes(x)));
      }
      return out;
    });
    report.tokens = { match: tokens.length === ortTokens.length && tokens.every((x, i) => x === ortTokens[i]), ort: ortTokens, gpu: tokens };
    return finish();
  }

  const gpuTokens = await guard("end-to-end", async () => {
    let t = performance.now();
    pre = await engine.prefill(ids, ours.buffer, { audioStart: audioAt, audioTokens: A, audioPadId: P.audio_pad_id });
    report.gpuPrefillMs = Math.round(performance.now() - t);
    const out = [];
    if (!pre.done) {
      t = performance.now();
      const d = await engine.decode(pre.token, pre.position, { cap: maxNew });
      report.gpuDecodeMsPerToken = +((performance.now() - t) / Math.max(1, d.tokens.length)).toFixed(1);
      out.push(pre.token, ...d.tokens.filter((x) => !P.eos_ids.includes(x)));
    }
    return out;
  });
  ours.buffer.destroy();
  report.tokens = { match: gpuTokens.length === ortTokens.length && gpuTokens.every((x, i) => x === ortTokens[i]), ort: ortTokens, gpu: gpuTokens };

  // One streaming hop: 1 s of audio (100 mel frames) through encoder,
  // prefill and a 16-token decode, as the worker runs it.
  await guard("hop1s", async () => {
    const hopT = 100;
    const hopMel = new Float32Array(128 * hopT);
    for (let m = 0; m < 128; m++) hopMel.set(mel.subarray(m * T, m * T + hopT), m * hopT);
    const hopIds = [...ids.slice(0, audioAt), ...ids.slice(audioAt + A)];
    const t = performance.now();
    const hop = await encoder.encode(hopMel, hopT);
    const tEnc = performance.now();
    hopIds.splice(audioAt, 0, ...Array(hop.frames).fill(P.audio_pad_id));
    const hp = await engine.prefill(hopIds, hop.buffer, { audioStart: audioAt, audioTokens: hop.frames, audioPadId: P.audio_pad_id });
    const tPre = performance.now();
    const hd = hp.done ? { tokens: [] } : await engine.decode(hp.token, hp.position, { cap: 16 });
    const tEnd = performance.now();
    hop.buffer.destroy();
    report.hop1s = { encodeMs: Math.round(tEnc - t), prefillMs: Math.round(tPre - tEnc), decodeMs: Math.round(tEnd - tPre), tokens: hd.tokens.length + 1, totalMs: Math.round(tEnd - t) };
  });
  return finish();
}

window.__parity = { done: false };
run().then(
  (r) => { window.__parity = { done: true, ...r }; },
  (e) => { window.__parity = { done: true, error: String(e?.stack || e) }; },
);
