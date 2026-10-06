#!/usr/bin/env node
// Numerical parity of the WebGPU captions pipeline against onnxruntime-node
// on the same Hugging Face export, on one real speech clip:
//   encoder   ours vs encoder.fp16.onnx, same log-mel input
//   prefill   engine.prefill vs decoder_init.q4f16.onnx last-position logits,
//             both fed ORT's audio features so only the decoder differs
//   tokens    full WebGPU greedy (our encoder -> prefill -> decode) vs the
//             ORT encoder -> decoder_init -> decoder_step greedy loop
//   hop1s     latency of one 1 s streaming hop on the same device
// The ORT reference runs in a child process; the WGSL half
// (asr-parity.browser.mjs) is bundled with rolldown and run in headless Chrome
// through agent-browser. Without a GPU, Chrome's WebGPU is SwiftShader on the
// CPU, so its latencies say nothing about a real GPU. (Dawn's Node bindings
// were tried first: on lavapipe under WSL even a trivial compute dispatch
// aborts in a Mesa thread, so they cannot host this check.)
//
//   node scripts/asr-parity.mjs <hf model dir> <16 kHz mono f32le pcm> [lang] [engine model dir]
import { execFile, execFileSync } from "node:child_process";
import { promisify } from "node:util";
import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";

const [dir, pcmPath, lang = "en", engineDir = resolve(import.meta.dirname, "../engine-dev/model")] = process.argv.slice(2).map((x, i) => (i === 2 ? x : resolve(x)));
if (!dir || !pcmPath) throw new Error("usage: node scripts/asr-parity.mjs <hf model dir> <pcm.f32> [lang] [engine model dir]");
const out = join(tmpdir(), "asr-parity");
const execFileAsync = promisify(execFile);
mkdirSync(out, { recursive: true });
const file = (name) => readFileSync(resolve(dir, name));
const H = 2048, MAX_NEW = 64;

if (process.env.PARITY_ORT_CHILD) await ortReference();
else await main();

// ---- ORT reference (child process): encoder -> decoder_init -> greedy decoder_step ----
async function ortReference() {
  // onnxruntime-node turns float16 data into a Float16Array when that global
  // exists, and its native binding then reads the array as empty ("not enough
  // space ... got 0"), so the global is hidden before ORT loads.
  const F16 = globalThis.Float16Array;
  delete globalThis.Float16Array;
  const ort = createRequire(import.meta.url)("onnxruntime-node");
  const { logMel } = await import("../src/captions/mel.ts");
  const cfg = JSON.parse(file("prompt_config.json").toString());
  const P = cfg.prompt;
  const pcm = new Float32Array(new Uint8Array(readFileSync(pcmPath)).buffer);
  const mel = logMel(pcm, JSON.parse(file("mel_filters.json").toString()).data);
  const emb = new Int8Array(new Uint8Array(file("embed_tokens.int8.bin")).buffer);
  const esc = new Float32Array(new Uint8Array(file("embed_scales.f32.bin")).buffer);
  const f16 = (a) => new Uint16Array(F16.from(a).buffer);
  const toF32 = (t) => (t.type === "float16" ? Float32Array.from(new F16(t.data.buffer, t.data.byteOffset, t.data.length)) : t.data);
  const argmax = (a) => { let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; };
  const embed = (ids, audio, audioAt) => {
    const e = new Float32Array(ids.length * H);
    ids.forEach((id, r) => { for (let d = 0; d < H; d++) e[r * H + d] = emb[id * H + d] * esc[id]; });
    e.set(audio, audioAt * H);
    return e;
  };
  const report = { audioSec: +(pcm.length / 16000).toFixed(2), melFrames: mel.T };

  let session = await ort.InferenceSession.create(file("encoder.fp16.onnx"));
  let t = performance.now();
  const features = (await session.run({ mel: new ort.Tensor("float32", mel.data, [1, 128, mel.T]) })).audio_features;
  report.ortEncoderMs = Math.round(performance.now() - t);
  await session.release();
  const A = features.dims[1];
  const ids = [...P.prefix_ids, ...Array(A).fill(P.audio_pad_id), ...P.suffix_ids, ...cfg.language_prefix_ids[lang]];
  const audioAt = P.prefix_ids.length;
  // Created from paths so ORT finds decoder_weights.q4f16.data beside them.
  session = await ort.InferenceSession.create(resolve(dir, "decoder_init.q4f16.onnx"));
  t = performance.now();
  const init = await session.run({
    input_embeds: new ort.Tensor("float16", f16(embed(ids, features.data, audioAt)), [1, ids.length, H]),
    position_ids: new ort.Tensor("int64", BigInt64Array.from(ids, (_, i) => BigInt(i)), [1, ids.length]),
  });
  report.ortPrefillMs = Math.round(performance.now() - t);
  await session.release();
  const logits = toF32(init.logits);
  const tokens = [];
  session = await ort.InferenceSession.create(resolve(dir, "decoder_step.q4f16.onnx"));
  let tok = argmax(logits), pos = ids.length, pk = init.present_keys, pv = init.present_values;
  t = performance.now();
  while (!P.eos_ids.includes(tok) && tokens.length < MAX_NEW) {
    tokens.push(tok);
    const step = await session.run({
      input_embeds: new ort.Tensor("float16", f16(embed([tok], new Float32Array(0), 0)), [1, 1, H]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos++)]), [1, 1]),
      past_keys: pk, past_values: pv,
    });
    pk = step.present_keys; pv = step.present_values;
    tok = argmax(toF32(step.logits));
  }
  report.ortDecodeMsPerToken = +((performance.now() - t) / Math.max(1, tokens.length)).toFixed(1);
  await session.release();
  writeFileSync(join(out, "mel.f32"), mel.data);
  writeFileSync(join(out, "features.f32"), features.data);
  writeFileSync(join(out, "logits.f32"), logits);
  writeFileSync(join(out, "ref.json"), JSON.stringify({ report, P, ids, audioAt, A, T: mel.T, H, maxNew: MAX_NEW, tokens }));
}

async function main() {
  // PARITY_REUSE_REF=1 reuses the last ORT reference in the output dir.
  if (!process.env.PARITY_REUSE_REF) execFileSync(process.execPath, [import.meta.filename, ...process.argv.slice(2)], { env: { ...process.env, PARITY_ORT_CHILD: "1" }, stdio: "inherit" });
  const ref = JSON.parse(readFileSync(join(out, "ref.json"), "utf8"));

  const { build } = await import("rolldown");
  await build({ input: resolve(import.meta.dirname, "asr-parity.browser.mjs"), output: { file: join(out, "parity.js"), format: "esm" }, logLevel: "warn" });
  writeFileSync(join(out, "index.html"), '<!doctype html><meta charset="utf-8"><title>asr parity</title><script type="module" src="parity.js"></script>');

  const roots = [["/m/", dir], ["/e/", engineDir], ["/", out]];
  const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
  const server = createServer((q, r) => {
    const url = decodeURIComponent(q.url.split("?")[0]);
    const [prefix, root] = roots.find(([p]) => url.startsWith(p));
    const f = join(root, url.slice(prefix.length) || "index.html");
    try {
      r.writeHead(200, { "content-type": types[extname(f)] || "application/octet-stream", "content-length": statSync(f).size });
      createReadStream(f).pipe(r);
    } catch { r.writeHead(404); r.end(); }
  });
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  const origin = `http://127.0.0.1:${server.address().port}/`;

  // ASR_PARITY_SESSION=<name> drives an already-open agent-browser session:
  // it only navigates that session and leaves it open afterwards, so a
  // rerun never launches or closes a browser.
  const reused = process.env.ASR_PARITY_SESSION;
  const session = reused || `asr-parity-${process.pid}`;
  // Asynchronous: this process also serves the page, and a synchronous
  // child would block the server until the navigation timed out.
  const ab = async (...args) => (await execFileAsync("agent-browser", ["--session", session, ...args], { encoding: "utf8" })).stdout.trim();
  try {
    await ab(...(reused ? [] : ["--args", "--enable-unsafe-webgpu,--enable-unsafe-swiftshader"]), "open", origin);
    let result;
    for (;;) {
      result = JSON.parse(JSON.parse(await ab("eval", "JSON.stringify(window.__parity ?? { done: false })")));
      if (result.done) break;
      await new Promise((ok) => setTimeout(ok, 5000));
    }
    console.log(JSON.stringify({ ...ref.report, ...result }));
    if (result.error) process.exitCode = 1;
  } finally {
    if (!reused) try { await ab("close"); } catch {}
    server.close();
  }
}
