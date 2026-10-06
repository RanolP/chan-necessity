#!/usr/bin/env node
// Numerical parity of the WebGPU captions pipeline against the ORT
// reference frozen in fixtures/asr-parity-<lang>/ (ORT CPU on the same Hugging
// Face export, on one real speech clip):
//   encoder   ours vs ORT's encoder.fp16.onnx audio features, same log-mel input
//   prefill   engine.prefill vs ORT's decoder_init.q4f16.onnx last-position
//             logits, both fed ORT's audio features so only the decoder differs
//   tokens    full WebGPU greedy (our encoder -> prefill -> decode) vs ORT's
//             encoder -> decoder_init -> decoder_step greedy tokens
//   hop1s     latency of one 1 s streaming hop on the same device
// The fixture holds the clip's sha256; the clip itself is not committed. The
// WGSL half (asr-parity.browser.mjs) is bundled with rolldown and run in
// headless Chrome through agent-browser. Without a GPU, Chrome's WebGPU is
// SwiftShader on the CPU, so its latencies say nothing about a real GPU.
// (Dawn's Node bindings were tried first: on lavapipe under WSL even a trivial
// compute dispatch aborts in a Mesa thread, so they cannot host this check.)
//
//   node scripts/asr-parity.mjs <hf model dir> <16 kHz mono f32le pcm> [lang] [engine model dir]
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { promisify } from "node:util";
import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";

const [dir, pcmPath, lang = "en", engineDir = resolve(import.meta.dirname, "../engine-dev/model")] = process.argv.slice(2).map((x, i) => (i === 2 ? x : resolve(x)));
if (!dir || !pcmPath) throw new Error("usage: node scripts/asr-parity.mjs <hf model dir> <pcm.f32> [lang] [engine model dir]");
const out = join(tmpdir(), "asr-parity");
const execFileAsync = promisify(execFile);
mkdirSync(out, { recursive: true });
const fixture = resolve(import.meta.dirname, `fixtures/asr-parity-${lang}`);
await main();

// Writes what the browser half fetches: ref.json, mel.f32 (our log-mel of the
// clip, the encoder input on both sides), features.f32 and logits.f32 (ORT's).
async function stageReference() {
  const ref = JSON.parse(readFileSync(join(fixture, "ref.json"), "utf8"));
  const raw = readFileSync(pcmPath);
  const sha = createHash("sha256").update(raw).digest("hex");
  if (sha !== ref.source.pcmSha256) throw new Error(`${pcmPath}: sha256 ${sha}, the ${lang} reference was frozen from ${ref.source.pcmSha256}`);
  const { logMel } = await import("../src/captions/mel.ts");
  const pcm = new Float32Array(raw.buffer, raw.byteOffset, raw.length >> 2);
  const mel = logMel(pcm, JSON.parse(readFileSync(resolve(dir, "mel_filters.json"), "utf8")).data);
  if (mel.T !== ref.T) throw new Error(`log-mel frames: reference ${ref.T}, ours ${mel.T}`);
  const l16 = readFileSync(join(fixture, "logits.f16"));
  writeFileSync(join(out, "mel.f32"), mel.data);
  writeFileSync(join(out, "features.f32"), readFileSync(join(fixture, "features.f32")));
  writeFileSync(join(out, "logits.f32"), Float32Array.from(new Float16Array(l16.buffer, l16.byteOffset, l16.length >> 1)));
  writeFileSync(join(out, "ref.json"), JSON.stringify(ref));
  return ref;
}

async function main() {
  const ref = await stageReference();

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
    // ASR_PARITY_STAGE=encoder|decoder runs one stage alone (see asr-parity.browser.mjs).
    const stage = process.env.ASR_PARITY_STAGE;
    await ab(...(reused ? [] : ["--args", "--enable-unsafe-webgpu"]), "open", stage ? `${origin}?stage=${stage}` : origin);
    let result, printed = 0;
    for (;;) {
      result = JSON.parse(JSON.parse(await ab("eval", "JSON.stringify({ ...(window.__parity ?? { done: false }), log: window.__parityLog ?? [] })")));
      // Stream the page's stage log so a lost device still leaves the last stage on stdout.
      for (const line of result.log.slice(printed)) console.error(line);
      printed = result.log.length;
      if (result.done) break;
      await new Promise((ok) => setTimeout(ok, 5000));
    }
    delete result.log;
    console.log(JSON.stringify({ ...ref.report, ...result }));
    if (result.error) process.exitCode = 1;
  } finally {
    if (!reused) try { await ab("close"); } catch {}
    server.close();
  }
}
