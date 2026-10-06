#!/usr/bin/env node
// Sliding prefix KV against full recompute, on the streaming caption path.
// The same worker bundle (this tree) runs the same 1 s hops twice: once as
// shipped (prefix KV reused across hops, the oldest block evicted in place by
// engine.shiftKV, recomputed whole at a quiet hop), once with exactKv (a full
// prefill every hop). VAD is off. Per hop it compares the confirmed token ids
// (all but the last ROLLBACK) and reports where they first part.
//
//   node scripts/asr-kv-drift.mjs <hf model dir> <16 kHz mono f32le pcm> [lang]
//
// KV_DRIFT_QUIET=none (default) never flags a hop quiet, so no pause ever
// rebuilds the prefix and the K-shift drift accumulates over the whole clip;
// KV_DRIFT_QUIET=rms flags hops as captions/index.ts does, so pauses rebuild.
// Hops before the first slide, and after a rebuild until the next slide,
// reuse an exactly computed prefix and must match token for token.
// Chrome is reached over CDP at KV_DRIFT_CDP (default http://127.0.0.1:9444):
// a separate instance with --enable-unsafe-webgpu and its own
// --user-data-dir, never the live caption Chrome on 9333.
import { execFileSync } from "node:child_process";
import { createReadStream, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";

const [modelDir, pcmPath, lang = "ko"] = process.argv.slice(2).map((x, i) => (i < 2 ? resolve(x) : x));
if (!modelDir || !pcmPath) throw new Error("usage: node scripts/asr-kv-drift.mjs <hf model dir> <pcm.f32> [lang]");
const here = resolve(import.meta.dirname, "..");
const QUIET = process.env.KV_DRIFT_QUIET || "none";
const out = join(tmpdir(), `asr-kv-drift-${QUIET}`);
mkdirSync(out, { recursive: true });
const CDP = process.env.KV_DRIFT_CDP || "http://127.0.0.1:9444";
if (new URL(CDP).port === "9333") throw new Error("refusing CDP port 9333 (the live caption Chrome); launch a separate instance");
const MIN_FREE_MIB = Number(process.env.KV_DRIFT_MIN_FREE_MIB || 3500);
const SR = 16000, HOP = SR, ROLLBACK = 5, SILENCE_RMS = 0.004;
const REPO = "jiangzhuo9357/Qwen3-ASR-1.7B-ONNX", REV = "fcc238dfdc95cdcccaa9a7e2c7f5abc2f94f44a7";

// Hops pre-scaled with the page's slow-peak gain (index.ts).
const raw = readFileSync(pcmPath);
const audio = new Float32Array(raw.buffer, raw.byteOffset, raw.length >> 2);
const nHops = Math.floor(audio.length / HOP);
const hopsPcm = new Float32Array(nHops * HOP);
const hops = [];
let peakState = 0;
for (let i = 0; i < nHops; i++) {
    const pcm = audio.slice(i * HOP, (i + 1) * HOP);
    let sum = 0, peak = 0;
    for (const v of pcm) { sum += v * v; peak = Math.max(peak, Math.abs(v)); }
    peakState = Math.max(peak, peakState * 0.9);
    const gain = Math.min(20, 0.5 / Math.max(peakState, 1e-4));
    for (let j = 0; j < pcm.length; j++) pcm[j] *= gain;
    hopsPcm.set(pcm, i * HOP);
    hops.push({ quiet: QUIET === "rms" && Math.sqrt(sum / pcm.length) < SILENCE_RMS });
}
writeFileSync(join(out, "hops.f32"), hopsPcm);
writeFileSync(join(out, "hops.json"), JSON.stringify({ repo: REPO, rev: REV, lang, hopSamples: HOP, maxTokens: Math.round(16 + 12 * (HOP / SR)), hops }));

const prelude = `{
  const f = self.fetch.bind(self), o = self.location.origin;
  const hf = "https://huggingface.co/${REPO}/resolve/${REV}/";
  self.fetch = (input, init) => {
    const u = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (u.startsWith("https://huggingface.co/api/")) return Promise.resolve(new Response("offline", { status: 404 }));
    return f(u.startsWith(hf) ? o + "/m/" + u.slice(hf.length) : u.endsWith("/silero_vad_v5.onnx") ? o + "/m/silero_vad_v5.onnx" : u, init);
  };
}
`;
const { build } = await import("rolldown");
const built = await build({ input: join(here, "src/captions/worker/main.ts"), platform: "browser", write: false, logLevel: "warn", output: { format: "esm", minify: false, codeSplitting: false } });
const chunks = built.output.filter((c) => c.type === "chunk");
if (chunks.length !== 1) throw new Error(`worker: expected 1 chunk, got ${chunks.length}`);
writeFileSync(join(out, "worker.js"), prelude + chunks[0].code);
writeFileSync(join(out, "index.html"), '<!doctype html><meta charset="utf-8"><title>asr kv drift</title><script type="module" src="driver.js"></script>');
// One worker, two passes: the exact pass resets the stream at hop 0 like the sliding one.
writeFileSync(join(out, "driver.js"), `
const exact = new URLSearchParams(location.search).get("exact") === "1";
const res = (window.__res = { exact, phase: "load", hops: [], log: [] });
(async () => {
  const meta = await (await fetch("hops.json")).json();
  const pcm = new Float32Array(await (await fetch("hops.f32")).arrayBuffer());
  const w = new Worker("worker.js", { type: "module" });
  let waiter = null;
  w.onmessage = (e) => { const m = e.data; if (m.type === "error") res.log.push(JSON.stringify(m).slice(0, 2000)); if (waiter && ["ready", "result", "error"].includes(m.type)) { const r = waiter; waiter = null; r(m); } };
  w.onerror = (e) => res.log.push("onerror " + e.message);
  const ask = (msg, t = []) => new Promise((r) => { waiter = r; w.postMessage(msg, t); });
  const engineAssets = { manifest: await (await fetch("/e/manifest.json")).arrayBuffer(), qknorm: await (await fetch("/e/qknorm.bin")).arrayBuffer() };
  const ready = await ask({ type: "load", repo: meta.repo, rev: meta.rev, cacheName: "asr-kv-drift", engineAssets, logLevel: "warning" });
  if (ready.type !== "ready") throw new Error("load failed: " + JSON.stringify(ready).slice(0, 1500));
  res.phase = "hops";
  for (let i = 0; i < meta.hops.length; i++) {
    const hop = pcm.slice(i * meta.hopSamples, (i + 1) * meta.hopSamples);
    const r = await ask({ type: "stream", id: i + 1, pcm: hop, lang: meta.lang, gapMs: 0, pace: false, vad: false, exactKv: exact, reset: i === 0, quiet: meta.hops[i].quiet, maxTokens: meta.maxTokens }, [hop.buffer]);
    if (r.type === "error") throw new Error("hop " + i + ": " + r.stage + " " + r.message);
    const pick = ["decoded", "final", "hist", "conf", "tent", "ids", "prefill", "cached", "ctx", "prefillMs", "totalMs", "slides", "shiftMs", "rebuildMs", "rebuildTokens", "slidN"];
    res.hops.push({ i, quiet: meta.hops[i].quiet, ...Object.fromEntries(pick.map((k) => [k, r[k] ?? null])) });
  }
  w.terminate();
  res.phase = "done";
})().catch((e) => { res.log.push(String(e?.stack || e)); res.phase = "error"; });
`);

const roots = [["/m/", modelDir], ["/e/", join(here, "engine-dev/model")], ["/", out]];
const server = createServer((q, r) => {
    const url = decodeURIComponent(q.url.split("?")[0]);
    const [prefix, root] = roots.find(([p]) => url.startsWith(p));
    const f = join(root, url.slice(prefix.length) || "index.html");
    try {
        r.writeHead(200, { "content-type": { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" }[extname(f)] || "application/octet-stream", "content-length": statSync(f).size, "cache-control": "no-store" });
        createReadStream(f).pipe(r);
    } catch { r.writeHead(404); r.end(); }
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${server.address().port}/`;

const usedMiB = () => execFileSync("nvidia-smi", ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim().split(",").map(Number);
async function cdpRun(exact) {
    const [used, total] = usedMiB();
    console.error(`[exact=${exact}] GPU before load: ${used}/${total} MiB used`);
    if (total - used < MIN_FREE_MIB) throw new Error(`only ${total - used} MiB VRAM free (< ${MIN_FREE_MIB}); not loading a model`);
    const tab = await (await fetch(`${CDP}/json/new?${encodeURIComponent(`${origin}?exact=${exact}`)}`, { method: "PUT" })).json();
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });
    let id = 0, res, peak = used;
    const pend = new Map();
    ws.onmessage = (m) => { const d = JSON.parse(m.data); pend.get(d.id)?.(d); pend.delete(d.id); };
    const ev = (expression) => new Promise((ok) => { const i = ++id; pend.set(i, ok); ws.send(JSON.stringify({ id: i, method: "Runtime.evaluate", params: { expression, returnByValue: true } })); });
    try {
        for (;;) {
            await new Promise((ok) => setTimeout(ok, 2000));
            peak = Math.max(peak, usedMiB()[0]);
            res = (await ev("window.__res && JSON.parse(JSON.stringify(window.__res))")).result?.result?.value;
            if (res && (res.phase === "done" || res.phase === "error")) break;
        }
    } finally {
        ws.close();
        await fetch(`${CDP}/json/close/${tab.id}`).catch(() => {});
    }
    if (res.phase !== "done") throw new Error(`[exact=${exact}] ${res.log.at(-1)}`);
    console.error(`[exact=${exact}] peak GPU ${peak} MiB (+${peak - used})`);
    writeFileSync(join(out, `run-exact${exact}.json`), JSON.stringify(res));
    return res.hops;
}
let slide, exact;
try {
    slide = await cdpRun(0);
    exact = await cdpRun(1);
} finally {
    server.close();
}

const confOf = (h) => (h.ids ? h.ids.slice(0, Math.max(0, h.ids.length - ROLLBACK)) : null);
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// Slides since the prefix was last computed whole, as of each hop's decode.
let since = 0, matching = 0, exactHops = 0, exactMatching = 0, first = null;
const rows = slide.map((s, i) => {
    const x = exact[i];
    if (s.rebuildMs != null) since = 0;
    since += s.slides ?? 0;
    const ok = eq(confOf(s), confOf(x)) && s.decoded === x.decoded;
    if (ok) matching++;
    if (s.decoded && since === 0) { exactHops++; if (ok) exactMatching++; }
    if (!ok && !first) {
        const a = confOf(s) ?? [], b = confOf(x) ?? [];
        first = { hop: i, slidesSinceRebuild: since, tokenIndex: a.findIndex((t, k) => t !== b[k]), sliding: s.conf, full: x.conf };
    }
    return { i, ok, since, cached: s.cached, prefill: s.prefill, prefillMs: s.prefillMs && +s.prefillMs.toFixed(1), fullPrefill: x.prefill, fullPrefillMs: x.prefillMs && +x.prefillMs.toFixed(1), slides: s.slides, shiftMs: s.shiftMs && +s.shiftMs.toFixed(2), rebuildMs: s.rebuildMs && +s.rebuildMs.toFixed(1), rebuildTokens: s.rebuildTokens };
});
const shift = rows.filter((r) => r.slides);
const mean = (a) => (a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(2) : null);
const report = {
    quiet: QUIET,
    hops: rows.length,
    matchingHops: matching,
    exactPrefixHops: { decoded: exactHops, matching: exactMatching },
    firstDivergence: first,
    slideEvents: shift.length,
    shiftMsPerSlide: mean(shift.map((r) => r.shiftMs / r.slides)),
    fullPrefillMs: mean(rows.filter((r) => r.fullPrefillMs).map((r) => r.fullPrefillMs)),
    cachedPrefillMs: mean(rows.filter((r) => r.prefillMs && r.cached).map((r) => r.prefillMs)),
    rebuilds: rows.filter((r) => r.rebuildMs != null).map((r) => ({ hop: r.i, ms: r.rebuildMs, tokens: r.rebuildTokens })),
    finalText: { sliding: slide.at(-1).conf, full: exact.at(-1).conf },
};
writeFileSync(join(out, "report.json"), JSON.stringify({ ...report, rows }, null, 1));
console.log(JSON.stringify(report));
for (const r of rows) console.log(JSON.stringify(r));
console.error(`full report: ${join(out, "report.json")}`);
