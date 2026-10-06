#!/usr/bin/env node
// Hop-by-hop parity of the streaming caption path: the in-house WebGPU worker
// (this tree) against the ORT-era worker (an older checkout, e.g. a git
// worktree at e4059b5), both driven through the real `stream` message
// protocol on the same PCM. Each hop is 1 s (streamHopSec 1), pre-scaled and
// quiet-flagged exactly as captions/index.ts does before posting, and sent
// only after the previous result, so both sides see identical hop boundaries.
// Per hop it compares the VAD gate, the confirmed and tentative token ids
// (the last ROLLBACK = 5 are tentative) and the text the page would display.
//
//   node scripts/asr-stream-parity.mjs <ort-era monkey-script dir> <hf model dir> <16 kHz mono f32le pcm> [lang]
//
// The model dir is the Hugging Face export the repo/rev below names (1.7B:
// the in-house engine is built for that export only). Chrome is reached over
// CDP at STREAM_PARITY_CDP (default http://127.0.0.1:9444): launch a separate
// instance with --enable-unsafe-webgpu and its own --user-data-dir. Each run
// opens and closes its own tab. STREAM_PARITY_RUNS picks runs as side:vad
// pairs (default "ort:1,new:1,ort:0,new:0"); results stay in the output dir,
// so STREAM_PARITY_RUNS=none only re-compares saved runs.
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";

const [ortDir, modelDir, pcmPath, lang = "ko"] = process.argv.slice(2).map((x, i) => (i < 3 ? resolve(x) : x));
if (!ortDir || !modelDir || !pcmPath) throw new Error("usage: node scripts/asr-stream-parity.mjs <ort-era monkey-script dir> <hf model dir> <pcm.f32> [lang]");
const here = resolve(import.meta.dirname, "..");
const out = join(tmpdir(), "asr-stream-parity");
mkdirSync(out, { recursive: true });
const CDP = process.env.STREAM_PARITY_CDP || "http://127.0.0.1:9444";
// 9333 is the user's headed Chrome running a live caption tab; a parity run
// loads a second 1.7B model beside it and once ran the GPU out of memory.
if (new URL(CDP).port === "9333") throw new Error("refusing CDP port 9333 (the live caption Chrome); launch a separate instance");
const RUNS = (process.env.STREAM_PARITY_RUNS || "ort:1,new:1,ort:0,new:0").split(",").filter((r) => r !== "none").map((r) => r.split(":"));
const MIN_FREE_MIB = Number(process.env.STREAM_PARITY_MIN_FREE_MIB || 3500);
const SR = 16000, HOP = SR, ROLLBACK = 5, SILENCE_RMS = 0.004;
const REPO = "jiangzhuo9357/Qwen3-ASR-1.7B-ONNX", REV = "fcc238dfdc95cdcccaa9a7e2c7f5abc2f94f44a7";

// ---- hop schedule: the page's quiet flag and slow-peak gain (index.ts) ----
const raw = readFileSync(pcmPath);
const audio = new Float32Array(raw.buffer, raw.byteOffset, raw.length >> 2);
const nHops = Math.floor(audio.length / HOP);
const hopsPcm = new Float32Array(nHops * HOP);
const hops = [];
let statePeak = 0;
for (let i = 0; i < nHops; i++) {
    const pcm = audio.slice(i * HOP, (i + 1) * HOP);
    let sum = 0, peak = 0;
    for (const v of pcm) { sum += v * v; peak = Math.max(peak, Math.abs(v)); }
    const quiet = Math.sqrt(sum / pcm.length) < SILENCE_RMS;
    statePeak = Math.max(peak, statePeak * 0.9);
    const gain = Math.min(20, 0.5 / Math.max(statePeak, 1e-4));
    for (let j = 0; j < pcm.length; j++) pcm[j] *= gain;
    hopsPcm.set(pcm, i * HOP);
    hops.push({ quiet, gain: +gain.toFixed(4) });
}
writeFileSync(join(out, "hops.f32"), hopsPcm);
writeFileSync(join(out, "hops.json"), JSON.stringify({ repo: REPO, rev: REV, lang, hopSamples: HOP, maxTokens: Math.round(16 + 12 * (HOP / SR)), hops }));

// ---- the two worker bundles, as tsdown.config.ts builds the inlined one ----
// The prelude points Hugging Face and the VAD CDN at the local model dir; the
// ORT side still loads onnxruntime-web itself from jsDelivr.
const prelude = `{
  const f = self.fetch.bind(self), o = self.location.origin;
  const map = (u) => {
    if (u.startsWith("https://huggingface.co/api/")) return null;
    const hf = "https://huggingface.co/${REPO}/resolve/${REV}/";
    if (u.startsWith(hf)) return o + "/m/" + u.slice(hf.length);
    if (u.endsWith("/silero_vad_v5.onnx")) return o + "/m/silero_vad_v5.onnx";
    return u;
  };
  self.fetch = (input, init) => {
    const u = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const m = map(u);
    return m === null ? Promise.resolve(new Response("offline", { status: 404 })) : f(m, init);
  };
}
`;
const { build } = await import("rolldown");
for (const [side, root] of [["new", here], ["ort", ortDir]]) {
    const r = await build({ input: join(root, "src/captions/worker/main.ts"), platform: "browser", write: false, logLevel: "warn", output: { format: "esm", minify: false, codeSplitting: false } });
    const chunks = r.output.filter((c) => c.type === "chunk");
    if (chunks.length !== 1) throw new Error(`${side} worker: expected 1 chunk, got ${chunks.length}`);
    writeFileSync(join(out, `${side}-worker.js`), prelude + chunks[0].code);
}
writeFileSync(join(out, "index.html"), '<!doctype html><meta charset="utf-8"><title>asr stream parity</title><script type="module" src="driver.js"></script>');
writeFileSync(join(out, "driver.js"), readFileSync(resolve(import.meta.dirname, "asr-stream-parity.browser.mjs")));

const roots = [["/m/", modelDir], ["/e/", join(here, "engine-dev/model")], ["/", out]];
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
const server = createServer((q, r) => {
    const url = decodeURIComponent(q.url.split("?")[0]);
    const [prefix, root] = roots.find(([p]) => url.startsWith(p));
    const f = join(root, url.slice(prefix.length) || "index.html");
    try {
        r.writeHead(200, { "content-type": types[extname(f)] || "application/octet-stream", "content-length": statSync(f).size, "cache-control": "no-store" });
        createReadStream(f).pipe(r);
    } catch { r.writeHead(404); r.end(); }
});
await new Promise((ok) => server.listen(Number(process.env.STREAM_PARITY_HTTP_PORT || 0), "127.0.0.1", ok));
const origin = `http://127.0.0.1:${server.address().port}/`;

const freeMiB = () => {
    const [used, total] = execFileSync("nvidia-smi", ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim().split(",").map(Number);
    return { used, total, free: total - used };
};
async function cdpRun(side, vad) {
    const mem = freeMiB();
    console.error(`[${side} vad=${vad}] GPU before load: ${mem.used}/${mem.total} MiB used`);
    if (mem.free < MIN_FREE_MIB) throw new Error(`only ${mem.free} MiB VRAM free (< ${MIN_FREE_MIB}); not loading a model`);
    const tab = await (await fetch(`${CDP}/json/new?${encodeURIComponent(`${origin}?side=${side}&vad=${vad}`)}`, { method: "PUT" })).json();
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });
    let id = 0;
    const pend = new Map();
    ws.onmessage = (m) => { const d = JSON.parse(m.data); pend.get(d.id)?.(d); pend.delete(d.id); };
    const ev = (expression) => new Promise((ok) => { const i = ++id; pend.set(i, ok); ws.send(JSON.stringify({ id: i, method: "Runtime.evaluate", params: { expression, returnByValue: true } })); });
    let res, printed = 0, lastHop = -1;
    try {
        for (;;) {
            await new Promise((ok) => setTimeout(ok, 3000));
            res = (await ev("window.__res && JSON.parse(JSON.stringify(window.__res))")).result?.result?.value;
            if (!res) continue;
            for (const line of res.log.slice(printed)) console.error(`  ${line}`);
            printed = res.log.length;
            if (res.hops.length - 1 !== lastHop) { lastHop = res.hops.length - 1; if (lastHop >= 0 && lastHop % 10 === 0) console.error(`  hop ${lastHop} (${freeMiB().used} MiB used)`); }
            if (res.phase === "done" || res.phase === "error") break;
        }
    } finally {
        ws.close();
        await fetch(`${CDP}/json/close/${tab.id}`).catch(() => {});
    }
    if (res.phase !== "done") throw new Error(`[${side} vad=${vad}] ${res.log.at(-1)}`);
    if (res.side !== side || res.vad !== (vad === "1")) throw new Error(`asked side=${side} vad=${vad}, the page ran side=${res.side} vad=${res.vad}`);
    writeFileSync(join(out, `run-${side}-vad${vad}.json`), JSON.stringify(res));
    return res;
}
try {
    for (const [side, vad] of RUNS) await cdpRun(side, vad);
} finally {
    server.close();
}

// ---- compare ----
// What index.ts shows: history (+ hist of each result) then conf, tent after.
function display(hops) {
    let history = "", text = "", tent = "";
    return hops.map((h) => {
        if (h.hist) history = (history + " " + h.hist).trim().slice(-300);
        if (h.conf !== null) { text = (history + " " + h.conf).trim().slice(-300); tent = h.tent; }
        else if (h.final) { text = history; tent = ""; }
        return text + tent;
    });
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function compare(vad) {
    const f = (s) => join(out, `run-${s}-vad${vad}.json`);
    if (!existsSync(f("new")) || !existsSync(f("ort"))) return null;
    const ours = JSON.parse(readFileSync(f("new"), "utf8")).hops, ort = JSON.parse(readFileSync(f("ort"), "utf8")).hops;
    const dO = display(ours), dR = display(ort);
    const split = (h) => (h.ids ? { confirmed: h.ids.slice(0, Math.max(0, h.ids.length - ROLLBACK)), tentative: h.ids.slice(Math.max(0, h.ids.length - ROLLBACK)) } : { confirmed: null, tentative: null });
    let matching = 0, first = null;
    const rows = ours.map((o, i) => {
        const r = ort[i], so = split(o), sr = split(r);
        const stage = o.speech !== r.speech || o.speechSec !== r.speechSec ? "vad" : o.decoded !== r.decoded || o.final !== r.final ? "gate" : !eq(so.confirmed, sr.confirmed) ? "confirmed" : !eq(so.tentative, sr.tentative) ? "tentative" : dO[i] !== dR[i] ? "text" : null;
        if (!stage) matching++;
        else if (!first) {
            const at = o.ids && r.ids ? o.ids.findIndex((t, k) => t !== r.ids[k]) : -1;
            first = { hop: i, stage, tokenIndex: at, ours: { speechSec: o.speechSec, ids: o.ids, text: dO[i] }, ort: { speechSec: r.speechSec, ids: r.ids, text: dR[i] } };
        }
        return { i, stage, ours: { sp: o.speechSec, n: o.ids?.length ?? null, text: dO[i] }, ort: { sp: r.speechSec, n: r.ids?.length ?? null, text: dR[i] } };
    });
    return { hops: ours.length, matchingHops: matching, firstDivergence: first, textEqual: dO.at(-1) === dR.at(-1), finalText: { ours: dO.at(-1), ort: dR.at(-1) }, rows };
}
const report = { audio: { source: pcmPath, seconds: +(audio.length / SR).toFixed(2), hops: nHops }, vadOn: compare("1"), vadOff: compare("0") };
writeFileSync(join(out, "report.json"), JSON.stringify(report, null, 1));
for (const k of ["vadOn", "vadOff"]) if (report[k]) { const { rows, ...rest } = report[k]; console.log(k, JSON.stringify(rest)); }
console.error(`full report: ${join(out, "report.json")}`);
