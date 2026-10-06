// Browser half of scripts/asr-stream-parity.mjs: drives one captions worker
// bundle (?side=new|ort) through its real `stream` protocol, one 1 s hop at a
// time, each hop sent only after the previous result arrived, so hop
// boundaries depend on the audio alone and never on wall time. Pacing is off
// (no frame waits) and the ORT side runs with engine:false (its pure ORT
// decoder_step loop). The per-hop records land in window.__res.
const q = new URLSearchParams(location.search);
const side = q.get("side");
const vad = q.get("vad") === "1";
const res = (window.__res = { side, vad, phase: "load", hops: [], log: [] });
const note = (m) => res.log.push(`[${(performance.now() / 1000).toFixed(1)}s] ${m}`);

async function run() {
    const meta = await (await fetch("hops.json")).json();
    const pcm = new Float32Array(await (await fetch("hops.f32")).arrayBuffer());
    const w = new Worker(`${side}-worker.js`, { type: "module" });
    let waiter = null;
    w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "status" || m.type === "caps") return;
        if (m.type === "progress") { res.progress = m; return; }
        if (m.type === "error") note(`worker error ${JSON.stringify(m).slice(0, 2000)}`);
        if (waiter && (m.type === "ready" || m.type === "result" || m.type === "error")) { const r = waiter; waiter = null; r(m); }
    };
    w.onerror = (e) => note(`worker onerror ${e.message} ${e.filename}:${e.lineno}`);
    const ask = (msg, transfer = []) => new Promise((r) => { waiter = r; w.postMessage(msg, transfer); });
    const engineAssets = side === "new"
        ? { manifest: await (await fetch("/e/manifest.json")).arrayBuffer(), qknorm: await (await fetch("/e/qknorm.bin")).arrayBuffer() }
        : null; // ORT side: no engine, so decode is ORT decoder_step only
    const ready = await ask({ type: "load", repo: meta.repo, rev: meta.rev, cacheName: "asr-stream-parity", engineAssets, logLevel: "warning" });
    if (ready.type !== "ready") throw new Error(`load failed: ${JSON.stringify(ready).slice(0, 1500)}`);
    res.ready = ready;
    note(`ready ${JSON.stringify(ready)}`);
    res.phase = "hops";
    for (let i = 0; i < meta.hops.length; i++) {
        const h = meta.hops[i];
        const hop = pcm.slice(i * meta.hopSamples, (i + 1) * meta.hopSamples);
        const r = await ask({ type: "stream", id: i + 1, pcm: hop, lang: meta.lang, gapMs: 0, pace: false, vad, engine: false, reset: i === 0, quiet: h.quiet, maxTokens: meta.maxTokens }, [hop.buffer]);
        if (r.type === "error") throw new Error(`hop ${i}: ${r.stage} ${r.message}`);
        res.hops.push({ i, quiet: h.quiet, speech: r.speech, speechSec: r.speechSec, decoded: !!r.decoded, final: !!r.final, aborted: !!r.aborted, hist: r.hist ?? "", conf: r.conf ?? null, tent: r.tent ?? null, ids: r.ids ?? null, gen: r.tokens ?? null, prefill: r.prefill ?? null, audioTokens: r.audioTokens ?? null, slidN: r.slidN ?? null, cached: r.cached ?? null, ctx: r.ctx ?? null, slides: r.slides ?? null, prefillMs: r.prefillMs == null ? null : Math.round(r.prefillMs), ms: Math.round(r.totalMs ?? 0) });
    }
    w.terminate();
    res.phase = "done";
}
run().catch((e) => { note(String(e?.stack || e)); res.phase = "error"; });
