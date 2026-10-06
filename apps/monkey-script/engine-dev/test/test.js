import { createEngine } from "../../src/captions/engine/engine.ts";
import { prec } from "../../src/captions/engine/kernels.ts";
const qs = new URLSearchParams(location.search);
prec.r16 = qs.get("r16") !== "0";
const R = (window.__res = { phase: "start", log: [] });
const log = (x) => { R.log.push(x); document.getElementById("log").textContent += JSON.stringify(x).slice(0, 600) + "\n"; };
const bin = async (u) => (await fetch(u)).arrayBuffer();
const maxDiff = (a, b) => { let m = 0, mr = 0, at = -1; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) { m = d; at = i; } mr = Math.max(mr, Math.abs(b[i])); } return { maxAbs: +m.toPrecision(4), at, maxRef: +mr.toPrecision(4) }; };
const topk = (a, k) => Array.from(a.keys()).sort((i, j) => a[j] - a[i] || i - j).slice(0, k);
const top2 = (a) => { let b = -1, c = -1; for (let i = 0; i < a.length; i++) { if (b < 0 || a[i] > a[b]) { c = b; b = i; } else if (c < 0 || a[i] > a[c]) c = i; } return { top: b, second: c, margin: +(a[b] - a[c]).toPrecision(4) }; };
try {
  const modes = (qs.get("modes") || "l0,full").split(",");
  const EXPECTED = "현재 시간은 2020년 1월 22일 오후 12시 26분 22초입니다.";
  const ad = await navigator.gpu.requestAdapter();
  const device = await ad.requestDevice();
  device.lost.then((i) => { R.lost = { reason: i.reason, message: i.message }; R.phase = "error"; log({ lost: R.lost }); });
  device.onuncapturederror = (e) => { (R.gpuErrors ||= []).push(String(e.error.message).slice(0, 500)); };
  R.phase = "loading";
  const t0 = performance.now();
  const eng = await createEngine(device, {
    manifestUrl: "../model/manifest.json",
    urls: { "decoder_weights.q4f16.data": "../tools/refmodel/decoder_weights.q4f16.data", "qknorm.bin": "../model/qknorm.bin", embed: "../model/embed/embed_tokens.int8.bin", embedScales: "../model/embed/embed_scales.f32.bin" },
    onProgress: (a, b) => (R.progress = [a, b]),
  });
  R.loadMs = Math.round(performance.now() - t0); log({ loadMs: R.loadMs });
  const truth = await (await fetch("truth.json")).json();
  const pk = new Uint8Array(await bin("prefill_k.f16")), pv = new Uint8Array(await bin("prefill_v.f16"));
  const reset = () => eng.setPrefill(pk, pv, truth.S);
  R.phase = "running";
  const ref = { S: truth.S, tok0: truth.tok0, pk, pv, tokens: [...truth.tokens, truth.final], text: truth.text };
  const resetRef = () => eng.setPrefill(ref.pk, ref.pv, ref.S);
  if (modes.includes("l0")) {
    reset(); eng.start(truth.tok0, truth.S, 1);
    const enc = device.createCommandEncoder(); eng.encodeSteps(enc, 1, 1); device.queue.submit([enc.finish()]);
    const ref = new Float32Array(await bin("ref_l0.f32"));
    const qkv = new Float32Array(await eng.read(eng.bufs.qkv, 4096 * 4)), ao = new Float32Array(await eng.read(eng.bufs.ao, 2048 * 4)), h = new Float32Array(await eng.read(eng.bufs.h, 2048 * 4));
    R.l0 = { qkv: maxDiff(qkv, ref.subarray(0, 4096)), attn: maxDiff(ao, ref.subarray(4096, 6144)), h: maxDiff(h, ref.subarray(6144, 8192)) };
    log({ l0: R.l0 });
  }
  if (modes.includes("full")) {
    // logits of the first 3 steps, one step per submit, against ORT CPU
    reset(); eng.start(truth.tok0, truth.S, 64);
    const ref = new Float32Array(await bin("logits3.f32")); const V = 151936; R.logits = [];
    for (let s = 0; s < 3; s++) {
      const enc = device.createCommandEncoder(); eng.encodeSteps(enc, 1); device.queue.submit([enc.finish()]);
      const lg = new Float32Array(await eng.read(eng.bufs.logits, V * 4)); const rf = ref.subarray(s * V, (s + 1) * V);
      const a = topk(lg, 5), b = topk(rf, 5);
      R.logits.push({ step: s, ...maxDiff(lg, rf), top5Overlap: a.filter((x) => b.includes(x)).length, engTop2: top2(lg), ortTop2: top2(rf) });
    }
    log({ logits: R.logits });
  }
  if (modes.includes("full")) {
    const V = 151936;
    resetRef();
    const r = await eng.decode(ref.tok0, ref.S, { cap: 64, perSubmit: 16 });
    const ref2 = ref.tokens;
    const got = [ref.tok0, ...r.tokens];
    let div = -1; for (let i = 0; i < Math.max(got.length, ref2.length); i++) if (got[i] !== ref2[i]) { div = i; break; }
    R.full = { tokens: got, identical: div < 0, firstDivergence: div, timing: r.timing, done: r.done };
    if (div >= 0) {
      // margin at the divergent step: replay to it one step at a time and read the logits
      resetRef(); eng.start(ref.tok0, ref.S, 64);
      for (let s = 0; s < div; s++) { const enc = device.createCommandEncoder(); eng.encodeSteps(enc, 1); device.queue.submit([enc.finish()]); }
      const lg = new Float32Array(await eng.read(eng.bufs.logits, V * 4));
      R.full.divergentStepEngTop2 = top2(lg); R.full.oursAtDiv = got[div]; R.full.ortAtDiv = ref2[div];
      R.full.ortTokLogitInEngine = lg[ref2[div]]; R.full.engTokLogitInEngine = lg[got[div]];
    }
    const vocab = await (await fetch("../tools/refmodel/vocab.json")).json(); const inv = []; for (const [k, v] of Object.entries(vocab)) inv[v] = k;
    const bs = []; for (let i = 33; i < 127; i++) bs.push(i); for (let i = 161; i < 173; i++) bs.push(i); for (let i = 174; i < 256; i++) bs.push(i);
    const cs = bs.slice(); let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
    const byteOf = new Map(cs.map((c, i) => [String.fromCodePoint(c), bs[i]]));
    const bytes = []; for (const t of got) if (t < 151643 && inv[t]) for (const ch of inv[t]) bytes.push(byteOf.get(ch) ?? 63);
    let text = new TextDecoder().decode(new Uint8Array(bytes)); const k = text.lastIndexOf("<asr_text>"); text = (k >= 0 ? text.slice(k + 10) : text).trim();
    R.full.text = text; R.full.textEqualsRef = text === ref.text; R.full.textEqualsExpected = text === EXPECTED;
    log({ full: R.full });
  }
  if (modes.includes("time")) {
    // encode-only microbench: 16 steps encoded and discarded
    const e0 = performance.now(), d0 = eng.dispatches; const enc = device.createCommandEncoder(); eng.encodeSteps(enc, 16); enc.finish(); const em = performance.now() - e0;
    R.encodeBench = { dispatches: eng.dispatches - d0, ms: +em.toFixed(2), usPerDispatch: +((em * 1000) / (eng.dispatches - d0)).toFixed(2) };
    R.runs = [];
    const cfgs = JSON.parse(qs.get("pumps") || '[[0,false],[1,false],[0,true]]');
    for (const [iv, owd] of cfgs) for (let i = 0; i < 4; i++) {
      eng.pump.interval = iv; resetRef(); await device.queue.onSubmittedWorkDone();
      const r = await eng.decode(ref.tok0, ref.S, { cap: 64, perSubmit: 16, owd });
      R.runs.push({ iv, owd, warm: i > 0, n: r.tokens.length, same: JSON.stringify(r.tokens) === JSON.stringify(R.full?.tokens?.slice(1)), ...r.timing });
    }
    log({ encodeBench: R.encodeBench, runs: R.runs });
  }
  R.phase = "done";
} catch (e) { R.err = String(e?.stack || e).slice(0, 1500); R.phase = "error"; log({ err: R.err }); }
