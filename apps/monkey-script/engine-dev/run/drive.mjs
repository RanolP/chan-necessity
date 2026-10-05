// node drive.mjs <outfile> <modes>  — Firefox BiDi on 9531, page served on 8831
import fs from "node:fs";
const [outf, modes] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const url = `http://127.0.0.1:8831/test/index.html?modes=${modes}&t=${Date.now()}`;
const ws = new WebSocket("ws://127.0.0.1:9531/session");
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let id = 0; const pend = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { const p = pend.get(m.id); pend.delete(m.id); m.type === "error" ? p[1](new Error(m.message)) : p[0](m.result); } };
const send = (method, params = {}) => { const msg = { id: ++id, method, params }; ws.send(JSON.stringify(msg)); return new Promise((a, b) => pend.set(msg.id, [a, b])); };
await send("session.new", { capabilities: {} });
const { context } = await send("browsingContext.create", { type: "window" });
await send("browsingContext.navigate", { context, url, wait: "complete" });
const ev = async (e) => { const r = await send("script.evaluate", { expression: `JSON.stringify(${e})`, target: { context }, awaitPromise: true }); return r.type === "exception" ? { exc: r.exceptionDetails.text } : JSON.parse(r.result.value ?? "null"); };
const t0 = Date.now(); let last = "";
while (true) {
  await sleep(1500);
  const r = await ev("window.__res && { phase: window.__res.phase, n: window.__res.log.length, err: window.__res.gpuErrors?.length }");
  const s = JSON.stringify(r); if (s !== last) console.log(Math.round((Date.now() - t0) / 1000), s); last = s;
  if (r?.phase === "done" || r?.phase === "error" || Date.now() - t0 > 1500e3) break;
}
const res = await ev("window.__res");
fs.writeFileSync(outf, JSON.stringify(res, null, 1)); console.log("saved", outf, res?.phase);
await send("browsingContext.close", { context }).catch(() => {}); await send("session.end").catch(() => {}); ws.close(); process.exit(0);
