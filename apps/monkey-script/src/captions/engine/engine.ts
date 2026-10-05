// Model-specific WebGPU decoder for Qwen3-ASR 1.7B q4f16.
// Everything (pipelines, buffers, a fixed-size KV cache, bind groups) is built once at load;
// a decode step is 6 dispatches per layer + embed + 2 LM-head halves + argmax, and the
// argmax kernel feeds the next token back on the GPU, so steps chain with no readback.
import * as K from "./kernels.ts";

const ST_WORDS = 16 + 512;

export interface EngineConfig {
  hidden: number;
  intermediate: number;
  vocab: number;
  layers: number;
  rope_theta: number;
  attn_scale_f16: number;
  eos: number[];
  source?: unknown;
}
interface ManifestTensor {
  length: number;
  segments: [number, number][];
  src: string;
}
interface Manifest {
  config: EngineConfig;
  tensors: Record<string, ManifestTensor>;
}
export type Fetcher = (url: string) => Promise<Response>;
export interface EngineOptions {
  manifestUrl: string;
  urls: Record<string, string>;
  LMAX?: number;
  onProgress?: (loaded: number, total: number) => void;
  fetch?: Fetcher;
}
export interface DecodeOptions {
  cap?: number;
  perSubmit?: number;
  owd?: boolean;
}
export interface DecodeTiming {
  encodeMs: number;
  submits: number;
  writeBuffers: number;
  dispatches: number;
  waitMs: number;
  emptyPumpSubmits?: number;
  totalMs?: number;
}
interface Segment {
  off: number;
  len: number;
}

export async function createEngine(device: GPUDevice, opt: EngineOptions) {
  // `fetch` lets a caller serve the weights from somewhere other than the network (Cache Storage).
  const { manifestUrl, urls, LMAX = 448, onProgress = () => {}, fetch: get = (u: string) => fetch(u) } = opt;
  const man: Manifest = await (await get(manifestUrl)).json();
  const C = man.config;
  const H = C.hidden, I = C.intermediate, V = C.vocab, NL = C.layers;
  const LM_HALF = V / 2, EMB_PARTS = 4, EMB_ROWS = V / EMB_PARTS;
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buf = (size: number, usage = S | CD) => device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });

  const NP = 2 * (LM_HALF / 8);
  // ---- pipelines
  const mk = async (name: string, code: string) => {
    const module = device.createShaderModule({ code });
    try { return await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } }); }
    catch (e) { const ci = await module.getCompilationInfo(); throw new Error(`pipeline ${name}: ${(e as Error).message}\n${ci.messages.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`); }
  };
  const P = {
    embed: await mk("embed", K.embedWGSL(EMB_ROWS)),
    qkv: await mk("qkv", K.gemvWGSL({ K: H, N: 4096, mode: "plain", norm: true })),
    rope: await mk("rope", K.ropeWGSL({ LMAX, theta: C.rope_theta })),
    attn: await mk("attn", K.attnWGSL({ LMAX, scale: C.attn_scale_f16 })),
    o: await mk("o", K.gemvWGSL({ K: H, N: H, mode: "resid", norm: false })),
    gu: await mk("gu", K.gemvWGSL({ K: H, N: I, mode: "silu", norm: true })),
    down: await mk("down", K.gemvWGSL({ K: I, N: H, mode: "resid", norm: false })),
    lm: await mk("lm", K.gemvWGSL({ K: H, N: LM_HALF, mode: "lm", norm: true })),
    argmax: await mk("argmax", K.argmaxWGSL({ NP, eos: C.eos, LMAX })),
  };
  // ---- weights: one GPU buffer per fused tensor; LM head split by rows into two halves (128 MiB binding cap)
  const T: Record<string, ManifestTensor & { parts: { start: number; len: number; buf: GPUBuffer }[] }> = {};
  for (const [name, t] of Object.entries(man.tensors)) {
    const splitRows = name.startsWith("lm.") ? 2 : 1;
    const parts = [];
    const per = t.length / splitRows;
    for (let i = 0; i < splitRows; i++) parts.push({ start: i * per, len: per, buf: buf(per) });
    T[name] = { ...t, parts };
  }
  const bySrc: Record<string, (Segment & { dst: number; t: string })[]> = {};
  for (const [name, t] of Object.entries(T)) {
    let dst = 0;
    for (const [off, len] of t.segments) { (bySrc[t.src] ||= []).push({ off, len, dst, t: name }); dst += len; }
  }
  const writeT = (name: string, dst: number, data: Uint8Array) => {
    for (const p of T[name].parts) {
      const a = Math.max(dst, p.start), b = Math.min(dst + data.length, p.start + p.len);
      if (b > a) device.queue.writeBuffer(p.buf, a - p.start, data, a - dst, b - a);
    }
  };
  let loaded = 0; const total = Object.values(T).reduce((s, t) => s + t.length, 0) + V * H;
  const prog = (n: number) => { loaded += n; onProgress(loaded, total); };
  for (const [src, segs] of Object.entries(bySrc)) await streamSegments(device, get, urls[src], segs, (s, sp, d) => { writeT(s.t, s.dst + sp, d); prog(d.length); });

  // ---- embeddings: int8 rows in 4 buffers + per-row f32 scales
  const emb = Array.from({ length: EMB_PARTS }, () => buf(EMB_ROWS * H));
  const embSegs = emb.map((b, i) => ({ off: i * EMB_ROWS * H, len: EMB_ROWS * H, dst: 0, b }));
  await streamSegments(device, get, urls.embed, embSegs, (s, sp, d) => { device.queue.writeBuffer(s.b, sp, d); prog(d.length); });
  const escData = new Uint8Array(await (await get(urls.embedScales)).arrayBuffer());
  const esc = buf(escData.length); device.queue.writeBuffer(esc, 0, escData);

  // ---- activations, KV cache, state
  const st = buf(ST_WORDS * 4, S | CD | CS);
  const h = buf(H * 4, S | CD | CS), qkv = buf(4096 * 4, S | CS), qo = buf(2048 * 4, S | CS), ao = buf(2048 * 4, S | CS);
  const act = buf(I * 4, S | CS), logits = buf(V * 4, S | CS);
  const part = buf(NP * 8, S | CS);
  const kvBytes = NL * 8 * LMAX * 128 * 2;
  const kc = buf(kvBytes), vc = buf(kvBytes);

  const bg = (p: GPUComputePipeline, group: number, bufs: (GPUBuffer | null)[]) => device.createBindGroup({ layout: p.getBindGroupLayout(group), entries: bufs.map((b, i) => (b == null ? null : { binding: i, resource: { buffer: b } })).filter((e): e is NonNullable<typeof e> => Boolean(e)) });
  const t1 = (n: string, k = 0) => T[n].parts[k].buf;
  const uni = (vals: number[]) => { const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD }); device.queue.writeBuffer(b, 0, new Uint32Array([...vals, 0, 0, 0, 0].slice(0, 4))); return b; };

  const embedBG = bg(P.embed, 0, [st, h, ...emb, esc]);
  const layers: { qkv: GPUBindGroup; rope: GPUBindGroup[]; attn: GPUBindGroup[]; o: GPUBindGroup; gu: GPUBindGroup; down: GPUBindGroup }[] = [];
  for (let L = 0; L < NL; L++) {
    const u = uni([L]);
    const gm = (p: GPUComputePipeline, x: GPUBuffer, nw: GPUBuffer | null, tag: string, y: GPUBuffer) => bg(p, 0, [st, x, nw, t1(`l${L}.${tag}.q`), t1(`l${L}.${tag}.s`), t1(`l${L}.${tag}.z`), y]);
    layers.push({
      qkv: gm(P.qkv, h, t1(`l${L}.ln1`), "qkv", qkv),
      rope: [bg(P.rope, 0, [st, qkv, t1(`l${L}.qn`), t1(`l${L}.kn`), qo, kc, vc]), bg(P.rope, 1, [u])],
      attn: [bg(P.attn, 0, [st, qo, kc, vc, ao]), bg(P.attn, 1, [u])],
      o: gm(P.o, ao, null, "o", h),
      gu: gm(P.gu, h, t1(`l${L}.ln2`), "gu", act),
      down: gm(P.down, act, null, "down", h),
    });
  }
  const lmBG = [0, 1].map((k) => [
    bg(P.lm, 0, [st, h, t1("norm"), t1("lm.q", k), t1("lm.s", k), t1("lm.z", k), logits, part]),
    bg(P.lm, 1, [uni([k * LM_HALF, k * (LM_HALF / 8)])]),
  ]);
  const argBG = bg(P.argmax, 0, [st, part]);
  await device.queue.onSubmittedWorkDone();

  let dispatches = 0;
  const go = (pass: GPUComputePassEncoder, p: GPUComputePipeline, groups: GPUBindGroup[], n: number) => { pass.setPipeline(p); groups.forEach((g, i) => pass.setBindGroup(i, g)); pass.dispatchWorkgroups(n); dispatches++; };
  // Encodes `steps` full decode steps (or, with nLayers < NL, a truncated debug step without LM head).
  function encodeSteps(enc: GPUCommandEncoder, steps: number, nLayers = NL) {
    const pass = enc.beginComputePass();
    for (let s = 0; s < steps; s++) {
      go(pass, P.embed, [embedBG], H / 256);
      for (let L = 0; L < nLayers; L++) {
        const l = layers[L];
        go(pass, P.qkv, [l.qkv], 4096 / 8);
        go(pass, P.rope, l.rope, 32);
        go(pass, P.attn, l.attn, 16);
        go(pass, P.o, [l.o], H / 8);
        go(pass, P.gu, [l.gu], I / 8);
        go(pass, P.down, [l.down], H / 8);
      }
      if (nLayers === NL) {
        go(pass, P.lm, lmBG[0], LM_HALF / 8);
        go(pass, P.lm, lmBG[1], LM_HALF / 8);
        go(pass, P.argmax, [argBG], 1);
      }
    }
    pass.end();
  }

  // Prefill KV from ORT layout [NL, 1, 8, P, 128] (f16) into the fixed cache [NL, 8, LMAX, 128].
  function setPrefill(pk: ArrayBufferView, pv: ArrayBufferView, Pn: number) {
    const row = Pn * 128 * 2;
    for (let L = 0; L < NL; L++) for (let k = 0; k < 8; k++) {
      const src = (L * 8 + k) * row, dst = (L * 8 + k) * LMAX * 256;
      device.queue.writeBuffer(kc, dst, pk.buffer, pk.byteOffset + src, row);
      device.queue.writeBuffer(vc, dst, pv.buffer, pv.byteOffset + src, row);
    }
  }
  function start(tok0: number, pos: number, cap: number) {
    const s = new Uint32Array(16); s[0] = tok0; s[1] = pos; s[2] = 0; s[3] = 0; s[4] = cap;
    device.queue.writeBuffer(st, 0, s);
  }
  const pump = makePump(device);
  async function read(src: GPUBuffer, bytes: number, off = 0) {
    const m = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(src, off, m, 0, bytes); device.queue.submit([enc.finish()]);
    await pump(m.mapAsync(GPUMapMode.READ));
    const r = m.getMappedRange().slice(0); m.unmap(); m.destroy(); return r;
  }

  // Greedy decode from (tok0, pos): at most `perSubmit` steps per command buffer; between submits only
  // the 16-word state header is read back. Returns generated tokens (tok0 excluded) and timings.
  // Per chunk: 1 writeBuffer (start), ceil(n/perSubmit) work submits, 0 per-step CPU writes; the
  // readback after each submit carries the header and every token so far, so no final read is needed.
  const STB = 64 + 4 * 512, maps = [0, 1].map(() => device.createBuffer({ size: STB, usage: GPUBufferUsage.MAP_READ | CD }));
  async function decode(tok0: number, pos: number, { cap = 64, perSubmit = 16, owd = true }: DecodeOptions = {}) {
    cap = Math.min(cap, 512);
    const p0 = pump.submits;
    start(tok0, pos, cap);
    const t: DecodeTiming = { encodeMs: 0, submits: 0, writeBuffers: 1, dispatches: 0, waitMs: 0 };
    const t0 = performance.now();
    let all!: Uint32Array;
    for (let i = 0; ; i++) {
      const e0 = performance.now(); const d0 = dispatches;
      const enc = device.createCommandEncoder();
      encodeSteps(enc, perSubmit);
      const m = maps[i & 1];
      enc.copyBufferToBuffer(st, 0, m, 0, 64 + 4 * cap);
      const cb = enc.finish();
      t.encodeMs += performance.now() - e0; t.dispatches += dispatches - d0;
      device.queue.submit([cb]); t.submits++;
      const w0 = performance.now();
      if (owd) await device.queue.onSubmittedWorkDone();
      await pump(m.mapAsync(GPUMapMode.READ, 0, 64 + 4 * cap));
      t.waitMs += performance.now() - w0;
      all = new Uint32Array(m.getMappedRange(0, 64 + 4 * cap).slice(0)); m.unmap();
      if (all[2] !== 0) break;
    }
    t.emptyPumpSubmits = pump.submits - p0;
    t.totalMs = performance.now() - t0;
    return { tokens: Array.from(all.subarray(16, 16 + all[3])), done: all[2], eosStep: all[2] === 1 ? all[5] : -1, timing: t };
  }

  return { config: C, LMAX, encodeSteps, setPrefill, start, decode, read, bufs: { st, h, qkv, qo, ao, act, logits, part, kc, vc }, pump, get dispatches() { return dispatches; } };
}

// Streams `url` once and hands every 4-byte-aligned piece of each segment to `write(seg, offsetInSeg, bytes)`.
async function streamSegments<S extends Segment>(device: GPUDevice, get: Fetcher, url: string, segs: readonly S[], write: (seg: S, offsetInSeg: number, bytes: Uint8Array) => void) {
  segs = [...segs].sort((a, b) => a.off - b.off);
  const res = await get(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${await res.text().catch(() => "")}`.slice(0, 300));
  const rd = res.body!.getReader();
  let pos = 0, cur: Uint8Array = new Uint8Array(0), si = 0, sp = 0, sinceSync = 0;
  while (si < segs.length) {
    const { done, value } = await rd.read();
    if (done) break;
    cur = cur.length ? concat(cur, value) : value;
    while (si < segs.length) {
      const s = segs[si], at = s.off + sp;
      if (at < pos) throw new Error(`overlapping segment at ${at} < ${pos}`);
      const n = (Math.min(s.off + s.len, pos + cur.length) - at) & ~3;
      if (n <= 0) break;
      write(s, sp, cur.subarray(at - pos, at - pos + n)); sinceSync += n;
      sp += n; if (sp === s.len) { si++; sp = 0; }
    }
    const need = si < segs.length ? segs[si].off + sp : pos + cur.length;
    const drop = Math.min(need - pos, cur.length);
    if (drop > 0) { cur = cur.slice(drop); pos += drop; }
    if (sinceSync > (128 << 20)) { sinceSync = 0; await device.queue.onSubmittedWorkDone(); }
  }
  rd.cancel().catch(() => {});
  if (si < segs.length) throw new Error(`${url}: stream ended at ${pos}, segment ${si}/${segs.length} unfinished`);
}
function concat(a: Uint8Array, b: Uint8Array) { const c = new Uint8Array(a.length + b.length); c.set(a); c.set(b, a.length); return c; }

// Firefox only resolves mapAsync while the queue is being polled; submit empty work until it settles.
// `interval` ms between empty submits: each one is a trip through the GPU process, so spinning them
// back-to-back (interval 0) floods it. After 3 s without progress it falls back to interval 0.
export interface Pump {
  <T>(p: Promise<T>): Promise<T>;
  submits: number;
  interval: number;
}
export function makePump(device: GPUDevice, { interval = 1 }: { interval?: number } = {}): Pump {
  const mc = new MessageChannel(); const ticks: (() => void)[] = [];
  mc.port1.onmessage = () => ticks.shift()?.();
  const yieldTask = () => new Promise<void>((r) => { ticks.push(r); mc.port2.postMessage(0); });
  const f = async <T,>(p: Promise<T>): Promise<T> => {
    let done = false; p = p.finally(() => (done = true));
    const t0 = performance.now(); let last = t0;
    while (!done) {
      await yieldTask(); if (done) break;
      const now = performance.now(), iv = now - t0 > 3000 ? 0 : f.interval;
      if (now - last >= iv) { device.queue.submit([]); f.submits++; last = now; }
    }
    return p;
  };
  f.submits = 0; f.interval = interval;
  return f;
}

export type Engine = Awaited<ReturnType<typeof createEngine>>;
