// Bench prelude, prepended to the extracted workerMain. Not shipped.
self.__PROF = true;
self.__PUMP = new URLSearchParams(self.location.search).get("pump") === "1";
self.__runs = [];
const __now = () => performance.now();
// ---- GPU API counters + optional per-dispatch timestamp profiling ----
const G = { submit: 0, submitMs: 0, dispatch: 0, map: 0, mapMs: 0, mapBytes: 0, write: 0, writeMs: 0, writeBytes: 0, pipe: 0, pipeMs: 0, shader: 0, shaderMs: 0 };
self.__G = G;
const modLabel = new WeakMap();
const pipeLabel = new WeakMap();
let shaderSeq = 0;
{
    const P = GPUDevice.prototype;
    const csm = P.createShaderModule;
    P.createShaderModule = function (d) {
        const t = __now();
        const m = csm.call(this, d);
        G.shader++; G.shaderMs += __now() - t;
        const ep = /fn\s+(\w+)\s*\(/.exec(d.code || "");
        const L = d.label || `shader#${++shaderSeq}`;
        modLabel.set(m, L);
        (self.__shaderSrc ||= {})[L] = d.code || "";
        return m;
    };
    const lab = (d) => (d.label ? d.label : modLabel.get(d.compute?.module) || "?");
    const ccp = P.createComputePipeline;
    P.createComputePipeline = function (d) {
        const t = __now();
        const p = ccp.call(this, d);
        G.pipe++; G.pipeMs += __now() - t;
        pipeLabel.set(p, lab(d));
        return p;
    };
    const ccpa = P.createComputePipelineAsync;
    P.createComputePipelineAsync = async function (d) {
        const t = __now();
        const p = await ccpa.call(this, d);
        G.pipe++; G.pipeMs += __now() - t;
        pipeLabel.set(p, lab(d));
        return p;
    };
    const Q = GPUQueue.prototype;
    const sub = Q.submit;
    Q.submit = function (cbs) {
        const t = __now();
        const r = sub.call(this, cbs);
        G.submit++; G.submitMs += __now() - t;
        if (prof.on) for (const cb of cbs) prof.afterSubmit(this, cb);
        return r;
    };
    const wb = Q.writeBuffer;
    Q.writeBuffer = function (b, o, data, ...rest) {
        const t = __now();
        const r = wb.call(this, b, o, data, ...rest);
        G.write++; G.writeMs += __now() - t; G.writeBytes += rest[1] ?? data.byteLength ?? 0;
        return r;
    };
    const ma = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function (...a) {
        const t = __now();
        const r = await (self.__PUMP && self.__pumpWrap ? self.__pumpWrap(this, ma.apply(this, a)) : ma.apply(this, a));
        if (!this.__cbInternal) { G.map++; G.mapMs += __now() - t; G.mapBytes += a[2] ?? this.size; }
        return r;
    };
    const ad = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (desc = {}) {
        if (self.__PROF && this.features.has("timestamp-query")) desc = { ...desc, requiredFeatures: [...new Set([...(desc.requiredFeatures || []), "timestamp-query"])] };
        const dev = await ad.call(this, desc);
        self.__ortDevices = (self.__ortDevices || 0) + 1;
        self.__lastDevice = dev;
        self.__lastDeviceFeatures = [...dev.features];
        self.__lastAdapterInfo = this.info ? { vendor: this.info.vendor, architecture: this.info.architecture, device: this.info.device, description: this.info.description, isFallbackAdapter: this.info.isFallbackAdapter, subgroupMinSize: this.info.subgroupMinSize, subgroupMaxSize: this.info.subgroupMaxSize } : null;
        return dev;
    };
}
// Per-dispatch timestamps: every dispatch gets its own real compute pass.
const prof = {
    on: false,
    pending: new WeakMap(), // command buffer -> {qs, labels, buf}
    acc: new Map(), // label -> {n, ns}
    waits: [],
    afterSubmit(queue, cb) {
        const p = this.pending.get(cb);
        if (!p) return;
        this.waits.push(
            p.read.mapAsync(GPUMapMode.READ).then(() => {
                const a = new BigInt64Array(p.read.getMappedRange());
                for (let i = 0; i < p.labels.length; i++) {
                    const ns = Number(a[2 * i + 1] - a[2 * i]);
                    const e = this.acc.get(p.labels[i]) || { n: 0, ns: 0 };
                    e.n++; e.ns += ns > 0 && ns < 1e10 ? ns : 0;
                    this.acc.set(p.labels[i], e);
                }
                p.read.unmap(); p.read.destroy(); p.res.destroy(); p.qs.destroy();
            }),
        );
    },
};
{
    const E = GPUCommandEncoder.prototype;
    const bcp = E.beginComputePass;
    const fin = E.finish;
    E.beginComputePass = function (desc) {
        if (!prof.on || !this.device?.features?.has?.("timestamp-query")) return bcp.call(this, desc);
        const enc = this;
        const dev = enc.device;
        const st = (enc.__cbProf ||= { labels: [], qs: dev.createQuerySet({ type: "timestamp", count: 4096 }) });
        let real = null;
        let pipe = null;
        const bgs = [];
        const open = () => {
            const i = st.labels.length;
            if (i >= 2048) return bcp.call(enc, desc);
            st.labels.push(pipeLabel.get(pipe) || "?");
            return bcp.call(enc, { ...(desc || {}), timestampWrites: { querySet: st.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        };
        return {
            setPipeline(p) { pipe = p; },
            setBindGroup(i, g, ...r) { bgs[i] = [g, ...r]; },
            dispatchWorkgroups(x, y, z) {
                real = open();
                real.setPipeline(pipe);
                bgs.forEach((b, i) => b && real.setBindGroup(i, ...b));
                real.dispatchWorkgroups(x, y, z);
                real.end();
                G.dispatch++;
            },
            dispatchWorkgroupsIndirect(b, o) {
                real = open();
                real.setPipeline(pipe);
                bgs.forEach((g, i) => g && real.setBindGroup(i, ...g));
                real.dispatchWorkgroupsIndirect(b, o);
                real.end();
                G.dispatch++;
            },
            end() {},
            pushDebugGroup() {}, popDebugGroup() {}, insertDebugMarker() {},
            set label(v) {}, get label() { return ""; },
        };
    };
    E.finish = function (d) {
        const st = this.__cbProf;
        if (st && st.labels.length) {
            const n = st.labels.length;
            const dev = this.device;
            const res = dev.createBuffer({ size: n * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            const read = dev.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            read.__cbInternal = true;
            this.resolveQuerySet(st.qs, 0, 2 * n, res, 0);
            this.copyBufferToBuffer(res, 0, read, 0, n * 16);
            const cb = fin.call(this, d);
            prof.pending.set(cb, { labels: st.labels, qs: st.qs, res, read });
            return cb;
        }
        if (st) st.qs.destroy();
        return fin.call(this, d);
    };
    const cce = GPUDevice.prototype.createCommandEncoder;
    GPUDevice.prototype.createCommandEncoder = function (d) {
        const e = cce.call(this, d);
        e.device = this;
        return e;
    };
}

self.__instr = (ort) => {
    self.__ortRef = ort;
    const R = ort.InferenceSession.prototype.run;
    ort.InferenceSession.prototype.run = async function (feeds, ...a) {
        const k = "mel" in feeds ? "encoder" : "past_keys" in feeds ? "decoder_step" : "input_embeds" in feeds ? "decoder_init" : "vad";
        const t = __now();
        const r = await R.call(this, feeds, ...a);
        self.__runs.push([k, __now() - t]);
        return r;
    };
    // count CPU-side argmax/readback of logits
    const GD = ort.Tensor.prototype.getData;
    ort.Tensor.prototype.getData = async function (...a) {
        const t = __now();
        const r = await GD.apply(this, a);
        self.__runs.push(["getData:" + (this.location || "cpu"), __now() - t]);
        return r;
    };
};

const sumRuns = () => {
    const o = {};
    for (const [k, ms] of self.__runs) {
        const e = (o[k] ||= { n: 0, ms: 0, list: [] });
        e.n++; e.ms += ms; e.list.push(+ms.toFixed(1));
    }
    for (const e of Object.values(o)) { e.ms = +e.ms.toFixed(1); if (e.list.length > 40) e.list = e.list.slice(0, 40); }
    return o;
};
const snapG = () => ({ ...G });
const diffG = (a) => Object.fromEntries(Object.entries(G).map(([k, v]) => [k, +(v - a[k]).toFixed(1)]));

async function probes() {
    const ad = await navigator.gpu.requestAdapter();
    const feats = [...ad.features].filter((f) => ad.features.has(f));
    const lim = {};
    for (const k of ["maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup", "maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupSizeX", "maxStorageBuffersPerShaderStage", "maxComputeWorkgroupsPerDimension"]) lim[k] = ad.limits[k];
    const info = ad.info ? { vendor: ad.info.vendor, architecture: ad.info.architecture, device: ad.info.device, description: ad.info.description, isFallbackAdapter: ad.info.isFallbackAdapter, subgroupMinSize: ad.info.subgroupMinSize, subgroupMaxSize: ad.info.subgroupMaxSize } : null;
    const adHP = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const adLP = await navigator.gpu.requestAdapter({ powerPreference: "low-power" });
    const inf = (a) => a && a.info && `${a.info.vendor}/${a.info.architecture}/${a.info.description}`;
    const dev = await ad.requestDevice({ requiredFeatures: feats.filter((f) => ["shader-f16", "subgroups", "timestamp-query"].includes(f)) });
    const q = dev.queue;
    const med = (a) => { a = [...a].sort((x, y) => x - y); return +a[a.length >> 1].toFixed(3); };
    // empty dispatch + submit + completion
    const sm = dev.createShaderModule({ code: "@group(0) @binding(0) var<storage, read_write> o: array<u32>; @compute @workgroup_size(1) fn main() { o[0] = o[0] + 1u; }" });
    const buf = dev.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const tp0 = __now();
    const pipe = await dev.createComputePipelineAsync({ layout: "auto", compute: { module: sm, entryPoint: "main" } });
    const firstPipeMs = __now() - tp0;
    const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const rt = [];
    for (let i = 0; i < 60; i++) {
        const t = __now();
        const e = dev.createCommandEncoder();
        const p = e.beginComputePass();
        p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end();
        q.submit([e.finish()]);
        await q.onSubmittedWorkDone();
        rt.push(__now() - t);
    }
    // 50 dispatches in one submit (like one decoder step's batch)
    const rt50 = [];
    for (let i = 0; i < 20; i++) {
        const t = __now();
        const e = dev.createCommandEncoder();
        for (let j = 0; j < 50; j++) { const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end(); }
        q.submit([e.finish()]);
        await q.onSubmittedWorkDone();
        rt50.push(__now() - t);
    }
    // 50 submits of 1 dispatch, one wait
    const rt50s = [];
    for (let i = 0; i < 20; i++) {
        const t = __now();
        for (let j = 0; j < 50; j++) { const e = dev.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end(); q.submit([e.finish()]); }
        await q.onSubmittedWorkDone();
        rt50s.push(__now() - t);
    }
    // readback latency
    const readback = async (size, n) => {
        const src = dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const dst = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        dst.__cbInternal = true;
        const out = [];
        for (let i = 0; i < n; i++) {
            const t = __now();
            const e = dev.createCommandEncoder();
            e.copyBufferToBuffer(src, 0, dst, 0, size);
            q.submit([e.finish()]);
            await dst.mapAsync(GPUMapMode.READ);
            new Uint8Array(dst.getMappedRange()).slice(0, 16);
            dst.unmap();
            out.push(__now() - t);
        }
        src.destroy(); dst.destroy();
        return med(out);
    };
    const rb16 = await readback(16, 40);
    const rbLogits = await readback(151936 * 2, 30); // one f16 logits row
    const rbKV = await readback(16 << 20, 10); // KV-sized
    // writeBuffer throughput
    const wbuf = dev.createBuffer({ size: 16 << 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const data = new Uint8Array(16 << 20);
    const wt = [];
    for (let i = 0; i < 8; i++) { const t = __now(); q.writeBuffer(wbuf, 0, data); await q.onSubmittedWorkDone(); wt.push(__now() - t); }
    const smallW = [];
    const small = new Uint8Array(4096);
    for (let i = 0; i < 50; i++) { const t = __now(); q.writeBuffer(wbuf, 0, small); await q.onSubmittedWorkDone(); smallW.push(__now() - t); }
    // first-use pipeline creation: a fresh, moderately big f16 matmul-ish shader
    const hasF16 = dev.features.has("shader-f16");
    const bigCode = `${hasF16 ? "enable f16;" : ""}
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;
var<workgroup> tile: array<vec4<f32>, 256>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var acc = vec4<f32>(0.0);
  for (var k = 0u; k < 64u; k++) { tile[li] = a[wg.x * 64u + li + k]; workgroupBarrier(); for (var j = 0u; j < 64u; j++) { acc += tile[j] * b[k * 64u + j]; } workgroupBarrier(); }
  c[wg.x * 64u + li] = acc; }`;
    const pc = [];
    for (let i = 0; i < 5; i++) {
        const t = __now();
        const m = dev.createShaderModule({ code: bigCode + `\n// v${i}${Math.random()}` });
        await dev.createComputePipelineAsync({ layout: "auto", compute: { module: m, entryPoint: "main" } });
        pc.push(__now() - t);
    }
    dev.destroy();
    return {
        adapter: info, features: feats, limits: lim, hp: inf(adHP), lp: inf(adLP),
        emptyRoundTripMs: med(rt), dispatch50OneSubmitMs: med(rt50), submit50Ms: med(rt50s),
        readback16BMs: rb16, readbackLogits300KBMs: rbLogits, readback16MBMs: rbKV,
        write16MBMs: med(wt), write16MBGBps: +((16 / 1024) / (med(wt) / 1000)).toFixed(2), write4KBRoundTripMs: med(smallW),
        firstPipelineTinyMs: +firstPipeMs.toFixed(2), pipelineMatmulMs: pc.map((x) => +x.toFixed(1)),
    };
}

self.__bench = async (m, api) => {
    if (m.mode === "probe") return { probes: await probes() };
    if (m.mode === "src") return { src: Object.fromEntries((m.keys || []).map((k) => [k, (self.__shaderSrc || {})[k]])) };
    if (m.mode === "ortdev") return { ortDevices: self.__ortDevices, features: self.__lastDeviceFeatures, adapter: self.__lastAdapterInfo, limits: self.__lastDevice && { maxComputeWorkgroupStorageSize: self.__lastDevice.limits.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: self.__lastDevice.limits.maxStorageBufferBindingSize, maxBufferSize: self.__lastDevice.limits.maxBufferSize } };
    const pcm = new Float32Array(m.pcm);
    const out = [];
    prof.on = !!m.prof;
    prof.acc.clear();
    for (let i = 0; i < (m.n || 1); i++) {
        self.__runs = [];
        const g0 = snapG();
        const t = __now();
        let r;
        if (m.mode === "run") r = await api.transcribe(pcm, "ko", m.maxTokens || 96, 0, !!m.pace);
        else {
            api.resetStream();
            const hops = [];
            for (let h = 0; h < 8; h++) {
                self.__runs = [];
                const th = __now();
                const x = await api.streamHop({ pcm: pcm.slice(h * 16000, (h + 1) * 16000), reset: h === 0, quiet: false, lang: "ko", maxTokens: Math.round(16 + 12 * 1), pace: !!m.pace, vad: true, gapMs: 0 });
                hops.push({ wall: +(__now() - th).toFixed(1), tokens: x.tokens, prefill: x.prefill, encMs: x.encMs && +x.encMs.toFixed(1), vadMs: x.vadMs && +x.vadMs.toFixed(1), decodeMs: x.decodeMs && +x.decodeMs.toFixed(1), runs: sumRuns(), text: (x.conf || "") + (x.tent || "") });
            }
            r = { hops };
        }
        await Promise.all(prof.waits.splice(0));
        out.push({ wall: +(__now() - t).toFixed(1), r: m.mode === "run" ? { ...r, text: r.text } : r, runs: m.mode === "run" ? sumRuns() : undefined, gpu: diffG(g0) });
    }
    prof.on = false;
    const kernels = [...prof.acc.entries()].map(([k, v]) => ({ k, n: v.n, ms: +(v.ns / 1e6).toFixed(2) })).sort((a, b) => b.ms - a.ms);
    return { out, wideHits: self.__wideHits || 0, kernels: m.prof ? kernels : undefined };
};
// experimental pump: keep wgpu polling while a map is pending
{
    const bufDev = new WeakMap();
    const cb = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function (d) { const b = cb.call(this, d); bufDev.set(b, this); return b; };
    const mc = new MessageChannel();
    const ticks = [];
    mc.port1.onmessage = () => ticks.shift()?.();
    const yieldTask = () => new Promise((r) => { ticks.push(r); mc.port2.postMessage(0); });
    self.__pumpWrap = async (buf, p) => {
        const dev = bufDev.get(buf);
        if (!dev) return p;
        let done = false;
        p = p.finally(() => (done = true));
        while (!done) { await yieldTask(); if (!done) dev.queue.submit([]); }
        return p;
    };
}


    function closeBlocks(st, block, keep, rollback, histKeep) {
        const slid = [];
        while (st.open.length >= block) {
            st.closed.push({ pcm: st.open.slice(0, block), af: null, mark: Math.max(0, st.tokens.length - rollback) });
            st.open = st.open.slice(block);
            while (st.closed.length > keep) {
                const gone = st.closed.shift();
                const n = Math.min(gone.mark, Math.max(0, st.tokens.length - rollback));
                slid.push(...st.tokens.splice(0, n));
                for (const b of st.closed) b.mark = Math.max(0, b.mark - n);
            }
        }
        if (slid.length) st.histIds = [...st.histIds, ...slid].slice(-histKeep);
        return slid;
    }
    function utf8Cut(bytes, at) {
        at = Math.max(0, Math.min(at, bytes.length));
        while (at > 0 && at < bytes.length && (bytes[at] & 0xc0) === 0x80) at--;
        return at;
    }
(function workerMain() {
        const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
        const VAD_URL = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/silero_vad_v5.onnx";
        const VAD_THRESHOLD = 0.5;
        const VAD_MIN_SPEECH_SEC = 0.15;
        const QUIET_RESET_SEC = 1.5;
        const HIST_CONTEXT = 24;
        let ort;
        let model = null;
        let f16 = false;
        const post = (m, t) => self.postMessage(m, t || []);
        // Firefox's Error.stack holds only frames, never the message, and a
        // thrown value need not be an Error, so every field travels apart.
        let stage = "idle";
        const errFields = (error) => ({ stage, name: error?.name ?? typeof error, message: error?.message ?? String(error), stack: error?.stack ?? "" });

        // Whisper-compatible log-mel with a mixed-radix FFT for n_fft = 400.
        const N = 400;
        const HOP = 160;
        const COS = new Float64Array(N);
        const SIN = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            COS[i] = Math.cos((2 * Math.PI * i) / N);
            SIN[i] = Math.sin((2 * Math.PI * i) / N);
        }
        function fft(re, im, n) {
            if (n === 1) return;
            let p = 2;
            while (n % p) p++;
            const m = n / p;
            const subRe = [];
            const subIm = [];
            for (let r = 0; r < p; r++) {
                const sr = new Float64Array(m);
                const si = new Float64Array(m);
                for (let k = 0; k < m; k++) {
                    sr[k] = re[k * p + r];
                    si[k] = im[k * p + r];
                }
                fft(sr, si, m);
                subRe.push(sr);
                subIm.push(si);
            }
            const step = N / n;
            for (let k = 0; k < n; k++) {
                let accR = 0;
                let accI = 0;
                const km = k % m;
                for (let r = 0; r < p; r++) {
                    const idx = ((r * k) % n) * step;
                    const c = COS[idx];
                    const s = -SIN[idx];
                    const xr = subRe[r][km];
                    const xi = subIm[r][km];
                    accR += xr * c - xi * s;
                    accI += xr * s + xi * c;
                }
                re[k] = accR;
                im[k] = accI;
            }
        }
        self.__fft = fft;
        const HANN = new Float64Array(N);
        for (let i = 0; i < N; i++) HANN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
        function logMel(pcm, filters) {
            const P = N / 2;
            const len = pcm.length;
            const x = new Float32Array(len + 2 * P);
            x.set(pcm, P);
            for (let i = 0; i < P; i++) {
                x[P - 1 - i] = pcm[i + 1];
                x[P + len + i] = pcm[len - 2 - i];
            }
            const T = Math.floor(len / HOP); // last frame dropped
            const out = new Float32Array(128 * T);
            const re = new Float64Array(N);
            const im = new Float64Array(N);
            const pw = new Float64Array(201);
            let mx = -Infinity;
            for (let t = 0; t < T; t++) {
                for (let n = 0; n < N; n++) {
                    re[n] = x[t * HOP + n] * HANN[n];
                    im[n] = 0;
                }
                fft(re, im, N);
                for (let k = 0; k < 201; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
                for (let m = 0; m < 128; m++) {
                    const f = filters[m];
                    let s = 0;
                    for (let k = 0; k < 201; k++) s += f[k] * pw[k];
                    const v = Math.log10(Math.max(s, 1e-10));
                    out[m * T + t] = v;
                    if (v > mx) mx = v;
                }
            }
            for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], mx - 8) + 4) / 4;
            return { data: out, T };
        }

        const f2h = (() => {
            const f = new Float32Array(1);
            const u = new Uint32Array(f.buffer);
            return (v) => {
                f[0] = v;
                const x = u[0];
                const s = (x >>> 16) & 0x8000;
                const e = ((x >>> 23) & 0xff) - 112;
                const m = x & 0x7fffff;
                if (e <= 0) return s;
                if (e >= 31) return s | 0x7c00;
                return s | (e << 10) | ((m + 0x1000) >>> 13);
            };
        })();
        const h2f = (v) => {
            const e = (v >> 10) & 31;
            const f = v & 1023;
            const a = e === 0 ? (f / 1024) * 2 ** -14 : 2 ** (e - 15) * (1 + f / 1024);
            return v & 0x8000 ? -a : a;
        };

        // The page closes the gate while playback needs the machine (thin
        // video buffer, a stall, a hidden tab); loading waits at the next
        // chunk or step boundary until it reopens.
        let gateOpen = true;
        let gateWaiters = [];
        const gate = () => (gateOpen ? Promise.resolve() : new Promise((r) => gateWaiters.push(r)));
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

        // Frame pacing: GPU work goes out in slices, one slice per display
        // frame, sized to a fraction of the frame so the compositor and the
        // video decoder get the rest. The worker's own requestAnimationFrame
        // is used where it exists; otherwise the page posts a tick per frame.
        // Hidden tabs get no frames, which also parks the work.
        const workerRaf = typeof self.requestAnimationFrame === "function";
        let tickWaiters = [];
        const pacer = { k: 1, frameMs: 16.7, last: 0, sliceStart: 0, inSlice: 0, frames: 0 };
        const SLICE_SHARE = 0.35;
        const nextFrame = () =>
            new Promise((resolve) => {
                const done = (t) => {
                    const now = t ?? performance.now();
                    if (pacer.last) pacer.frameMs += (Math.min(100, Math.max(4, now - pacer.last)) - pacer.frameMs) * 0.1;
                    pacer.last = now;
                    pacer.frames++;
                    resolve();
                };
                if (workerRaf) self.requestAnimationFrame(done);
                else tickWaiters.push(done);
            });
        // Call before each unit of GPU work (one decoder step, or a whole
        // encoder pass with k = 1); waits for a frame when the slice is full
        // and adapts k to the time the last slice took.
        async function paced(enabled, unitIsWhole = false) {
            if (!enabled) return;
            if (pacer.inSlice > 0 && (unitIsWhole || pacer.inSlice >= pacer.k)) {
                const took = performance.now() - pacer.sliceStart;
                if (!unitIsWhole) {
                    if (took > SLICE_SHARE * pacer.frameMs && pacer.k > 1) pacer.k--;
                    else if (took < 0.5 * SLICE_SHARE * pacer.frameMs) pacer.k = Math.min(16, pacer.k + 1);
                }
                pacer.inSlice = 0;
            }
            if (pacer.inSlice === 0) {
                await nextFrame();
                pacer.sliceStart = performance.now();
            }
            pacer.inSlice++;
        }

        // One copy in memory: the body is teed into the cache as it streams
        // and into a buffer sized from content-length.
        // Firefox can commit a cache entry whose streamed body was cut off
        // (worker terminated or tab closed mid-download), and the stored
        // content-length is the compressed size when the host sends br/gzip,
        // so neither proves a hit is whole. A separate entry, written only
        // after the body finished, records the decoded byte count; a hit
        // without a matching record is dropped and fetched again, unless it
        // matches the size the Hub lists (entries cached before the record).
        const doneKey = (url) => `https://chzzkbest.invalid/complete?${encodeURIComponent(url)}`;
        // Firefox's WGSL front end (naga) types bitcast<vec2<f16>>(u32) as a
        // scalar, so ORT's f16 kernels that index the result (Pad reads its
        // constant that way) fail to compile and every encoder run throws
        // "WebGPU validation failed ... Invalid access into expression".
        // unpack2x16float yields the same two halves exactly, so where a probe
        // shader shows the gap, ORT's shaders are rewritten before compiling.
        // Some WebGPU implementations (Firefox's wgpu, measured) only
        // resolve mapAsync on a ~100 ms poll timer unless the queue sees new
        // work, which turns every per-token logits readback into a 100 ms
        // stall. A timed 16-byte readback detects that, and if it is slow,
        // an empty submit per task while a map is pending drives the poll:
        // tight for the first 20 ms (decode-step readbacks), then every 4 ms.
        async function installMapPump() {
            if (GPUBuffer.prototype.mapAsync.cbPumped) return;
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return;
            const device = await adapter.requestDevice();
            const src = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
            const dst = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            const times = [];
            for (let i = 0; i < 3; i++) {
                const t = performance.now();
                const e = device.createCommandEncoder();
                e.copyBufferToBuffer(src, 0, dst, 0, 16);
                device.queue.submit([e.finish()]);
                await dst.mapAsync(GPUMapMode.READ);
                dst.unmap();
                times.push(performance.now() - t);
            }
            device.destroy();
            const ms = times.sort((a, b) => a - b)[1];
            if (ms < 25) return;
            const owner = new WeakMap();
            const createBuffer = GPUDevice.prototype.createBuffer;
            GPUDevice.prototype.createBuffer = function (desc) {
                const b = createBuffer.call(this, desc);
                owner.set(b, this);
                return b;
            };
            const ch = new MessageChannel();
            const waiting = [];
            ch.port1.onmessage = () => waiting.shift()?.();
            const nextTask = () => new Promise((r) => (waiting.push(r), ch.port2.postMessage(0)));
            const mapAsync = GPUBuffer.prototype.mapAsync;
            const pumped = function (...args) {
                const p = mapAsync.apply(this, args);
                const device = owner.get(this);
                if (!device) return p;
                let pending = true;
                const settle = () => (pending = false);
                p.then(settle, settle);
                (async () => {
                    const t0 = performance.now();
                    while (pending) {
                        await sleep(4);
                        if (pending) device.queue.submit([]);
                    }
                })();
                return p;
            };
            pumped.cbPumped = true;
            GPUBuffer.prototype.mapAsync = pumped;
            console.info(`[ChzzkBest] stt mapAsync pump on (16 B readback took ${ms.toFixed(0)} ms)`);
        }

        // Firefox (naga -> HLSL) runs ORT's MatMulNBits wide-tile kernel ~5x slower than Chrome; a fully
        // unrolled main with scalar accumulators is bit-identical and closes the gap (shadertune probe).
        function patchMatMulNBitsWide() {
            const P = GPUDevice.prototype;
            if (P.createShaderModule.cbWidePatched) return;
            const need = ["const kTileM : u32 = 16;", "const kTileN : u32 = 128;", "const KAVecSizeForBlock32 = 8u;", "const workgroup_size_x: u32 = 128;", "var results : array<f32, kTileM>;", "results[m_idx] += f32(dot(a_data0, b_dequantized[0])) +", "fn dequantize(packed_data : u32,", "fn load_zero(row : u32, col : u32, r_dim : u32, c_dim : u32)"];
            const L = ["@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)", "fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_index) local_idx : u32, @builtin(num_workgroups) num_workgroups : vec3<u32>) {",
                "let workgroup_idx = workgroup_id.z * num_workgroups[0] * num_workgroups[1] + workgroup_id.y * num_workgroups[0] + workgroup_id.x;",
                "let batch = workgroup_idx / (uniforms.num_M_tile * uniforms.num_N_tile);", "let row = ((workgroup_idx / uniforms.num_N_tile) % uniforms.num_M_tile) * kTileM;", "let col = (workgroup_idx % uniforms.num_N_tile) * kTileN;"];
            for (let m = 0; m < 16; m++) L.push(`var r${m} : f32 = 0.0;`);
            L.push("let a_row_idx = local_idx / KAVecSizeForBlock32;", "let a_col_idx = local_idx % KAVecSizeForBlock32;", "let b_row = col + local_idx;", "for (var block_idx = 0u; block_idx < uniforms.n_blocks_per_col; block_idx++) {",
                "a_data_tile[a_row_idx][a_col_idx] = load_a(batch, row + a_row_idx, block_idx * KAVecSizeForBlock32 + a_col_idx);", "workgroupBarrier();",
                "let scale = load_scale(b_row, block_idx);", "let zero_point = load_zero(b_row, block_idx, uniforms.N, uniforms.zero_blocks_per_col);", "let b_data = load_b(b_row, block_idx);");
            for (let b = 0; b < 4; b++) {
                L.push(`let bd${b} = dequantize(b_data.${"xyzw"[b]}, zero_point, scale);`, `let d0_${b} = bd${b}[0];`, `let d1_${b} = bd${b}[1];`);
                for (let m = 0; m < 16; m++) L.push(`r${m} += f32(dot(a_data_tile[${m}][${2 * b}], d0_${b})) + f32(dot(a_data_tile[${m}][${2 * b + 1}], d1_${b}));`);
            }
            L.push("workgroupBarrier();", "}");
            for (let m = 0; m < 16; m++) L.push(`write_output(batch, row + ${m}u, col + local_idx, output_element_t(r${m}));`);
            L.push("}");
            const main = L.join("\n") + "\n";
            const create = P.createShaderModule;
            const patched = function (desc) {
                const c = desc?.code;
                if (c && need.every((s) => c.includes(s))) { const i = c.indexOf("@compute"); if (i > 0) { desc = { ...desc, code: c.slice(0, i) + main }; self.__wideHits = (self.__wideHits || 0) + 1; } }
                return create.call(this, desc);
            };
            patched.cbWidePatched = true;
            P.createShaderModule = patched;
        }
        async function patchF16Bitcast(adapter) {
            const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
            const probe = device.createShaderModule({
                code: "enable f16;\n@group(0) @binding(0) var<storage, read_write> o: array<f16>;\n@compute @workgroup_size(1) fn main() { o[0] = bitcast<vec2<f16>>(0x3c00u)[0]; }",
            });
            const broken = (await probe.getCompilationInfo()).messages.some((m) => m.type === "error");
            device.destroy();
            if (!broken || GPUDevice.prototype.createShaderModule.cbPatched) return;
            const create = GPUDevice.prototype.createShaderModule;
            const patched = function (desc) {
                if (desc?.code?.includes("bitcast<vec2<f16>>")) desc = { ...desc, code: desc.code.replace(/bitcast<vec2<f16>>\(([^()]*)\)/g, "vec2<f16>(unpack2x16float($1))") };
                return create.call(this, desc);
            };
            patched.cbPatched = true;
            GPUDevice.prototype.createShaderModule = patched;
            console.info("[ChzzkBest] stt WGSL bitcast<vec2<f16>> rewrite on (naga gap)");
        }
        // Downloads stream straight into Cache Storage, never into a JS
        // buffer, so every file can be in flight at once; a session reads
        // its files back only right before it is created. A hit counts as
        // present when it has a completion record or a Hub size to check
        // against; the byte count itself is checked on read-back.
        async function ensureCached(cache, url, onBytes, expect = 0, force = false) {
            if (!force) {
                const hit = await cache.match(url);
                const done = hit && (await cache.match(doneKey(url)));
                if (hit && (done || expect > 0)) {
                    onBytes(done ? Number(await done.text()) : expect, true);
                    return;
                }
            }
            await cache.delete(doneKey(url));
            await gate();
            const res = await fetch(url);
            if (!res.ok) throw new Error(`${url} → HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
            let n = 0;
            // Not pulling back-pressures the socket, so a closed gate also
            // stops the download from competing with the stream.
            const count = new TransformStream({
                async transform(chunk, ctl) {
                    await gate();
                    n += chunk.byteLength;
                    onBytes(chunk.byteLength, false);
                    ctl.enqueue(chunk);
                },
            });
            await cache.put(url, new Response(res.body.pipeThrough(count)));
            await cache.put(doneKey(url), new Response(String(n)));
        }
        async function readCached(cache, url, onBytes, expect = 0) {
            for (let attempt = 0; ; attempt++) {
                const hit = await cache.match(url);
                const done = await cache.match(doneKey(url));
                const want = done ? Number(await done.text()) : expect;
                const buf = hit && (await hit.arrayBuffer());
                if (buf && want > 0 && buf.byteLength === want) {
                    if (!done) await cache.put(doneKey(url), new Response(String(want)));
                    return buf;
                }
                if (attempt) throw new Error(`${url}: cached ${buf?.byteLength ?? 0} B, expected ${want} B`);
                console.warn("[ChzzkBest] stt cache entry incomplete, refetching", url, buf?.byteLength ?? 0);
                await cache.delete(url);
                await ensureCached(cache, url, onBytes, expect, true);
            }
        }
        async function fetchCached(cache, url, onBytes, expect = 0) {
            await ensureCached(cache, url, onBytes, expect);
            await gate();
            return readCached(cache, url, onBytes, expect);
        }

        async function load({ repo, rev, cacheName }) {
            const t0 = performance.now();
            post({ type: "caps", workerRaf });
            ort = await import(ORT + "ort.webgpu.bundle.min.mjs");
            ort.env.wasm.wasmPaths = ORT;
            ort.env.wasm.numThreads = 1; self.__instr(ort);
            if (!navigator.gpu) throw new Error("WebGPU 없음 (navigator.gpu undefined)");
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error("WebGPU 어댑터 없음");
            f16 = adapter.features.has("shader-f16");
            patchMatMulNBitsWide(); if (f16) await patchF16Bitcast(adapter);
            await installMapPump();
            const base = `https://huggingface.co/${repo}/resolve/${rev}/`;
            const V = f16
                ? { enc: "encoder.fp16.onnx", init: "decoder_init.q4f16.onnx", step: "decoder_step.q4f16.onnx", w: "decoder_weights.q4f16.data" }
                : { enc: "encoder.onnx", init: "decoder_init.int4.onnx", step: "decoder_step.int4.onnx", w: "decoder_weights.int4.data" };
            const cache = await caches.open(cacheName);
            // File sizes for the progress bar; cached with the weights so a
            // warm start makes no network request at all.
            let meta = null;
            try {
                const buf = await fetchCached(cache, `https://huggingface.co/api/models/${repo}/revision/${rev}?blobs=true`, () => {});
                meta = JSON.parse(new TextDecoder().decode(buf));
            } catch (error) {
                console.warn("[ChzzkBest] stt size metadata unavailable", error);
            }
            const sizes = {};
            for (const s of meta?.siblings ?? []) sizes[s.rfilename] = s.size;
            const files = ["prompt_config.json", "config.json", "mel_filters.json", "vocab.json", "added_tokens.json", "embed_tokens.int8.bin", "embed_scales.f32.bin", V.enc, V.init, V.step, V.w];
            const total = files.reduce((a, f) => a + (sizes[f] || 0), 0);
            let loaded = 0;
            let fromNet = 0;
            let lastPost = 0;
            const bufs = {};
            let createMs = 0;
            const onBytes = (n, cached) => {
                loaded += n;
                if (!cached) fromNet += n;
                const now = performance.now();
                if (now - lastPost > 250) {
                    lastPost = now;
                    post({ type: "progress", loaded, total, fromNet });
                }
            };
            // Every download starts now; sessions are still built one at a
            // time in this order, each as soon as its own files are cached,
            // so only one large buffer is alive at once and shader
            // compilation comes in separate, gated bursts.
            const cached = Object.fromEntries(files.map((f) => [f, ensureCached(cache, base + f, onBytes, sizes[f])]));
            for (const p of Object.values(cached)) p.catch(() => {});
            const get = async (f) => {
                await cached[f];
                await gate();
                bufs[f] = await readCached(cache, base + f, onBytes, sizes[f]);
                return bufs[f];
            };
            const create = async (bytes, options) => {
                await gate();
                await sleep(1000);
                await gate();
                const t = performance.now();
                const session = await ort.InferenceSession.create(new Uint8Array(bytes), options);
                createMs += performance.now() - t;
                return session;
            };
            stage = "load:VAD";
            await loadVad(cache);
            stage = "load:files";
            for (const f of files.slice(0, 7)) await get(f);
            const json = (f) => JSON.parse(new TextDecoder().decode(bufs[f]));
            const cfg = json("prompt_config.json");
            const D = json("config.json").decoder.hidden_size;
            const filters = json("mel_filters.json").data;
            const vocab = json("vocab.json");
            const opt = { executionProviders: ["webgpu"] };
            stage = "load:encoder";
            const enc = await create(await get(V.enc), opt);
            delete bufs[V.enc];
            const w = new Uint8Array(await get(V.w));
            const kvOpt = {
                ...opt,
                externalData: [{ path: V.w, data: w }],
                // f16 logits stay on the GPU; argmax reduces them there and reads back one u32.
                preferredOutputLocation: f16 ? { present_keys: "gpu-buffer", present_values: "gpu-buffer", logits: "gpu-buffer" } : { present_keys: "gpu-buffer", present_values: "gpu-buffer" },
            };
            stage = "load:decoder_init";
            const init = await create(await get(V.init), kvOpt);
            delete bufs[V.init];
            stage = "load:decoder_step";
            const step = await create(await get(V.step), kvOpt);
            delete bufs[V.step];
            delete bufs[V.w];
            post({ type: "progress", loaded, total, fromNet });
            const tDl = performance.now() - createMs;
            const embB = bufs["embed_tokens.int8.bin"];
            const emb = new Int8Array(embB);
            const sc = new Float32Array(bufs["embed_scales.f32.bin"]);
            // GPT-2 byte-level BPE alphabet → bytes.
            const byteOf = {};
            {
                const bs = [];
                for (let b = 33; b <= 126; b++) bs.push(b);
                for (let b = 161; b <= 172; b++) bs.push(b);
                for (let b = 174; b <= 255; b++) bs.push(b);
                const cs = [...bs];
                let n = 0;
                for (let b = 0; b < 256; b++)
                    if (!bs.includes(b)) {
                        bs.push(b);
                        cs.push(256 + n++);
                    }
                bs.forEach((b, i) => (byteOf[String.fromCharCode(cs[i])] = b));
            }
            const inv = [];
            for (const [k, v] of Object.entries(vocab)) inv[v] = k;
            model = { enc, init, step, emb, sc, cfg, D, filters, inv, byteOf };
            const tLoad = performance.now();
            post({ type: "ready", f16, downloadMs: tDl - t0, sessionMs: tLoad - tDl, fromNet, total });
        }

        const toT = (arr, dims) =>
            f16 ? new ort.Tensor("float16", Uint16Array.from(arr, f2h), dims) : new ort.Tensor("float32", arr, dims);
        // Greedy pick over f16 logits without reading the 300 KB row back:
        // one 256-thread workgroup reduces it, ties go to the lowest index
        // exactly like the CPU loop below.
        let gpuArgmaxState = null;
        async function gpuArgmax(t) {
            const dev = await ort.env.webgpu.device;
            let st = gpuArgmaxState;
            if (!st || st.dev !== dev) {
                const code = `@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> u: vec2<u32>;
var<workgroup> bv: array<f32, 256>; var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li: u32) {
  var v = -3.0e38; var idx = 0u;
  for (var i = li; i < u.y; i += 256u) { let p = unpack2x16float(x[u.x + i]); if (p.x > v) { v = p.x; idx = 2u * i; } if (p.y > v) { v = p.y; idx = 2u * i + 1u; } }
  bv[li] = v; bi[li] = idx; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { let ov = bv[li + s]; let oi = bi[li + s]; if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; } } workgroupBarrier(); }
  if (li == 0u) { o[0] = bi[0]; }
}`;
                const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code, label: "cb-argmax" }), entryPoint: "main" } });
                st = gpuArgmaxState = {
                    dev,
                    pipe,
                    out: dev.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
                    uni: dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
                    rb: dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
                };
            }
            const V = t.dims[t.dims.length - 1];
            const rows = t.dims.reduce((a, b) => a * b, 1) / V;
            dev.queue.writeBuffer(st.uni, 0, new Uint32Array([((rows - 1) * V) / 2, V / 2, 0, 0]));
            const bg = dev.createBindGroup({ layout: st.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: t.gpuBuffer } }, { binding: 1, resource: { buffer: st.out } }, { binding: 2, resource: { buffer: st.uni } }] });
            const e = dev.createCommandEncoder();
            const p = e.beginComputePass();
            p.setPipeline(st.pipe);
            p.setBindGroup(0, bg);
            p.dispatchWorkgroups(1);
            p.end();
            e.copyBufferToBuffer(st.out, 0, st.rb, 0, 4);
            dev.queue.submit([e.finish()]);
            await st.rb.mapAsync(GPUMapMode.READ, 0, 4);
            const r = new Uint32Array(st.rb.getMappedRange(0, 4))[0];
            st.rb.unmap();
            return r;
        }
        async function argmax(t) {
            if (t.location === "gpu-buffer") {
                try {
                    return await gpuArgmax(t);
                } finally {
                    t.dispose();
                }
            }
            const d = await t.getData();
            let bi = 0;
            let bv = -Infinity;
            const half = d instanceof Uint16Array;
            for (let i = 0; i < d.length; i++) {
                const v = half ? h2f(d[i]) : d[i];
                if (v > bv) {
                    bv = v;
                    bi = i;
                }
            }
            return bi;
        }
        function embedRow(id, out, off) {
            const { emb, sc, D } = model;
            const s = sc[id];
            const b = id * D;
            for (let d = 0; d < D; d++) out[off + d] = emb[b + d] * s;
        }

        async function transcribe(pcm, lang, maxTokens, gapMs = 0, pace = false) {
            pacer.inSlice = 0;
            const { enc, init, step, cfg, D, filters, inv, byteOf } = model;
            const t0 = performance.now();
            const mel = logMel(pcm, filters);
            await gate();
            const tMel = performance.now();
            await paced(pace, true);
            stage = "encoder";
            const eo = await enc.run({ mel: new ort.Tensor("float32", mel.data, [1, 128, mel.T]) });
            const af = await eo.audio_features.getData();
            const A = eo.audio_features.dims[1];
            eo.audio_features.dispose?.();
            const P = cfg.prompt;
            const ids = [...P.prefix_ids, ...Array(A).fill(P.audio_pad_id), ...P.suffix_ids, ...(cfg.language_prefix_ids[lang] || [])];
            const S = ids.length;
            const ie = new Float32Array(S * D);
            const audioAt = P.prefix_ids.length;
            ids.forEach((id, i) => {
                if (i >= audioAt && i < audioAt + A) return;
                embedRow(id, ie, i * D);
            });
            ie.set(af, audioAt * D);
            await paced(pace, true);
            stage = "decoder_init";
            let o = await init.run({
                input_embeds: toT(ie, [1, S, D]),
                position_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map((_, i) => BigInt(i))), [1, S]),
            });
            const tPre = performance.now();
            const gen = [];
            let pos = S;
            let tok = await argmax(o.logits);
            let pk = o.present_keys;
            let pv = o.present_values;
            const CAP = self.__cap;
            if (CAP) {
                const bytes = async (t) => { const d = await t.getData(); return new Uint8Array(d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength)); };
                Object.assign(CAP, { S, tok0: tok, gen, pk: await bytes(pk), pv: await bytes(pv), kvDims: pk.dims });
            }
            const e = new Float32Array(D);
            let aborted = false;
            while (!P.eos_ids.includes(tok) && gen.length < maxTokens) {
                // A closed gate abandons the window; the next one covers the
                // same audio, and playback gets the GPU back within a step.
                if (!gateOpen) {
                    aborted = true;
                    break;
                }
                gen.push(tok);
                // Stop a decoder stuck repeating one short phrase.
                if (gen.length >= 24) {
                    const tail = gen.slice(-12).join(",");
                    if (gen.slice(-24, -12).join(",") === tail) break;
                }
                embedRow(tok, e, 0);
                await paced(pace);
                stage = "decoder_step";
                const n = await step.run({
                    input_embeds: toT(e, [1, 1, D]),
                    position_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos++)]), [1, 1]),
                    past_keys: pk,
                    past_values: pv,
                });
                pk.dispose();
                pv.dispose();
                pk = n.present_keys;
                pv = n.present_values;
                if (CAP) {
                    const d = await n.logits.getData(); const half = d instanceof Uint16Array; const g = (i) => (half ? h2f(d[i]) : d[i]);
                    let b = 0, c = 1; if (g(c) > g(b)) [b, c] = [c, b];
                    for (let i = 2; i < d.length; i++) { const v = g(i); if (v > g(b)) { c = b; b = i; } else if (v > g(c)) c = i; }
                    CAP.top2.push([b, g(b), c, g(c)]);
                }
                tok = await argmax(n.logits);
                // Under load the page asks for breathing room between steps
                // so frames get GPU time mid-utterance, not only between runs.
                if (gapMs && gen.length % 4 === 0) await sleep(gapMs);
            }
            if (CAP) CAP.final = tok;
            pk.dispose();
            pv.dispose();
            const tEnd = performance.now();
            const bytes = [];
            for (const t of gen) {
                const piece = inv[t];
                if (piece === undefined || t >= 151643) continue;
                for (const c of piece) bytes.push(byteOf[c] ?? 63);
            }
            let text = new TextDecoder().decode(Uint8Array.from(bytes));
            const cut = text.lastIndexOf("<asr_text>");
            if (cut >= 0) text = text.slice(cut + 10);
            return {
                aborted,
                text: aborted ? "" : text.trim(),
                tokens: gen.length,
                melMs: tMel - t0,
                encPrefillMs: tPre - tMel,
                decodeMs: tEnd - tPre,
                totalMs: tEnd - t0,
                pace: pace ? { k: pacer.k, frameMs: +pacer.frameMs.toFixed(1), frames: pacer.frames } : null,
            };
        }

        // ---- voice activity: Silero VAD v5 on the CPU (wasm), never WebGPU,
        // so the gate costs the video nothing. State carries across hops.
        let vad = null;
        async function loadVad(cache) {
            try {
                const buf = await fetchCached(cache, VAD_URL, () => {});
                const session = await ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ["wasm"] });
                vad = { session, state: new Float32Array(256), ctx: new Float32Array(64), carry: new Float32Array(0) };
            } catch (error) {
                console.warn("[ChzzkBest] stt VAD unavailable; RMS gate only", errFields(error), error);
            }
        }
        async function vadSpeech(pcm) {
            if (!vad) return null;
            const t0 = performance.now();
            const x = new Float32Array(vad.carry.length + pcm.length);
            x.set(vad.carry);
            x.set(pcm, vad.carry.length);
            const sr = new ort.Tensor("int64", BigInt64Array.from([16000n]), []);
            const inp = new Float32Array(576);
            let speech = 0;
            let off = 0;
            for (; off + 512 <= x.length; off += 512) {
                inp.set(vad.ctx, 0);
                inp.set(x.subarray(off, off + 512), 64);
                stage = "VAD";
                const r = await vad.session.run({ input: new ort.Tensor("float32", inp.slice(), [1, 576]), state: new ort.Tensor("float32", vad.state, [2, 1, 128]), sr });
                vad.state = new Float32Array(await r.stateN.getData());
                vad.ctx = x.slice(off + 448, off + 512);
                if ((await r.output.getData())[0] >= VAD_THRESHOLD) speech++;
            }
            vad.carry = x.slice(off);
            return { speechSec: speech * 0.032, ms: performance.now() - t0 };
        }

        // ---- streaming decode (QwenLM streaming_transcribe scheme) --------
        // Each hop re-encodes only the open 8 s block, reuses the closed
        // block's features, forces the previous text minus its last
        // ROLLBACK tokens as the start of the answer and decodes only the
        // continuation. Older blocks slide out with their text (closeBlocks),
        // so compute and memory stay flat on an endless stream. A pause
        // (no speech for QUIET_RESET_SEC) ends the utterance.
        const BLOCK = 8 * 16000;
        const ROLLBACK = 5;
        const st = { open: new Float32Array(0), closed: [], tokens: [], histIds: [], quietSec: 0 };
        const resetStream = () => {
            st.open = new Float32Array(0);
            st.closed = [];
            st.tokens = [];
            st.quietSec = 0;
        };
        const bytesOf = (ids) => {
            const { inv, byteOf } = model;
            const bytes = [];
            for (const t of ids) {
                const piece = inv[t];
                if (piece === undefined || t >= 151643) continue;
                for (const c of piece) bytes.push(byteOf[c] ?? 63);
            }
            return Uint8Array.from(bytes);
        };
        const textOf = (ids) => new TextDecoder().decode(bytesOf(ids));
        async function encodeBlock(pcm) {
            const mel = logMel(pcm, model.filters);
            stage = "encoder";
            const eo = await model.enc.run({ mel: new ort.Tensor("float32", mel.data, [1, 128, mel.T]) });
            const af = new Float32Array(await eo.audio_features.getData());
            eo.audio_features.dispose?.();
            return af;
        }
        async function streamHop(m) {
            pacer.inSlice = 0;
            const t0 = performance.now();
            if (m.reset) resetStream();
            const appended = new Float32Array(st.open.length + m.pcm.length);
            appended.set(st.open);
            appended.set(m.pcm, st.open.length);
            st.open = appended;
            const v = m.quiet ? { speechSec: 0, ms: 0 } : await vadSpeech(m.pcm);
            const speech = m.quiet ? false : v ? v.speechSec >= VAD_MIN_SPEECH_SEC : true;
            const gated = m.quiet || (m.vad !== false && v !== null);
            const base = { stream: true, speechSec: v?.speechSec ?? null, vadMs: v?.ms ?? null, speech, decoded: false, hist: "", conf: null, tent: null };
            if (!speech && gated) {
                st.quietSec += m.pcm.length / 16000;
                if (!st.tokens.length && !st.closed.length) {
                    // Nothing said yet: keep only a short lead-in for the onset.
                    if (st.open.length > 8000) st.open = st.open.slice(-8000);
                    return { ...base, totalMs: performance.now() - t0 };
                }
                if (st.quietSec >= QUIET_RESET_SEC) {
                    const hist = textOf(st.tokens);
                    st.histIds = [...st.histIds, ...st.tokens].slice(-24);
                    resetStream();
                    return { ...base, hist, conf: "", tent: "", ids: [], final: true, totalMs: performance.now() - t0 };
                }
                // Hold the tentative text; the next speech hop re-decodes it.
                closeBlocks(st, BLOCK, 1, ROLLBACK, 24);
                return { ...base, totalMs: performance.now() - t0 };
            }
            st.quietSec = 0;
            const slid = closeBlocks(st, BLOCK, 1, ROLLBACK, 24);
            const { init, step, cfg, D } = model;
            await gate();
            const tMel = performance.now();
            let encMs = 0;
            const parts = [];
            for (const b of st.closed) {
                if (!b.af) {
                    await paced(m.pace, true);
                    const te = performance.now();
                    b.af = await encodeBlock(b.pcm);
                    encMs += performance.now() - te;
                }
                parts.push(b.af);
            }
            if (st.open.length >= 1600) {
                await paced(m.pace, true);
                const te = performance.now();
                parts.push(await encodeBlock(st.open));
                encMs += performance.now() - te;
            }
            const A = parts.reduce((a, p) => a + p.length / D, 0);
            const P = cfg.prompt;
            const forced = st.tokens.slice(0, Math.max(0, st.tokens.length - ROLLBACK));
            // History tail goes into the (otherwise empty) system turn.
            const pre = [...P.prefix_ids.slice(0, 3), ...st.histIds.slice(-HIST_CONTEXT), ...P.prefix_ids.slice(3)];
            const audioAt = pre.length;
            const ids = [...pre, ...Array(A).fill(P.audio_pad_id), ...P.suffix_ids, ...(cfg.language_prefix_ids[m.lang] || []), ...forced];
            const S = ids.length;
            const ie = new Float32Array(S * D);
            ids.forEach((id, i) => {
                if (i >= audioAt && i < audioAt + A) return;
                embedRow(id, ie, i * D);
            });
            let at = audioAt * D;
            for (const p of parts) {
                ie.set(p, at);
                at += p.length;
            }
            await paced(m.pace, true);
            stage = "decoder_init";
            const o = await init.run({
                input_embeds: toT(ie, [1, S, D]),
                position_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map((_, i) => BigInt(i))), [1, S]),
            });
            const tPre = performance.now();
            const gen = [];
            let pos = S;
            let tok = await argmax(o.logits);
            let pk = o.present_keys;
            let pv = o.present_values;
            const e = new Float32Array(D);
            let aborted = false;
            while (!P.eos_ids.includes(tok) && gen.length < m.maxTokens) {
                if (!gateOpen) {
                    aborted = true;
                    break;
                }
                gen.push(tok);
                if (gen.length >= 16) {
                    const tail = gen.slice(-8).join(",");
                    if (gen.slice(-16, -8).join(",") === tail) break;
                }
                embedRow(tok, e, 0);
                await paced(m.pace);
                stage = "decoder_step";
                const n = await step.run({
                    input_embeds: toT(e, [1, 1, D]),
                    position_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos++)]), [1, 1]),
                    past_keys: pk,
                    past_values: pv,
                });
                pk.dispose();
                pv.dispose();
                pk = n.present_keys;
                pv = n.present_values;
                tok = await argmax(n.logits);
                if (m.gapMs && gen.length % 4 === 0) await sleep(m.gapMs);
            }
            pk.dispose();
            pv.dispose();
            const tEnd = performance.now();
            const hist = slid.length ? textOf(slid) : "";
            const timing = { melEncMs: tPre - tMel, encMs, decodeMs: tEnd - tPre, totalMs: tEnd - t0, tokens: gen.length, prefill: S, audioTokens: A, pace: m.pace ? { k: pacer.k, frameMs: +pacer.frameMs.toFixed(1), frames: pacer.frames } : null };
            if (aborted) return { ...base, ...timing, hist, aborted: true };
            const prevLen = st.tokens.length;
            st.tokens = [...forced, ...gen];
            const bytes = bytesOf(st.tokens);
            const cut = utf8Cut(bytes, bytesOf(st.tokens.slice(0, Math.max(0, st.tokens.length - ROLLBACK))).length);
            const dec = new TextDecoder();
            return { ...base, ...timing, slidN: slid.length, decoded: true, grew: st.tokens.length > prevLen, hist, conf: dec.decode(bytes.subarray(0, cut)), tent: dec.decode(bytes.subarray(cut)), ids: st.tokens };
        }

        let queue = Promise.resolve();
        self.onmessage = (ev) => {
            const m = ev.data;
            if (m.type === "tick") {
                for (const r of tickWaiters.splice(0)) r();
                return;
            }
            if (m.type === "gate") {
                gateOpen = m.open;
                if (gateOpen) for (const r of gateWaiters.splice(0)) r();
                return;
            }
            queue = queue.then(async () => {
                try {
                    if (m.type === "capture") { self.__cap = { top2: [] }; const r = await transcribe(new Float32Array(m.pcm), "ko", 96); const c = self.__cap; self.__cap = null; post({ type: "capture", ...c, text: r.text, timing: { encPrefillMs: r.encPrefillMs, decodeMs: r.decodeMs } }); }
                    else if (m.type === "load") await load(m); else if (m.type === "bench") post({ type: "bench", id: m.id, ...(await self.__bench(m, { transcribe, streamHop, model: () => model, resetStream })) });
                    else if (m.type === "stream") post({ type: "result", id: m.id, ...(await streamHop(m)) });
                    else if (m.type === "run") {
                        // VAD over the window's new audio: labels the hop, and
                        // gates it unless m.vad is false.
                        const v = m.newPcm ? await vadSpeech(m.newPcm) : null;
                        const speech = v ? v.speechSec >= VAD_MIN_SPEECH_SEC : true;
                        const vadInfo = { speechSec: v?.speechSec ?? null, vadMs: v?.ms ?? null, speech };
                        if (!speech && m.vad !== false) post({ type: "result", id: m.id, ...vadInfo, skipped: "vad", text: "", totalMs: 0 });
                        else post({ type: "result", id: m.id, ...vadInfo, ...(await transcribe(m.pcm, m.lang, m.maxTokens, m.gapMs, m.pace)) });
                    }
                } catch (error) {
                    post({ type: "error", id: m.id, ...errFields(error) });
                }
            });
        };
    })();
