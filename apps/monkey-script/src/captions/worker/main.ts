// The STT Web Worker: model loading, VAD, decoding. Built as its own ESM
// bundle and inlined into the userscript as a string (see tsdown.config.ts).
import { closeBlocks, utf8Cut, repeatStop, type BlockState } from "../merge.ts";
import { createEngine, type Engine } from "../engine/engine.ts";
import { ORT_VERSION } from "../constants.ts";
import { configureLogging, getLogger, toLogLevel } from "../../shared/logtape.ts";
import type { EngineAssets, EngineInfo, LoadMessage, PageMessage, ResultMessage, StreamMessage, WorkerMessage } from "../protocol.ts";

configureLogging("info");
const logger = getLogger(["captions", "worker"]);

// The slice of onnxruntime-web used here; the bundle is imported from
// the CDN at run time, so its package types are not available.
interface OrtTensor {
    readonly dims: readonly number[];
    getData(): Promise<Float32Array | Uint16Array>;
    dispose(): void;
}
type OrtOutputs = Record<string, OrtTensor>;
interface OrtSession {
    run(feeds: Record<string, OrtTensor>): Promise<OrtOutputs>;
}
interface OrtSessionOptions {
    executionProviders: string[];
    externalData?: { path: string; data: Uint8Array }[];
    preferredOutputLocation?: Record<string, string>;
}
interface Ort {
    env: { wasm: { wasmPaths: string; numThreads: number } };
    InferenceSession: { create(model: Uint8Array, options: OrtSessionOptions): Promise<OrtSession> };
    Tensor: new (type: "float32" | "float16" | "int64", data: Float32Array | Uint16Array | BigInt64Array, dims: readonly number[]) => OrtTensor;
}
// The part of ORT's Emscripten wasm Module that hookWebGpuEpConfig calls.
interface OrtWasmModule {
    lengthBytesUTF8(s: string): number;
    _malloc(n: number): number;
    _free(p: number): void;
    stringToUTF8(s: string, p: number, n: number): void;
    UTF8ToString(p: number): string;
    _OrtAddSessionConfigEntry(h: number, k: number, v: number): number;
}
type AppendEp = (h: number, name: number, ...rest: number[]) => number;
interface PromptConfig {
    prompt: { prefix_ids: number[]; suffix_ids: number[]; audio_pad_id: number; eos_ids: number[] };
    language_prefix_ids: Record<string, number[]>;
}
interface Model {
    enc: OrtSession;
    init: OrtSession;
    step: OrtSession;
    emb: Int8Array;
    sc: Float32Array;
    cfg: PromptConfig;
    D: number;
    filters: number[][];
    inv: string[];
    byteOf: Record<string, number>;
}
interface PipeRec {
    label?: string;
    layout: "auto" | (GPUBindGroupLayoutEntry[] | undefined)[];
    code: string;
    entryPoint?: string;
    constants?: Record<string, GPUPipelineConstantValue>;
}
interface Pipes {
    key: string;
    cache: Cache | null;
    saves: number;
    held: GPUComputePipeline[];
    device: GPUDevice | null;
    code: WeakMap<GPUShaderModule, string>;
    bgl: WeakMap<GPUBindGroupLayout, GPUBindGroupLayoutEntry[]>;
    layout: WeakMap<GPUPipelineLayout, (GPUBindGroupLayoutEntry[] | undefined)[]>;
    seen: Map<string, PipeRec>;
    install(): void;
    save(): Promise<void>;
    prewarm(cache: Cache, key: string, waitFrame: () => Promise<void>): Promise<number>;
}
// Marks left on patched WebGPU methods so a second load patches nothing.
interface PatchMarks {
    cbWidePatched?: boolean;
    cbKernPatched?: boolean;
    cbPatched?: boolean;
    cbPumped?: boolean;
}
type Marked<F> = F & PatchMarks;
declare global {
    interface GPUDevice {
        cbPipeRec?: boolean;
    }
    var __fft: ((re: Float64Array, im: Float64Array, n: number) => void) | undefined;
    var __wideHits: number | undefined;
    var __kernHits: Record<string, number> | undefined;
}
type OnBytes = (n: number, cached: boolean) => void;
interface VadState {
    state: Float32Array;
    ctx: Float32Array;
    carry: Float32Array;
}
type StreamState = BlockState & { quietSec: number };
type StreamResult = Omit<ResultMessage, "type" | "id" | "col">;

const ORT = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const VAD_URL = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/silero_vad_v5.onnx";
const VAD_THRESHOLD = 0.5;
const VAD_MIN_SPEECH_SEC = 0.15;
const QUIET_RESET_SEC = 1.5;
const HIST_CONTEXT = 24;
let ort: Ort;
let model: Model | null = null;
let f16 = false;
let engine: Engine | null = null;
// One worker serves every split-view column: each request carries
// the column id, the decode and VAD state live per column (cols),
// and replies to a request go back tagged with its column.
let curCol: string | null = null;
const post = (m: WorkerMessage, t?: Transferable[]) => {
    self.postMessage(curCol !== null && (m.type === "result" || m.type === "error") ? { ...m, col: curCol } : m, t || []);
    if (m.type === "result" && pipes.saves < 3) pipes.save();
};
// Chrome compiles a WebGPU pipeline in the GPU process, on the same
// thread that composites the page, and ORT creates its pipelines
// synchronously the first time each kernel runs: the first encoder
// pass compiles some 30 at once and the video freezes for ~1.5 s.
// Dawn keeps a content-keyed cache of live pipelines, so the
// descriptors the first runs use are recorded into Cache Storage and,
// on the next load, built ahead with createComputePipelineAsync (a
// GPU-process worker thread), one per frame; ORT's synchronous calls
// then hit that cache. The very first load still pays once.
const pipes: Pipes = {
    key: "", cache: null, saves: 0, held: [], device: null,
    code: new WeakMap(), bgl: new WeakMap(), layout: new WeakMap(), seen: new Map(),
    install() {
        if (!self.GPUDevice || GPUDevice.prototype.cbPipeRec) return;
        const P = GPUDevice.prototype, A = GPUAdapter.prototype;
        const rd = A.requestDevice;
        A.requestDevice = async function (...a) { const d = await rd.apply(this, a); pipes.device = d; return d; };
        const sm = P.createShaderModule;
        P.createShaderModule = function (desc) { const m = sm.call(this, desc); pipes.code.set(m, desc.code); return m; };
        const bgl = P.createBindGroupLayout;
        P.createBindGroupLayout = function (desc) { const o = bgl.call(this, desc); pipes.bgl.set(o, JSON.parse(JSON.stringify(desc.entries))); return o; };
        const pl = P.createPipelineLayout;
        P.createPipelineLayout = function (desc) { const o = pl.call(this, desc); pipes.layout.set(o, (desc.bindGroupLayouts as GPUBindGroupLayout[]).map((b) => pipes.bgl.get(b))); return o; };
        const cp = P.createComputePipeline;
        P.createComputePipeline = function (desc) {
            const o = cp.call(this, desc);
            const code = pipes.code.get(desc.compute?.module);
            const layout = desc.layout === "auto" ? "auto" : pipes.layout.get(desc.layout);
            if (code && layout && !(layout as readonly unknown[]).includes?.(undefined)) {
                const rec: PipeRec = { label: desc.label, layout, code, entryPoint: desc.compute.entryPoint, constants: desc.compute.constants };
                const k = JSON.stringify(rec);
                if (!pipes.seen.has(k)) pipes.seen.set(k, rec);
            }
            return o;
        };
        P.cbPipeRec = true;
    },
    async save() {
        this.saves++;
        if (!this.cache || !this.seen.size) return;
        try { await this.cache.put(this.key, new Response(JSON.stringify([...this.seen.values()]))); } catch (error) { logger.warn("pipeline list not saved {error}", { error }); }
    },
    async prewarm(cache, key, waitFrame) {
        this.cache = cache;
        this.key = key;
        const d = this.device;
        const hit = await cache.match(key);
        if (!d || !hit) return 0;
        let list: (Omit<PipeRec, "layout"> & { layout: "auto" | GPUBindGroupLayoutEntry[][] })[];
        try { list = await hit.json(); } catch { return 0; }
        for (const r of list) {
            await waitFrame();
            try {
                const module = d.createShaderModule({ code: r.code });
                const layout = r.layout === "auto" ? "auto" : d.createPipelineLayout({ bindGroupLayouts: r.layout.map((entries) => d.createBindGroupLayout({ entries })) });
                this.held.push(await d.createComputePipelineAsync({ label: r.label, layout, compute: { module, entryPoint: r.entryPoint, constants: r.constants } }));
                this.seen.set(JSON.stringify(r), r);
            } catch (error) {
                logger.warn("pipeline prewarm skipped {label} {message}", { label: r.label, message: (error as Error | undefined)?.message });
            }
        }
        return list.length;
    },
};
// Firefox's Error.stack holds only frames, never the message, and a
// thrown value need not be an Error, so every field travels apart.
let stage = "idle";
type ErrorLike = { name?: string; message?: string; stack?: string } | null | undefined;
const errFields = (error: unknown) => ({ stage, name: (error as ErrorLike)?.name ?? typeof error, message: (error as ErrorLike)?.message ?? String(error), stack: (error as ErrorLike)?.stack ?? "" });

// Whisper-compatible log-mel with a mixed-radix FFT for n_fft = 400.
const N = 400;
const HOP = 160;
const COS = new Float64Array(N);
const SIN = new Float64Array(N);
for (let i = 0; i < N; i++) {
    COS[i] = Math.cos((2 * Math.PI * i) / N);
    SIN[i] = Math.sin((2 * Math.PI * i) / N);
}
function fft(re: Float64Array, im: Float64Array, n: number) {
    if (n === 1) return;
    let p = 2;
    while (n % p) p++;
    const m = n / p;
    const subRe: Float64Array[] = [];
    const subIm: Float64Array[] = [];
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
function logMel(pcm: Float32Array, filters: number[][]) {
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
    return (v: number) => {
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
const h2f = (v: number) => {
    const e = (v >> 10) & 31;
    const f = v & 1023;
    const a = e === 0 ? (f / 1024) * 2 ** -14 : 2 ** (e - 15) * (1 + f / 1024);
    return v & 0x8000 ? -a : a;
};

// The page closes the gate while playback needs the machine (thin
// video buffer, a stall, a hidden tab); loading waits at the next
// chunk or step boundary until it reopens.
// Per column: a column whose playback suffers pauses only its own jobs.
const gates = new Map<string | null, boolean>();
let gateWaiters: (() => void)[] = [];
const isOpen = () => gates.get(curCol) !== false;
const gate = async () => {
    while (!isOpen()) await new Promise<void>((r) => gateWaiters.push(r));
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Frame pacing: GPU work goes out in slices, one slice per display
// frame, sized to a fraction of the frame so the compositor and the
// video decoder get the rest. The worker's own requestAnimationFrame
// is used where it exists; otherwise the page posts a tick per frame.
// Hidden tabs get no frames, which also parks the work.
const workerRaf = typeof self.requestAnimationFrame === "function";
let tickWaiters: ((t?: number) => void)[] = [];
const pacer = { k: 1, frameMs: 16.7, last: 0, sliceStart: 0, inSlice: 0, frames: 0 };
const SLICE_SHARE = 0.35;
const nextFrame = () =>
    new Promise<void>((resolve) => {
        const done = (t?: number) => {
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
async function paced(enabled: boolean, unitIsWhole = false) {
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
const doneKey = (url: string) => `https://chzzkbest.invalid/complete?${encodeURIComponent(url)}`;
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
// an empty submit every 4 ms while a map is pending drives the poll.
// ORT web parses WebGPU EP config when the EP is appended, but adds
// `extra` session config entries only afterwards, so
// ep.webgpuexecutionprovider.forceCpuNodeNames never reaches the EP.
// Catch the wasm Module's first assignment of the append entry point
// and add the entry for the session being created just before it.
let forceCpuNext = "";
function hookWebGpuEpConfig() {
    const key = "_OrtAppendExecutionProvider";
    if (Object.getOwnPropertyDescriptor(Object.prototype, key)) return;
    Object.defineProperty(Object.prototype, key, {
        configurable: true,
        set(this: OrtWasmModule, v: AppendEp) {
            const M = this;
            const ours = new WeakSet<AppendEp>();
            const str = (x: string) => {
                const n = M.lengthBytesUTF8(x) + 1;
                const p = M._malloc(n);
                M.stringToUTF8(x, p, n);
                return p;
            };
            // Emscripten's asyncify layer swaps this entry point in
            // and out, so every value stored here is wrapped once and
            // the getter hands back a stable function.
            const wrap = (fn: AppendEp): AppendEp => {
                if (ours.has(fn)) return fn;
                const w = (h: number, name: number, ...rest: number[]) => {
                    if (forceCpuNext && M.UTF8ToString(name) === "WebGPU") {
                        const k = str("ep.webgpuexecutionprovider.forceCpuNodeNames");
                        const val = str(forceCpuNext);
                                            if (M._OrtAddSessionConfigEntry(h, k, val) !== 0) logger.warn("forceCpuNodeNames rejected");
                        M._free(k);
                        M._free(val);
                    }
                    return fn(h, name, ...rest);
                };
                ours.add(w);
                return w;
            };
            let cur = wrap(v);
            Object.defineProperty(M, key, { configurable: true, enumerable: true, get: () => cur, set: (nv: AppendEp) => (cur = wrap(nv)) });
        },
    });
}

async function installMapPump() {
    if ((GPUBuffer.prototype.mapAsync as Marked<GPUBuffer["mapAsync"]>).cbPumped) return;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    const device = await adapter.requestDevice();
    const src = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
    const dst = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const times: number[] = [];
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
    const owner = new WeakMap<GPUBuffer, GPUDevice>();
    const createBuffer = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function (desc) {
        const b = createBuffer.call(this, desc);
        owner.set(b, this);
        return b;
    };
    const mapAsync = GPUBuffer.prototype.mapAsync;
    const pumped = function (this: GPUBuffer, ...args: Parameters<GPUBuffer["mapAsync"]>) {
        const p = mapAsync.apply(this, args);
        const device = owner.get(this);
        if (!device) return p;
        let pending = true;
        const settle = () => (pending = false);
        p.then(settle, settle);
        (async () => {
            while (pending) {
                await sleep(4);
                if (pending) device.queue.submit([]);
            }
        })();
        return p;
    };
    pumped.cbPumped = true;
    GPUBuffer.prototype.mapAsync = pumped;
    logger.info("mapAsync pump on (16 B readback took {ms} ms)", { ms: ms.toFixed(0) });
}

// Firefox (naga -> HLSL) runs ORT's MatMulNBits wide-tile kernel ~5x slower than Chrome; a fully
// unrolled main with scalar accumulators is bit-identical and closes the gap (shadertune probe).
function patchMatMulNBitsWide() {
    const P = GPUDevice.prototype;
    if ((P.createShaderModule as Marked<GPUDevice["createShaderModule"]>).cbWidePatched) return;
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
    const patched = function (this: GPUDevice, desc: GPUShaderModuleDescriptor) {
        const c = desc?.code;
        if (c && need.every((s) => c.includes(s))) { const i = c.indexOf("@compute"); if (i > 0) { desc = { ...desc, code: c.slice(0, i) + main }; self.__wideHits = (self.__wideHits || 0) + 1; } }
        return create.call(this, desc);
    };
    patched.cbWidePatched = true;
    P.createShaderModule = patched;
}
// Firefox (naga -> HLSL -> DXC) keeps ORT's small constant-trip loops rolled and spills their
// dynamically indexed local arrays; RMSNorm also runs one thread per row. These rewrites keep the
// per-element arithmetic order of ORT 1.23.0 (bit-identical in the shaderopt2 probe) and are skipped
// unless every exact marker of the expected template is present.
function patchFirefoxKernels() {
    const P = GPUDevice.prototype;
    if ((P.createShaderModule as Marked<GPUDevice["createShaderModule"]>).cbKernPatched) return;
    const RMS_NEED = ["const workgroup_size_x: u32 = 64;", "alias f32_val_t = vec4<f32>;", "  if (global_idx >= uniforms.norm_count) { return; }\nlet offset = global_idx * uniforms.norm_size_vectorized;\nvar mean_vector = f32_val_t(0);\nvar mean_square_vector = f32_val_t(0);\nfor (var h: u32 = 0u; h < uniforms.norm_size_vectorized; h++) {\n   let value = f32_val_t(x[h + offset]);\n   mean_vector += value;\n   mean_square_vector += value * value;\n}\nlet mean = (mean_vector.x + mean_vector.y + mean_vector.z + mean_vector.w) / f32(uniforms.norm_size);\nlet inv_std_dev = inverseSqrt((mean_square_vector.x + mean_square_vector.y + mean_square_vector.z + mean_square_vector.w) / f32(uniforms.norm_size) + uniforms.epsilon);\nfor (var j: u32 = 0; j < uniforms.norm_size_vectorized; j++) {\n   let f32input = f32_val_t(x[j + offset]);\n   let f32scale = f32_val_t(scale[j]);\n   y[j + offset] =  x_value_t((f32input) * inv_std_dev * f32scale);\n}\n\n}"];
    const RMS_MAIN = "var<workgroup> cb_inv : array<f32, 64>;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_index) local_idx : u32) {\n  let row0 = workgroup_id.x * 64u;\n  let rows = min(64u, uniforms.norm_count - row0);\n  let nsv = uniforms.norm_size_vectorized;\n  if (local_idx < rows) {\n    let offset = (row0 + local_idx) * nsv;\n    var ms = f32_val_t(0);\n    for (var h: u32 = 0u; h < nsv; h++) { let v = f32_val_t(x[h + offset]); ms += v * v; }\n    cb_inv[local_idx] = inverseSqrt((ms.x + ms.y + ms.z + ms.w) / f32(uniforms.norm_size) + uniforms.epsilon);\n  }\n  workgroupBarrier();\n  let total = rows * nsv;\n  for (var i = local_idx; i < total; i += 64u) {\n    let r = i / nsv; let j = i - r * nsv;\n    let o = (row0 + r) * nsv + j;\n    y[o] = x_value_t(f32_val_t(x[o]) * cb_inv[r] * f32_val_t(scale[j]));\n  }\n}\n";
    function cbRmsNorm(c: string) {
        if (!RMS_NEED.every((s) => c.includes(s)) || !/^\s*$/.test(c.slice(c.indexOf(RMS_NEED[2]) + RMS_NEED[2].length))) return null;
        const i = c.indexOf("@compute");
        return i > 0 ? c.slice(0, i) + RMS_MAIN : null;
    }
    function cbMatMulScalar(c: string, unrollK: boolean) {
        const accDecl = " var acc: array<array<a_element_t, colPerThread>, rowPerThread>;\n";
        const inner = "var BCached: array<a_element_t, colPerThread>;\n  for (var k = 0; k < tileInner; k = k + 1) {\n    for (var inner = 0; inner < i32(colPerThread); inner = inner + 1) {\n      BCached[inner] = mm_Bsub[k][tileCol + inner];\n    }\n    for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n      let ACached = mm_Asub[tileRow + innerRow][k];\n      for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];\n      }\n    }\n  }\n";
        const write = "for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n  for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n    mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);\n  }\n}\n";
        const R = +(/\nconst rowPerThread = (\d+);\n/.exec(c) || [])[1], C = +(/\nconst colPerThread = (\d+);\n/.exec(c) || [])[1];
        if (!(R >= 1 && R <= 8 && C >= 1 && C <= 8) || !c.includes("\nconst tileInner = 32;\n") || !c.includes("alias a_element_t = f16;") || !c.includes(accDecl) || !c.includes(inner) || !c.includes(write)) return null;
        const A: string[] = [];
        for (let r = 0; r < R; r++) for (let q = 0; q < C; q++) A.push(` var acc_${r}_${q} = a_element_t(0);`);
        const step = (k: number | string) => {
            const L: string[] = [];
            for (let q = 0; q < C; q++) L.push(`    let b_${q} = mm_Bsub[${k}][tileCol + ${q}];`);
            for (let r = 0; r < R; r++) { L.push(`    let a_${r} = mm_Asub[tileRow + ${r}][${k}];`); for (let q = 0; q < C; q++) L.push(`    acc_${r}_${q} = acc_${r}_${q} + a_${r} * b_${q};`); }
            return L.join("\n");
        };
        let K: string;
        if (unrollK) { const L: string[] = []; for (let k = 0; k < 32; k++) L.push("  {\n" + step(k) + "\n  }"); K = L.join("\n") + "\n"; }
        else K = "  for (var k = 0; k < tileInner; k = k + 1) {\n" + step("k") + "\n  }\n";
        const W: string[] = [];
        for (let r = 0; r < R; r++) for (let q = 0; q < C; q++) W.push(`mm_write(batch, globalRow + ${r}, globalCol + ${q}, acc_${r}_${q});`);
        return c.replace(accDecl, A.join("\n") + "\n").replace(inner, K).replace(write, W.join("\n") + "\n");
    }
    function cbMatMulVec4(c: string, unrollK: boolean) {
        const accDecl = "  var acc: array<vec4<a_element_t>, rowPerThread>;\n";
        const inner = "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n      let BCached0 = mm_Bsub[k * innerElementSize][tileCol];\n      let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];\n      let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];\n      let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];\n      for (var i = 0; i < rowPerThread; i = i + 1) {\n        let ACached = mm_Asub[tileRow + i][k];\n        acc[i] = BCached0 * ACached.x + acc[i];\n        acc[i] = BCached1 * ACached.y + acc[i];\n        acc[i] = BCached2 * ACached.z + acc[i];\n        acc[i] = BCached3 * ACached.w + acc[i];\n      }\n    }\n";
        const write = "  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);\n  }\n";
        const R = +(/\nconst rowPerThread = (\d+);\n/.exec(c) || [])[1];
        if (!(R >= 1 && R <= 8) || !c.includes("\nconst innerElementSize = 4;\n") || !c.includes("\nconst tileInner = 32;\n") || !c.includes("alias a_element_t = f16;") || !c.includes(accDecl) || !c.includes(inner) || !c.includes(write)) return null;
        const A: string[] = []; for (let r = 0; r < R; r++) A.push(`  var acc_${r} = vec4<a_element_t>(0);`);
        const step = (k: number | string) => {
            const kk = typeof k === "number" ? (j: number) => `${4 * k + j}` : (j: number) => (j ? `k * 4 + ${j}` : "k * 4");
            const L = [0, 1, 2, 3].map((j) => `      let b_${j} = mm_Bsub[${kk(j)}][tileCol];`);
            for (let r = 0; r < R; r++) { L.push(`      let a_${r} = mm_Asub[tileRow + ${r}][${k}];`); for (let j = 0; j < 4; j++) L.push(`      acc_${r} = b_${j} * a_${r}.${"xyzw"[j]} + acc_${r};`); }
            return L.join("\n");
        };
        let K: string;
        if (unrollK) { const L: string[] = []; for (let k = 0; k < 8; k++) L.push("    {\n" + step(k) + "\n    }"); K = L.join("\n") + "\n"; }
        else K = "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n" + step("k") + "\n    }\n";
        const W: string[] = []; for (let r = 0; r < R; r++) W.push(`  mm_write(batch, globalRow + ${r}, globalCol, acc_${r});`);
        return c.replace(accDecl, A.join("\n") + "\n").replace(inner, K).replace(write, W.join("\n") + "\n");
    }
    const create = P.createShaderModule;
    const patched = function (this: GPUDevice, desc: GPUShaderModuleDescriptor) {
        const c = desc?.code;
        if (c && c.includes("fn main(")) {
            let o: string | null = null, k = "";
            if (c.includes("norm_size_vectorized")) { o = cbRmsNorm(c); k = "rms"; }
            else if (c.includes("mm_Asub")) { o = cbMatMulScalar(c, false); k = "mms"; if (!o) { o = cbMatMulVec4(c, false); k = "mmv"; } }
            if (o) { desc = { ...desc, code: o }; const h = (self.__kernHits ||= {}); h[k] = (h[k] || 0) + 1; }
        }
        return create.call(this, desc);
    };
    patched.cbKernPatched = true;
    P.createShaderModule = patched;
}
async function patchF16Bitcast(adapter: GPUAdapter) {
    const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
    const probe = device.createShaderModule({
        code: "enable f16;\n@group(0) @binding(0) var<storage, read_write> o: array<f16>;\n@compute @workgroup_size(1) fn main() { o[0] = bitcast<vec2<f16>>(0x3c00u)[0]; }",
    });
    const broken = (await probe.getCompilationInfo()).messages.some((m) => m.type === "error");
    device.destroy();
    if (!broken || (GPUDevice.prototype.createShaderModule as Marked<GPUDevice["createShaderModule"]>).cbPatched) return;
    const create = GPUDevice.prototype.createShaderModule;
    const patched = function (this: GPUDevice, desc: GPUShaderModuleDescriptor) {
        if (desc?.code?.includes("bitcast<vec2<f16>>")) desc = { ...desc, code: desc.code.replace(/bitcast<vec2<f16>>\(([^()]*)\)/g, "vec2<f16>(unpack2x16float($1))") };
        return create.call(this, desc);
    };
    patched.cbPatched = true;
    GPUDevice.prototype.createShaderModule = patched;
    logger.info("WGSL bitcast<vec2<f16>> rewrite on (naga gap)");
}
// Downloads stream straight into Cache Storage, never into a JS
// buffer, so every file can be in flight at once; a session reads
// its files back only right before it is created. A hit counts as
// present when it has a completion record or a Hub size to check
// against; the byte count itself is checked on read-back.
async function ensureCached(cache: Cache, url: string, onBytes: OnBytes, expect = 0, force = false) {
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
    const count = new TransformStream<Uint8Array, Uint8Array>({
        async transform(chunk, ctl) {
            await gate();
            n += chunk.byteLength;
            onBytes(chunk.byteLength, false);
            ctl.enqueue(chunk);
        },
    });
    await cache.put(url, new Response(res.body!.pipeThrough(count)));
    await cache.put(doneKey(url), new Response(String(n)));
}
// Response.arrayBuffer() on a ~1 GB cache entry stalls Chrome's whole
// renderer, page main thread included, for ~150 ms as it lands. The
// body is pulled chunk by chunk into a buffer of the recorded size
// instead, yielding to the event loop every 64 MB.
async function readBody(res: Response, want: number): Promise<ArrayBuffer> {
    if (!(want > 64 << 20) || !res.body) return res.arrayBuffer();
    const out = new Uint8Array(want);
    const reader = res.body.getReader();
    let n = 0, mark = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (n + value.byteLength > want) { await reader.cancel(); return new ArrayBuffer(n + value.byteLength); }
        out.set(value, n);
        n += value.byteLength;
        if (n - mark >= 64 << 20) { mark = n; await sleep(0); }
    }
    return n === want ? out.buffer : out.buffer.slice(0, n);
}
async function readCached(cache: Cache, url: string, onBytes: OnBytes, expect = 0): Promise<ArrayBuffer> {
    for (let attempt = 0; ; attempt++) {
        const hit = await cache.match(url);
        const done = await cache.match(doneKey(url));
        const want = done ? Number(await done.text()) : expect;
        const buf = hit && (await readBody(hit, want));
        if (buf && want > 0 && buf.byteLength === want) {
            if (!done) await cache.put(doneKey(url), new Response(String(want)));
            return buf;
        }
        if (attempt) throw new Error(`${url}: cached ${buf?.byteLength ?? 0} B, expected ${want} B`);
        logger.warn("cache entry incomplete, refetching {url} {bytes}", { url, bytes: buf?.byteLength ?? 0 });
        await cache.delete(url);
        await ensureCached(cache, url, onBytes, expect, true);
    }
}
async function fetchCached(cache: Cache, url: string, onBytes: OnBytes, expect = 0) {
    await ensureCached(cache, url, onBytes, expect);
    await gate();
    return readCached(cache, url, onBytes, expect);
}

async function load({ repo, rev, cacheName, engineAssets }: LoadMessage) {
    const t0 = performance.now();
    post({ type: "caps", workerRaf });
    hookWebGpuEpConfig();
    pipes.install();
    ort = await import(ORT + "ort.webgpu.bundle.min.mjs");
    ort.env.wasm.wasmPaths = ORT;
    ort.env.wasm.numThreads = 1;
    if (!navigator.gpu) throw new Error("WebGPU 없음 (navigator.gpu undefined)");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU 어댑터 없음");
    // A lost device ends the session: report it once and never
    // recreate the device or retry, since a driver reset loop
    // (TDR) is what a retry would feed.
    const reqDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...a) {
        const d = await reqDevice.apply(this, a);
        d.lost.then((info) => {
            if (info.reason !== "destroyed") post({ type: "error", stage: "device-lost", name: "GPUDeviceLost", message: info.message || "GPU device lost" });
        });
        return d;
    };
    f16 = adapter.features.has("shader-f16");
    patchMatMulNBitsWide();
    patchFirefoxKernels();
    if (f16) await patchF16Bitcast(adapter);
    await installMapPump();
    const base = `https://huggingface.co/${repo}/resolve/${rev}/`;
    const V = f16
        ? { enc: "encoder.fp16.onnx", init: "decoder_init.q4f16.onnx", step: "decoder_step.q4f16.onnx", w: "decoder_weights.q4f16.data" }
        : { enc: "encoder.onnx", init: "decoder_init.int4.onnx", step: "decoder_step.int4.onnx", w: "decoder_weights.int4.data" };
    const cache = await caches.open(cacheName);
    // File sizes for the progress bar; cached with the weights so a
    // warm start makes no network request at all.
    let meta: { siblings?: { rfilename: string; size: number }[] } | null = null;
    try {
        const buf = await fetchCached(cache, `https://huggingface.co/api/models/${repo}/revision/${rev}?blobs=true`, () => {});
        meta = JSON.parse(new TextDecoder().decode(buf));
    } catch (error) {
        logger.warn("size metadata unavailable {error}", { error });
    }
    const sizes: Record<string, number> = {};
    for (const s of meta?.siblings ?? []) sizes[s.rfilename] = s.size;
    const files = ["prompt_config.json", "config.json", "mel_filters.json", "vocab.json", "added_tokens.json", "embed_tokens.int8.bin", "embed_scales.f32.bin", V.enc, V.init, V.step, V.w];
    const total = files.reduce((a, f) => a + (sizes[f] || 0), 0);
    let loaded = 0;
    let fromNet = 0;
    let lastPost = 0;
    const bufs: Record<string, ArrayBuffer> = {};
    let createMs = 0;
    const onBytes: OnBytes = (n, cached) => {
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
    const get = async (f: string) => {
        await cached[f];
        await gate();
        bufs[f] = await readCached(cache, base + f, onBytes, sizes[f]);
        return bufs[f];
    };
    const create = async (bytes: ArrayBuffer, options: OrtSessionOptions) => {
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
    const json = (f: string): unknown => JSON.parse(new TextDecoder().decode(bufs[f]));
    const cfg = json("prompt_config.json") as PromptConfig;
    const D = (json("config.json") as { decoder: { hidden_size: number } }).decoder.hidden_size;
    const filters = (json("mel_filters.json") as { data: number[][] }).data;
    const vocab = json("vocab.json") as Record<string, number>;
    const opt: OrtSessionOptions = { executionProviders: ["webgpu"] };
    stage = "load:encoder";
    const enc = await create(await get(V.enc), opt);
    delete bufs[V.enc];
    const w = new Uint8Array(await get(V.w));
    const kvOpt: OrtSessionOptions = {
        ...opt,
        externalData: [{ path: V.w, data: w }],
        preferredOutputLocation: { present_keys: "gpu-buffer", present_values: "gpu-buffer" },
    };
    // ORT puts the rotary-position subgraph (position_ids → cos/sin
    // indices) on the CPU EP but its MatMul/Expand/Cast upstream on
    // WebGPU, so every run read three tiny tensors back to the CPU —
    // three mapAsync waits per token on top of the logits. Pinning the
    // whole subgraph (it depends only on position_ids) to the CPU
    // leaves only uploads. Names are the same in the 0.6B and 1.7B
    // exports; a name that is absent is ignored.
    const cpuNodes = (names: string[]) => ((forceCpuNext = names.join("\n")), kvOpt);
    const rope = ["node_unsqueeze", "node_expand", "node_unsqueeze_4", "node__to_copy", "node_Concat_21", "node_expand_1", "node_matmul", "node_transpose"];
    stage = "load:decoder_init";
    const init = await create(await get(V.init), cpuNodes([...rope, "node_Concat_132", "node_full", "node_Gather_72", "node_Range_73", "node_Slice_77", "node_Unsqueeze_79", "node_Gather_112", "node_Range_113", "node_Slice_117", "node_Unsqueeze_118"]));
    delete bufs[V.init];
    stage = "load:decoder_step";
    const step = await create(await get(V.step), cpuNodes([...rope, "node_select_1", "node_slice_2", "node_Transpose_71", "node_select_2", "node_Transpose_72", "node_ScatterND_73", "node_slice_scatter", "node_Unsqueeze_74", "node_select_scatter", "node_select_8", "node_slice_8", "node_Transpose_110", "node_select_9", "node_Transpose_111", "node_ScatterND_112", "node_slice_scatter_1", "node_Unsqueeze_113"]));
    forceCpuNext = "";
    delete bufs[V.step];
    delete bufs[V.w];
    stage = "load:pipelines";
    const tWarm = performance.now();
    const warmed = await pipes.prewarm(cache, `https://chzzkbest.invalid/pipelines?ort=${ORT}&enc=${V.enc}&step=${V.step}&kern=2`, async () => { await gate(); await (workerRaf ? nextFrame() : sleep(16)); });
    createMs += performance.now() - tWarm;
    post({ type: "progress", loaded, total, fromNet });
    const tDl = performance.now() - createMs;
    const embB = bufs["embed_tokens.int8.bin"];
    const emb = new Int8Array(embB);
    const sc = new Float32Array(bufs["embed_scales.f32.bin"]);
    // GPT-2 byte-level BPE alphabet → bytes.
    const byteOf: Record<string, number> = {};
    {
        const bs: number[] = [];
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
    const inv: string[] = [];
    for (const [k, v] of Object.entries(vocab)) inv[v] = k;
    model = { enc, init, step, emb, sc, cfg, D, filters, inv, byteOf };
    const engineInfo = await loadEngine(engineAssets, cache, base, `${repo}@${rev}`);
    const tLoad = performance.now();
    post({ type: "ready", f16, engine: engineInfo, downloadMs: tDl - t0, sessionMs: tLoad - tDl, fromNet, total, warmed, warmMs: performance.now() - tWarm });
}

// The engine decodes on ORT's own device. It needs its manifest and
// qknorm.bin from the page (the release assets) and reads the decoder
// weights and embeddings out of the Cache Storage entries the ORT load
// just filled, which are the same bytes. It is built for exactly one
// export (the manifest names it), so any other model, a device without
// shader-f16, or any load error leaves engine null and the ORT loop in use.
async function loadEngine(assets: EngineAssets | null, cache: Cache, base: string, source: string): Promise<EngineInfo> {
    engine = null;
    if (!assets) return { on: false, reason: "no assets" };
    if (!f16) return { on: false, reason: "no shader-f16" };
    if (!pipes.device) return { on: false, reason: "no ORT device" };
    const man: { config?: { source?: string } } = JSON.parse(new TextDecoder().decode(assets.manifest));
    if (man.config?.source !== source) return { on: false, reason: `manifest is for ${man.config?.source}, model is ${source}` };
    const t = performance.now();
    try {
        stage = "load:engine";
        await gate();
        const local: Record<string, ArrayBuffer> = { "engine:manifest.json": assets.manifest, "engine:qknorm.bin": assets.qknorm };
        engine = await createEngine(pipes.device, {
            manifestUrl: "engine:manifest.json",
            urls: {
                "decoder_weights.q4f16.data": base + "decoder_weights.q4f16.data",
                "qknorm.bin": "engine:qknorm.bin",
                embed: base + "embed_tokens.int8.bin",
                embedScales: base + "embed_scales.f32.bin",
            },
            fetch: async (url: string) => {
                if (local[url]) return new Response(local[url]);
                const hit = await cache.match(url);
                if (!hit) throw new Error(`engine: ${url} is not in Cache Storage`);
                return hit;
            },
        });
        return { on: true, loadMs: Math.round(performance.now() - t) };
    } catch (error) {
        engine = null;
        logger.warn("engine unavailable, using the ORT decode loop {error}", { error });
        return { on: false, reason: String((error as Error | undefined)?.message || error).slice(0, 200) };
    }
}

const toT = (arr: Float32Array, dims: number[]) =>
    f16 ? new ort.Tensor("float16", Uint16Array.from(arr, f2h), dims) : new ort.Tensor("float32", arr, dims);
async function argmax(t: OrtTensor) {
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
function embedRow(id: number, out: Float32Array, off: number) {
    const { emb, sc, D } = model!;
    const s = sc[id];
    const b = id * D;
    for (let d = 0; d < D; d++) out[off + d] = emb[b + d] * s;
}

// Greedy decode after decoder_init `o` (prefill length S). Returns the
// generated ids, eos excluded, and stops early once stop(gen) holds.
// With the engine loaded and the KV cache fitting its fixed LMAX, the
// whole decode runs in the engine with ORT's prefill KV; the engine does
// not stop mid-decode, so its output is cut where the ORT loop would
// have stopped. Otherwise each step is an ORT decoder_step run.
interface DecodeGreedyOptions {
    stop: (gen: readonly number[]) => boolean;
    pace?: boolean;
    gapMs?: number;
    useEngine?: boolean;
}
async function decodeGreedy(o: OrtOutputs, S: number, maxTokens: number, { stop, pace = false, gapMs = 0, useEngine = true }: DecodeGreedyOptions): Promise<{ gen: number[]; aborted: boolean; via: "engine" | "ort"; engineMs?: number }> {
    const { step, cfg, D } = model!;
    const P = cfg.prompt;
    const gen: number[] = [];
    let tok = await argmax(o.logits);
    const cap = maxTokens - 1;
    if (engine && useEngine && cap > 0 && S + cap <= engine.LMAX && !P.eos_ids.includes(tok)) {
        stage = "engine";
        const pk = await o.present_keys.getData();
        const pv = await o.present_values.getData();
        o.present_keys.dispose();
        o.present_values.dispose();
        if (pk.BYTES_PER_ELEMENT !== 2 || pk.length !== engine.config.layers * 8 * S * 128) throw new Error(`engine prefill: unexpected KV ${pk.constructor.name}[${pk.length}] for S=${S}`);
        engine.setPrefill(pk, pv, S);
        const r = await engine.decode(tok, S, { cap, owd: true });
        for (const t of [tok, ...r.tokens]) {
            if (P.eos_ids.includes(t) || gen.length >= maxTokens) break;
            gen.push(t);
            if (stop(gen)) break;
        }
        return { gen, aborted: false, via: "engine", engineMs: r.timing.totalMs };
    }
    let pos = S;
    let pk = o.present_keys;
    let pv = o.present_values;
    const e = new Float32Array(D);
    let aborted = false;
    while (!P.eos_ids.includes(tok) && gen.length < maxTokens) {
        // A closed gate abandons the window; the next one covers the
        // same audio, and playback gets the GPU back within a step.
        if (!isOpen()) {
            aborted = true;
            break;
        }
        gen.push(tok);
        if (stop(gen)) break;
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
        tok = await argmax(n.logits);
        // Under load the page asks for breathing room between steps
        // so frames get GPU time mid-utterance, not only between runs.
        if (gapMs && gen.length % 4 === 0) await sleep(gapMs);
    }
    pk.dispose();
    pv.dispose();
    return { gen, aborted, via: "ort" };
}
const stopWindow = repeatStop(12);
const stopStream = repeatStop(8);

async function transcribe(pcm: Float32Array, lang: string, maxTokens: number, gapMs = 0, pace = false, useEngine = true) {
    pacer.inSlice = 0;
    const { enc, init, step, cfg, D, filters, inv, byteOf } = model!;
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
    const ids = [...P.prefix_ids, ...Array<number>(A).fill(P.audio_pad_id), ...P.suffix_ids, ...(cfg.language_prefix_ids[lang] || [])];
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
    const { gen, aborted, via, engineMs } = await decodeGreedy(o, S, maxTokens, { stop: stopWindow, pace, gapMs, useEngine });
    const tEnd = performance.now();
    const bytes: number[] = [];
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
        via,
        engineMs,
        melMs: tMel - t0,
        encPrefillMs: tPre - tMel,
        decodeMs: tEnd - tPre,
        totalMs: tEnd - t0,
        pace: pace ? { k: pacer.k, frameMs: +pacer.frameMs.toFixed(1), frames: pacer.frames } : null,
    };
}

// ---- voice activity: Silero VAD v5 on the CPU (wasm), never WebGPU,
// so the gate costs the video nothing. State carries across hops.
let vad: (VadState & { session: OrtSession }) | null = null;
async function loadVad(cache: Cache) {
    try {
        const buf = await fetchCached(cache, VAD_URL, () => {});
        const session = await ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ["wasm"] });
        vad = { session, state: new Float32Array(256), ctx: new Float32Array(64), carry: new Float32Array(0) };
    } catch (error) {
        logger.warn("VAD unavailable; RMS gate only {fields} {error}", { fields: errFields(error), error });
    }
}
async function vadSpeech(pcm: Float32Array) {
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
const newStream = (): StreamState => ({ open: new Float32Array(0), closed: [], tokens: [], histIds: [], quietSec: 0 });
const newVadState = (): VadState => ({ state: new Float32Array(256), ctx: new Float32Array(64), carry: new Float32Array(0) });
const cols = new Map<string | null, { st: StreamState; vs: VadState }>(); // column id -> { st, vs }
let st = newStream();
const resetStream = () => {
    st.open = new Float32Array(0);
    st.closed = [];
    st.tokens = [];
    st.histIds = [];
    st.quietSec = 0;
};
const bytesOf = (ids: readonly number[]) => {
    const { inv, byteOf } = model!;
    const bytes: number[] = [];
    for (const t of ids) {
        const piece = inv[t];
        if (piece === undefined || t >= 151643) continue;
        for (const c of piece) bytes.push(byteOf[c] ?? 63);
    }
    return Uint8Array.from(bytes);
};
const textOf = (ids: readonly number[]) => new TextDecoder().decode(bytesOf(ids));
async function encodeBlock(pcm: Float32Array) {
    const mel = logMel(pcm, model!.filters);
    stage = "encoder";
    const eo = await model!.enc.run({ mel: new ort.Tensor("float32", mel.data, [1, 128, mel.T]) });
    const af = new Float32Array(await eo.audio_features.getData());
    eo.audio_features.dispose?.();
    return af;
}
async function streamHop(m: StreamMessage): Promise<StreamResult> {
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
            // The finished utterance stays as context for the next one;
            // only an explicit reset (m.reset) forgets it.
            const histIds = [...st.histIds, ...st.tokens].slice(-24);
            resetStream();
            st.histIds = histIds;
            return { ...base, hist, conf: "", tent: "", ids: [], final: true, totalMs: performance.now() - t0 };
        }
        // Hold the tentative text; the next speech hop re-decodes it.
        closeBlocks(st, BLOCK, 1, ROLLBACK, 24);
        return { ...base, totalMs: performance.now() - t0 };
    }
    st.quietSec = 0;
    const slid = closeBlocks(st, BLOCK, 1, ROLLBACK, 24);
    const { init, step, cfg, D } = model!;
    await gate();
    const tMel = performance.now();
    let encMs = 0;
    const parts: Float32Array[] = [];
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
    const ids = [...pre, ...Array<number>(A).fill(P.audio_pad_id), ...P.suffix_ids, ...(cfg.language_prefix_ids[m.lang] || []), ...forced];
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
    const { gen, aborted, via, engineMs } = await decodeGreedy(o, S, m.maxTokens, { stop: stopStream, pace: m.pace, gapMs: m.gapMs, useEngine: m.engine !== false });
    const tEnd = performance.now();
    const hist = slid.length ? textOf(slid) : "";
    const timing = { via, engineMs, melEncMs: tPre - tMel, encMs, decodeMs: tEnd - tPre, totalMs: tEnd - t0, tokens: gen.length, prefill: S, audioTokens: A, pace: m.pace ? { k: pacer.k, frameMs: +pacer.frameMs.toFixed(1), frames: pacer.frames } : null };
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
    const m: PageMessage = ev.data;
    if (m.type === "tick") {
        for (const r of tickWaiters.splice(0)) r();
        return;
    }
    if (m.type === "gate") {
        gates.set(m.col ?? null, m.open);
        for (const r of gateWaiters.splice(0)) r();
        return;
    }
    if (m.type === "drop") {
        cols.delete(m.col);
        gates.delete(m.col);
        for (const r of gateWaiters.splice(0)) r();
        return;
    }
    queue = queue.then(async () => {
        curCol = m.col ?? null;
        let cx: { st: StreamState; vs: VadState } | null | undefined = null;
        if (m.type !== "load") {
            cx = cols.get(curCol);
            if (!cx) cols.set(curCol, (cx = { st: newStream(), vs: newVadState() }));
            st = cx.st;
            if (vad) Object.assign(vad, cx.vs);
        }
        try {
            if (m.type === "load") {
                configureLogging(toLogLevel(m.logLevel));
                await load(m);
            } else if (m.type === "stream") post({ type: "result", id: m.id, ...(await streamHop(m)) });
            else if (m.type === "run") {
                // VAD over the window's new audio: labels the hop, and
                // gates it unless m.vad is false.
                const v = m.newPcm ? await vadSpeech(m.newPcm) : null;
                const speech = v ? v.speechSec >= VAD_MIN_SPEECH_SEC : true;
                const vadInfo = { speechSec: v?.speechSec ?? null, vadMs: v?.ms ?? null, speech };
                if (!speech && m.vad !== false) post({ type: "result", id: m.id, ...vadInfo, skipped: "vad", text: "", totalMs: 0 });
                else post({ type: "result", id: m.id, ...vadInfo, ...(await transcribe(m.pcm, m.lang, m.maxTokens, m.gapMs, m.pace, m.engine !== false)) });
            }
        } catch (error) {
            post({ type: "error", id: m.id, ...errFields(error) });
        } finally {
            if (cx && vad) cx.vs = { state: vad.state, ctx: vad.ctx, carry: vad.carry };
            curCol = null;
        }
    });
};
