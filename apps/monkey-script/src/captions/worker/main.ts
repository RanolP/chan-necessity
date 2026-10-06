// The STT Web Worker: model loading, VAD, decoding. Built as its own ESM
// bundle and inlined into the userscript as a string (see tsdown.config.ts).
// Everything runs on one WebGPU device the worker owns: the encoder's
// output stays a GPU buffer that the decoder prefill reads directly, and
// the only readbacks are the decoded token ids.
import { closeBlocks, utf8Cut, loopStop, trimLoop, LOOP_SPAN, type BlockState } from "../merge.ts";
import { createEngine, makePump, type Engine } from "../engine/engine.ts";
import { createEncoder, type Encoder, type EncoderResult } from "../engine/encoder.ts";
import { logMel } from "../mel.ts";
import { createVad, parseSileroVadWeights, SILERO_VAD_V5_URL, type SileroVad, type SileroVadWeights } from "../vad.ts";
import { configureLogging, getLogger, toLogLevel } from "../../shared/logtape.ts";
import type { LoadMessage, PageMessage, ResultMessage, StreamMessage, WorkerMessage } from "../protocol.ts";

configureLogging("info");
const logger = getLogger(["captions", "worker"]);

interface PromptConfig {
    prompt: { prefix_ids: number[]; suffix_ids: number[]; audio_pad_id: number; eos_ids: number[] };
    language_prefix_ids: Record<string, number[]>;
}
interface Model {
    encoder: Encoder;
    cfg: PromptConfig;
    D: number;
    filters: number[][];
    inv: string[];
    byteOf: Record<string, number>;
}
type OnBytes = (n: number, cached: boolean) => void;
// Silero keeps its recurrent state inside `net`; ctx and carry are the
// sample context the next 512-sample frame needs.
interface VadState {
    net: SileroVad;
    ctx: Float32Array;
    carry: Float32Array;
}
type StreamState = BlockState<EncoderResult> & { quietSec: number };
type StreamResult = Omit<ResultMessage, "type" | "id" | "col">;

// The decoder engine and the encoder are built for exactly these files.
const ENCODER_FILE = "encoder.fp16.onnx";
const WEIGHTS_FILE = "decoder_weights.q4f16.data";
const VAD_THRESHOLD = 0.5;
const VAD_MIN_SPEECH_SEC = 0.15;
const QUIET_RESET_SEC = 1.5;
const HIST_CONTEXT = 24;
let device: GPUDevice | null = null;
let model: Model | null = null;
let engine: Engine | null = null;
let gpuError: string | null = null;
function failOnGpuError() {
    if (gpuError === null) return;
    const message = gpuError;
    gpuError = null;
    throw new Error(`WebGPU validation: ${message}`);
}
// One worker serves every split-view column: each request carries
// the column id, the decode and VAD state live per column (cols),
// and replies to a request go back tagged with its column.
let curCol: string | null = null;
const post = (m: WorkerMessage, t?: Transferable[]) => {
    self.postMessage(curCol !== null && (m.type === "result" || m.type === "error") ? { ...m, col: curCol } : m, t || []);
};
let stage = "idle";
type ErrorLike = { name?: string; message?: string; stack?: string } | null | undefined;
const errFields = (error: unknown) => ({ stage, name: (error as ErrorLike)?.name ?? typeof error, message: (error as ErrorLike)?.message ?? String(error), stack: (error as ErrorLike)?.stack ?? "" });

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
// Hidden tabs get no frames: the page stops pacing jobs it sends while
// hidden, and a job already waiting goes on after FRAME_WAIT_MAX_MS.
const workerRaf = typeof self.requestAnimationFrame === "function";
const FRAME_WAIT_MAX_MS = 100;
let tickWaiters: ((t?: number) => void)[] = [];
const pacer = { k: 1, frameMs: 16.7, last: 0, sliceStart: 0, inSlice: 0, frames: 0 };
const SLICE_SHARE = 0.35;
const nextFrame = () =>
    new Promise<void>((resolve) => {
        let settled = false;
        const timer = setTimeout(() => {
            settled = true;
            resolve();
        }, FRAME_WAIT_MAX_MS);
        const done = (t?: number) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
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
// Downloads stream straight into Cache Storage, never into a JS
// buffer, so every file can be in flight at once; a file is read
// back only right before it is needed. A hit counts as
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
    // The decoder engine is built for one export, named by its manifest;
    // without matching assets there is nothing to decode with.
    if (!engineAssets) throw new Error(`${repo}: 디코더 엔진 자산 없음 (지원 모델은 1.7B뿐)`);
    const source = `${repo}@${rev}`;
    const man: { config?: { source?: string } } = JSON.parse(new TextDecoder().decode(engineAssets.manifest));
    if (man.config?.source !== source) throw new Error(`engine manifest is for ${man.config?.source}, model is ${source}`);
    if (!navigator.gpu) throw new Error("WebGPU 없음 (navigator.gpu undefined)");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU 어댑터 없음");
    // The kernels read f16 weights as packed u32, so shader-f16 is not
    // needed; the decoder weights do need the adapter's largest buffers.
    const gpu = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
        },
    });
    device = gpu;
    // A validation error never throws at the call site; it invalidates the
    // command buffer, and the job would decode from zeroed buffers. The
    // first one fails the job in flight (see failOnGpuError).
    gpu.addEventListener("uncapturederror", (event) => {
        const message = (event as GPUUncapturedErrorEvent).error.message;
        logger.error("uncaptured WebGPU error {message}", { message });
        gpuError ??= message;
    });
    // A lost device ends the session: report it once and never
    // recreate the device or retry, since a driver reset loop
    // (TDR) is what a retry would feed.
    gpu.lost.then((info) => {
        if (info.reason !== "destroyed") post({ type: "error", stage: "device-lost", name: "GPUDeviceLost", message: info.message || "GPU device lost" });
    });
    const base = `https://huggingface.co/${repo}/resolve/${rev}/`;
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
    const files = ["prompt_config.json", "config.json", "mel_filters.json", "vocab.json", "added_tokens.json", "embed_tokens.int8.bin", "embed_scales.f32.bin", ENCODER_FILE, WEIGHTS_FILE];
    const total = files.reduce((a, f) => a + (sizes[f] || 0), 0);
    let loaded = 0;
    let fromNet = 0;
    let lastPost = 0;
    const onBytes: OnBytes = (n, cached) => {
        loaded += n;
        if (!cached) fromNet += n;
        const now = performance.now();
        if (now - lastPost > 250) {
            lastPost = now;
            post({ type: "progress", loaded, total, fromNet });
        }
    };
    // Every download starts now; the GPU side is built once all are cached.
    const cached = files.map((f) => ensureCached(cache, base + f, onBytes, sizes[f]));
    for (const p of cached) p.catch(() => {});
    const get = async (f: string) => {
        await gate();
        return readCached(cache, base + f, onBytes, sizes[f]);
    };
    stage = "load:VAD";
    await loadVad(cache);
    stage = "load:files";
    await Promise.all(cached);
    post({ type: "progress", loaded, total, fromNet });
    const tDl = performance.now();
    const json = async (f: string): Promise<unknown> => JSON.parse(new TextDecoder().decode(await get(f)));
    const cfg = (await json("prompt_config.json")) as PromptConfig;
    const D = ((await json("config.json")) as { decoder: { hidden_size: number } }).decoder.hidden_size;
    const filters = ((await json("mel_filters.json")) as { data: number[][] }).data;
    const vocab = (await json("vocab.json")) as Record<string, number>;
    const pump = makePump(gpu);
    stage = "load:encoder";
    await gate();
    const encoder = await createEncoder(gpu, await get(ENCODER_FILE), { wait: pump });
    if (encoder.config.outputDim !== D) throw new Error(`encoder output ${encoder.config.outputDim} != decoder hidden ${D}`);
    stage = "load:engine";
    await gate();
    const local: Record<string, ArrayBuffer> = { "engine:manifest.json": engineAssets.manifest, "engine:qknorm.bin": engineAssets.qknorm };
    // The engine reads the decoder weights and embeddings straight out
    // of the Cache Storage entries filled above.
    engine = await createEngine(gpu, {
        manifestUrl: "engine:manifest.json",
        urls: {
            [WEIGHTS_FILE]: base + WEIGHTS_FILE,
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
    model = { encoder, cfg, D, filters, inv, byteOf };
    const info = adapter.info;
    post({ type: "ready", adapter: `${info.vendor} ${info.architecture}`.trim(), downloadMs: tDl - t0, buildMs: performance.now() - tDl, fromNet, total });
}

// Greedy decode of a prompt whose audio span [audioStart, audioStart +
// audio.frames) is filled from the encoder's GPU buffer. Returns the
// generated ids, eos excluded. The engine decodes a whole run in one
// go, so its output is cut where `stop` first holds.
async function decodeGreedy(ids: number[], audio: GPUBuffer, audioStart: number, audioTokens: number, maxTokens: number, stop: (gen: readonly number[]) => boolean, pace = false) {
    const eng = engine!;
    const P = model!.cfg.prompt;
    if (ids.length > eng.LMAX) throw new Error(`prompt of ${ids.length} tokens exceeds the engine's ${eng.LMAX}-token cache`);
    const t0 = performance.now();
    stage = "prefill";
    // A whole prefill is ~600 ms of GPU work: as one submit it drops video
    // frames, and the governor answers dropped frames with a 10 s pause.
    // Paced, it runs layer by layer and the compositor gets the gaps.
    const betweenLayers = pace
        ? async () => {
              await device!.queue.onSubmittedWorkDone();
              await paced(true);
          }
        : undefined;
    const pre = await eng.prefill(ids, audio, { audioStart, audioTokens, audioPadId: P.audio_pad_id, betweenLayers });
    const tPre = performance.now();
    const out = [pre.token];
    const cap = Math.min(maxTokens - 1, eng.LMAX - ids.length);
    if (!pre.done && cap > 0) {
        stage = "decode";
        await paced(pace, true);
        out.push(...(await eng.decode(pre.token, pre.position, { cap, owd: true, perSubmit: pace ? 4 : 16 })).tokens);
    }
    const gen: number[] = [];
    for (const t of out) {
        if (P.eos_ids.includes(t) || gen.length >= maxTokens) break;
        gen.push(t);
        if (stop(gen)) break;
    }
    return { gen, prefillMs: tPre - t0, decodeMs: performance.now() - tPre };
}

async function encode(pcm: Float32Array) {
    const mel = logMel(pcm, model!.filters);
    stage = "encoder";
    return model!.encoder.encode(mel.data, mel.T);
}

async function transcribe(pcm: Float32Array, lang: string, maxTokens: number, pace = false) {
    pacer.inSlice = 0;
    const { cfg } = model!;
    const t0 = performance.now();
    await gate();
    await paced(pace, true);
    const enc = await encode(pcm);
    const tEnc = performance.now();
    const P = cfg.prompt;
    const ids = [...P.prefix_ids, ...Array<number>(enc.frames).fill(P.audio_pad_id), ...P.suffix_ids, ...(cfg.language_prefix_ids[lang] || [])];
    await paced(pace, true);
    let r;
    try {
        r = await decodeGreedy(ids, enc.buffer, P.prefix_ids.length, enc.frames, maxTokens, loopStop, pace);
    } finally {
        enc.buffer.destroy();
    }
    const tEnd = performance.now();
    let text = textOf(r.gen);
    const cut = text.lastIndexOf("<asr_text>");
    if (cut >= 0) text = text.slice(cut + 10);
    return {
        text: text.trim(),
        tokens: r.gen.length,
        encMs: tEnc - t0,
        encPrefillMs: tEnc - t0 + r.prefillMs,
        prefillMs: r.prefillMs,
        decodeMs: r.decodeMs,
        totalMs: tEnd - t0,
        pace: pace ? { k: pacer.k, frameMs: +pacer.frameMs.toFixed(1), frames: pacer.frames } : null,
    };
}

// ---- voice activity: Silero VAD v5 on the CPU (vad.ts), never
// WebGPU, so the gate costs the video nothing. Each column has its own
// recurrent state, carried across hops.
let vadWeights: SileroVadWeights | null = null;
let vs: VadState | null = null;
async function loadVad(cache: Cache) {
    try {
        vadWeights = parseSileroVadWeights(await fetchCached(cache, SILERO_VAD_V5_URL, () => {}));
    } catch (error) {
        logger.warn("VAD unavailable; RMS gate only {fields} {error}", { fields: errFields(error), error });
    }
}
const newVadState = (): VadState | null => (vadWeights ? { net: createVad(vadWeights), ctx: new Float32Array(64), carry: new Float32Array(0) } : null);
async function vadSpeech(pcm: Float32Array) {
    if (!vs) return null;
    const t0 = performance.now();
    stage = "VAD";
    const x = new Float32Array(vs.carry.length + pcm.length);
    x.set(vs.carry);
    x.set(pcm, vs.carry.length);
    const inp = new Float32Array(576);
    let speech = 0;
    let off = 0;
    for (; off + 512 <= x.length; off += 512) {
        inp.set(vs.ctx, 0);
        inp.set(x.subarray(off, off + 512), 64);
        if (vs.net.prob(inp) >= VAD_THRESHOLD) speech++;
        vs.ctx = x.slice(off + 448, off + 512);
    }
    vs.carry = x.slice(off);
    return { speechSec: speech * 0.032, ms: performance.now() - t0 };
}

// ---- streaming decode (QwenLM streaming_transcribe scheme) --------
// Each hop re-encodes only the open block, reuses the closed block's
// features (kept on the GPU), forces the previous text minus its last
// ROLLBACK tokens as the start of the answer and decodes only the
// continuation. Older blocks slide out with their text (closeBlocks),
// so compute and memory stay flat on an endless stream.
// A pause (no speech for QUIET_RESET_SEC) ends the utterance.
// 3 s, not the model's 8 s attention window: the prompt (two blocks of
// audio) and with it the prefill stay short, so hops keep a ~1-2 s
// cadence. 3 s is 300 mel frames, three whole 100-frame conv chunks, so
// no block is zero-padded mid-chunk.
const BLOCK = 3 * 16000;
const ROLLBACK = 5;
const newStream = (): StreamState => ({ open: new Float32Array(0), closed: [], tokens: [], histIds: [], quietSec: 0 });
const cols = new Map<string | null, { st: StreamState; vs: VadState | null }>(); // column id -> { st, vs }
let st = newStream();
const freeBlocks = (blocks: StreamState["closed"]) => {
    for (const b of blocks) b.af?.buffer.destroy();
};
const resetStream = () => {
    freeBlocks(st.closed);
    st.open = new Float32Array(0);
    st.closed = [];
    st.tokens = [];
    st.histIds = [];
    st.quietSec = 0;
};
// closeBlocks drops the oldest closed blocks; their features go with them.
const slide = () => {
    const before = st.closed.slice();
    const slid = closeBlocks(st, BLOCK, 1, ROLLBACK, HIST_CONTEXT, bytesOf);
    freeBlocks(before.filter((b) => !st.closed.includes(b)));
    return slid;
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
// Bound for a prompt with an empty audio span (a hop before 0.1 s of
// speech): the binding needs a buffer, and none of it is read.
let emptyAudio: GPUBuffer | null = null;
const noAudio = () => (emptyAudio ??= device!.createBuffer({ size: model!.D * 4, usage: GPUBufferUsage.STORAGE }));
// The audio span of the prompt as one buffer: the parts in order.
function concatFeatures(parts: EncoderResult[]): { buffer: GPUBuffer; owned: boolean } {
    if (parts.length === 1) return { buffer: parts[0].buffer, owned: false };
    const rowBytes = model!.D * 4;
    const frames = parts.reduce((a, p) => a + p.frames, 0);
    const buffer = device!.createBuffer({ size: Math.max(4, frames * rowBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const e = device!.createCommandEncoder();
    let at = 0;
    for (const p of parts) {
        e.copyBufferToBuffer(p.buffer, 0, buffer, at, p.frames * rowBytes);
        at += p.frames * rowBytes;
    }
    device!.queue.submit([e.finish()]);
    return { buffer, owned: true };
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
            const histIds = [...st.histIds, ...st.tokens].slice(-HIST_CONTEXT);
            resetStream();
            st.histIds = histIds;
            return { ...base, hist, conf: "", tent: "", ids: [], final: true, totalMs: performance.now() - t0 };
        }
        // Hold the tentative text; the next speech hop re-decodes it.
        slide();
        return { ...base, totalMs: performance.now() - t0 };
    }
    st.quietSec = 0;
    const slid = slide();
    const { cfg } = model!;
    await gate();
    const tMel = performance.now();
    let encMs = 0;
    const parts: EncoderResult[] = [];
    for (const b of st.closed) {
        if (!b.af) {
            await paced(m.pace, true);
            const te = performance.now();
            b.af = await encode(b.pcm);
            encMs += performance.now() - te;
        }
        parts.push(b.af);
    }
    let open: EncoderResult | null = null;
    if (st.open.length >= 1600) {
        await paced(m.pace, true);
        const te = performance.now();
        open = await encode(st.open);
        parts.push(open);
        encMs += performance.now() - te;
    }
    const A = parts.reduce((a, p) => a + p.frames, 0);
    const P = cfg.prompt;
    // The loop check runs on the text the user has seen + forced + new
    // tokens, since a loop crosses hops and history rows; only forced and
    // new tokens are ever cut. A loop is never fed back as forced text.
    const seen = st.histIds;
    const forced = trimLoop(seen, st.tokens.slice(0, Math.max(0, st.tokens.length - ROLLBACK)));
    const lead = [...seen, ...forced].slice(-LOOP_SPAN);
    const stopStream = (gen: readonly number[]) => loopStop([...lead, ...gen]);
    // History tail goes into the (otherwise empty) system turn.
    // A history loop is cut to one copy in the prompt only: it never primes
    // the decoder, and the history shown stays as it was.
    const pre = [...P.prefix_ids.slice(0, 3), ...trimLoop([], st.histIds.slice(-HIST_CONTEXT)), ...P.prefix_ids.slice(3)];
    const audioAt = pre.length;
    const ids = [...pre, ...Array<number>(A).fill(P.audio_pad_id), ...P.suffix_ids, ...(cfg.language_prefix_ids[m.lang] || []), ...forced];
    await paced(m.pace, true);
    const tPre = performance.now();
    const audio = parts.length ? concatFeatures(parts) : null;
    let r;
    try {
        r = await decodeGreedy(ids, audio?.buffer ?? noAudio(), audioAt, A, m.maxTokens, stopStream, m.pace);
    } finally {
        if (audio?.owned) audio.buffer.destroy();
        open?.buffer.destroy();
    }
    const tEnd = performance.now();
    const hist = slid.length ? textOf(slid) : "";
    const timing = { melEncMs: tPre - tMel, encMs, prefillMs: r.prefillMs, decodeMs: tEnd - tPre, totalMs: tEnd - t0, tokens: r.gen.length, prefill: ids.length, audioTokens: A, pace: m.pace ? { k: pacer.k, frameMs: +pacer.frameMs.toFixed(1), frames: pacer.frames } : null };
    const prevLen = st.tokens.length;
    st.tokens = trimLoop(seen, [...forced, ...r.gen]);
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
        const cx = cols.get(m.col ?? null);
        if (cx) freeBlocks(cx.st.closed);
        cols.delete(m.col ?? null);
        gates.delete(m.col ?? null);
        for (const r of gateWaiters.splice(0)) r();
        return;
    }
    queue = queue.then(async () => {
        curCol = m.col ?? null;
        if (m.type !== "load") {
            let cx = cols.get(curCol);
            if (!cx) cols.set(curCol, (cx = { st: newStream(), vs: newVadState() }));
            st = cx.st;
            vs = cx.vs;
        }
        try {
            if (m.type === "load") {
                configureLogging(toLogLevel(m.logLevel));
                await load(m);
            } else if (m.type === "stream") {
                const result = await streamHop(m);
                failOnGpuError();
                post({ type: "result", id: m.id, ...result });
            }
            else if (m.type === "run") {
                // VAD over the window's new audio: labels the hop, and
                // gates it unless m.vad is false.
                const v = m.newPcm ? await vadSpeech(m.newPcm) : null;
                const speech = v ? v.speechSec >= VAD_MIN_SPEECH_SEC : true;
                const vadInfo = { speechSec: v?.speechSec ?? null, vadMs: v?.ms ?? null, speech };
                if (!speech && m.vad !== false) post({ type: "result", id: m.id, ...vadInfo, skipped: "vad", text: "", totalMs: 0 });
                else {
                    const result = await transcribe(m.pcm, m.lang, m.maxTokens, m.pace);
                    failOnGpuError();
                    post({ type: "result", id: m.id, ...vadInfo, ...result });
                }
            }
        } catch (error) {
            post({ type: "error", id: m.id, ...errFields(error) });
        } finally {
            curCol = null;
            vs = null;
        }
    });
};
