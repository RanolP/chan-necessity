import { cbSplit, type SttHub } from "../shared/split-context.ts";
import { cbAudio } from "../shared/audio.ts";
import { mergeTranscript, commitPoint } from "./merge.ts";
import WORKER_SOURCE from "virtual:captions-worker";
import { loadEngineAssets } from "./assets.ts";
import { createCaptionHistory } from "./history.ts";
import { getLogger, KEY_LOG_LEVEL, toLogLevel } from "../shared/logtape.ts";
import { registerPlayerButton } from "../shared/player-bar.ts";
import { kstToMs, liveOpenDate, watchedWallTime } from "../shared/broadcast-time.ts";
import type { CapsMessage, ErrorMessage, PageMessage, PaceStats, ReadyMessage, ResultMessage, WorkerMessage } from "./protocol.ts";

const logger = getLogger(["captions"]);
const hubLogger = getLogger(["captions", "hub"]);

declare global {
    interface Window {
        ChzzkBestStt?: object;
    }
}

// What the page posts to: its own Worker, or its column's port on the hub.
interface SttPort {
    postMessage(m: PageMessage, transfer?: Transferable[]): void;
    terminate(): void;
}
type ErrorLike = { name?: string; message?: string; stack?: string } | null | undefined;
type SttVideo = HTMLVideoElement & {
    __cbSttSeekHook?: boolean;
};
interface Mp4Box {
    type: string;
    start: number;
    end: number;
}
interface SoundTrack {
    trackId: number;
    timescale: number;
}
interface MediaChunk {
    start: number;
    end: number;
    pcm: Float32Array;
}
interface JobBase {
    newSec: number;
    sec: number;
}
interface MediaJob extends JobBase {
    source: "media";
    end: number;
}
interface RealtimeJob extends JobBase {
    source: "realtime";
    capturedAt: number;
}
type Job = MediaJob | RealtimeJob;
type WindowShown = Job & { id: number; text: string; tokens?: number; ms: number; stream?: undefined };
type StreamShown = Job & ResultMessage & { stream: true; text: string; ms: number };
type DueItem = (MediaJob & { id: number; text: string; tokens?: number; ms: number; stream?: undefined }) | (MediaJob & ResultMessage & { stream: true; text: string; ms: number });
interface Word {
    t: string;
    tent: boolean;
}
interface Look {
    size: string;
    bg: number;
    pos: string;
}
type PlayerHost = HTMLElement & { cbSubWatched?: boolean };

// ---- 실시간 자막 (speech-to-text) ---------------------------------------
// Transcribes the player's own audio with Qwen3-ASR in the browser:
// hand-written WebGPU kernels inside a Web Worker, weights read out of the
// jiangzhuo9357 layout-v2 ONNX exports, cached in Cache Storage so only the
// first enable downloads them. Off by default; the download starts only
// when the viewer turns the subtitle button on. Right-click the button for
// the model choice, cache controls and the caption history popup.
//
// Audio path: cbAudio's capture bus (see shared/audio.ts) → AudioWorklet that downmixes
// and decimates to 16 kHz mono, on the shared cbAudio context. That context
// runs at the device rate; a 16 kHz context would resample what the viewer
// hears.
(() => {
    const LIVE_RE = /^\/live\/([0-9a-f]{32})/i;
    const VIDEO_RE = /^\/video\/\d+/;
    const captionLog = createCaptionHistory({
        cap: 2000,
        live: () => LIVE_RE.test(location.pathname),
        logger,
        broadcastOpen: (path) => {
            const channelId = LIVE_RE.exec(path)?.[1];
            if (!channelId) return Promise.resolve(null);
            return liveOpenDate(channelId).then(
                (openDate) => {
                    const ms = kstToMs(openDate);
                    if (Number.isFinite(ms)) return ms;
                    logger.warn("history open date unreadable, showing wall clock {channelId} {openDate}", { channelId, openDate });
                    return null;
                },
                (error) => {
                    logger.warn("history open date fetch failed, showing wall clock {channelId} {error}", { channelId, error });
                    return null;
                },
            );
        },
        watched: (open) => watchedWallTime(open ?? NaN).wallTime,
    });
    const SR = 16000;
    const WINDOW_SEC = 6;
    // Live parts arrive about once a second; the next window goes out as
    // soon as the model is free and this much new audio exists.
    const MIN_HOP_SEC = 0.5;
    // Audio read from the player's buffer is ahead of the playhead; a
    // result is shown once playback is this close to the window's end.
    const SHOW_LEAD_SEC = 0.5;
    const MAX_AHEAD_SEC = 8;
    const MSE_STALE_MS = 4000;
    // Decoded parts kept behind the playhead (16 kHz mono, ~64 KB/s; the
    // part count cap below bounds it too).
    const KEEP_BEHIND_SEC = 90;
    // A playhead move larger than this between two checks is a seek; the
    // player's own gap-skipping micro-seeks stay below it.
    const SEEK_JUMP_SEC = 2;
    const SEEK_FIRST_HOP_SEC = 2.5;
    const SILENCE_RMS = 0.004;
    const CLEAR_AFTER_MS = 7000;
    const CACHE_NAME = "chzzkbest-stt-v1";
    const MODELS = {
        "1.7B": {
            repo: "jiangzhuo9357/Qwen3-ASR-1.7B-ONNX",
            rev: "fcc238dfdc95cdcccaa9a7e2c7f5abc2f94f44a7",
        },
        "0.6B": {
            repo: "jiangzhuo9357/Qwen3-ASR-0.6B-ONNX",
            rev: "4a01b95fafe2c9e3af77e33c18bbb7de349c62f6",
        },
    };
    const KEY_ENABLED = "stt.enabled";
    const KEY_MODEL = "stt.model";
    const ICON =
        '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true" class="pzp-ui-icon__svg"><rect x="8" y="11" width="20" height="14" rx="2.5" stroke="currentColor" stroke-width="1.7"/><path d="M16.2 16.2a2.4 2.4 0 1 0 0 3.6M23.2 16.2a2.4 2.4 0 1 0 0 3.6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';

    const pageWindow =
        typeof unsafeWindow === "undefined" ? window : unsafeWindow;
    type ModelName = keyof typeof MODELS;
    const getSetting = <T,>(key: string, fallback: T): T => {
        try {
            return GM_getValue(key, fallback);
        } catch (error) {
            logger.error("setting load failed {key} {error}", { key, error });
            return fallback;
        }
    };
    const modelName = () => {
        const m = getSetting(KEY_MODEL, "1.7B") as ModelName;
        return MODELS[m] ? m : "1.7B";
    };

    // Fetches the engine's files first; the worker queues nothing until
    // its load message, and the page sends audio only after "ready".
    const postLoad = async (worker: SttPort, name: ModelName) => {
        const engineAssets = name === "1.7B" ? await loadEngineAssets(CACHE_NAME, getSetting) : null;
        worker.postMessage({ type: "load", ...MODELS[name], cacheName: CACHE_NAME, engineAssets, logLevel: toLogLevel(GM_getValue(KEY_LOG_LEVEL)) }, engineAssets ? [engineAssets.manifest, engineAssets.qknorm] : []);
    };

    function makeWorker(name: string) {
        const url = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: "text/javascript" }));
        const worker = new Worker(url, { type: "module" });
        URL.revokeObjectURL(url);
        logger.info("worker created {name} {path}", { name, path: location.pathname });
        return worker;
    }

    // Split view: the model loads once, in the top-level page. Each column
    // frame keeps its own capture, lines and overlay, and sends its audio to
    // this one worker tagged with its column id; the worker keeps decode and
    // VAD state per column and runs all requests through one queue. With no
    // column connected for a while the worker (and its GPU device) goes.
    function installSttHub() {
        const hub: {
            worker: Worker | null;
            model: string;
            clients: Map<string | null, (m: WorkerMessage) => void>;
            last: { caps?: CapsMessage; ready?: ReadyMessage };
            created: number;
            idleTimer: number;
        } = { worker: null, model: "", clients: new Map(), last: {}, created: 0, idleTimer: 0 };
        const deliver = (col: string | null, fn: (m: WorkerMessage) => void, m: WorkerMessage) => {
            try {
                fn(m);
            } catch (error) {
                // The column's frame is gone (a dead wrapper throws).
                hub.clients.delete(col);
                hub.worker?.postMessage({ type: "drop", col });
            }
        };
        function spawn(name: ModelName) {
            hub.worker?.terminate();
            hub.last = {};
            const worker = makeWorker(name);
            hub.created++;
            hub.worker = worker;
            hub.model = name;
            worker.onmessage = (ev: MessageEvent<WorkerMessage>) => {
                const m = ev.data;
                if (m.type === "caps") hub.last[m.type] = m;
                else if (m.type === "ready") hub.last[m.type] = m;
                if (m.col !== undefined) {
                    const fn = hub.clients.get(m.col);
                    if (fn) deliver(m.col, fn, m);
                } else for (const [col, fn] of [...hub.clients]) deliver(col, fn, m);
            };
            worker.onerror = (ev) => {
                const m: ErrorMessage = { type: "error", stage: "worker", name: "WorkerError", message: `${ev.message} (${ev.filename}:${ev.lineno}:${ev.colno})`, stack: "" };
                if (hub.worker === worker) hub.worker = null;
                for (const [col, fn] of [...hub.clients]) deliver(col, fn, m);
            };
            postLoad(worker, name);
        }
        function idleLater() {
            if (hub.clients.size || !hub.worker) return;
            const worker = hub.worker;
            clearTimeout(hub.idleTimer);
            hub.idleTimer = setTimeout(() => {
                if (hub.clients.size || hub.worker !== worker) return;
                worker.terminate();
                hub.worker = null;
                hub.last = {};
                hubLogger.info("idle; worker stopped");
            }, 15000);
        }
        pageWindow.ChzzkBestSttHub = {
            stats: () => ({ workersCreated: hub.created, alive: !!hub.worker, model: hub.model, clients: [...hub.clients.keys()] }),
            // The split page calls this when it removes a column's frame.
            drop(col) {
                if (!hub.clients.delete(col)) return;
                hub.worker?.postMessage({ type: "drop", col });
                idleLater();
            },
            connect(col, name, fn) {
                clearTimeout(hub.idleTimer);
                if (!hub.worker || hub.model !== name) spawn(name as ModelName);
                const worker = hub.worker!;
                hub.clients.set(col, fn);
                for (const m of [hub.last.caps, hub.last.ready]) if (m) setTimeout(() => hub.clients.get(col) === fn && deliver(col, fn, m));
                const mine = () => hub.clients.get(col) === fn && hub.worker === worker;
                return {
                    postMessage(m, transfer) {
                        if (mine()) worker.postMessage({ ...m, col }, transfer || []);
                    },
                    terminate() {
                        if (!mine()) return;
                        hub.clients.delete(col);
                        worker.postMessage({ type: "drop", col });
                        idleLater();
                    },
                };
            },
        };
    }
    if (!cbSplit.frame && window.top === window) installSttHub();


    // ---- audio tap -------------------------------------------------------
    const WORKLET_SRC = `
class CbSttTap extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / ${SR}; this.acc = 0; this.n = 0; this.pos = 0; this.buf = new Float32Array(1600); this.len = 0; }
  process(inputs) {
    const ch = inputs[0];
    if (!ch || !ch.length) return true;
    const frames = ch[0].length;
    for (let i = 0; i < frames; i++) {
      let s = 0; for (let c = 0; c < ch.length; c++) s += ch[c][i];
      this.acc += s / ch.length; this.n++; this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        this.buf[this.len++] = this.acc / this.n; this.acc = 0; this.n = 0;
        if (this.len === this.buf.length) { this.port.postMessage(this.buf, [this.buf.buffer]); this.buf = new Float32Array(1600); this.len = 0; }
      }
    }
    return true;
  }
}
registerProcessor('cb-stt-tap', CbSttTap);`;

    const RING_SEC = 30;
    const ring = new Float32Array(SR * RING_SEC);
    let ringWrite = 0; // total samples ever written
    let lastSampleAt = 0; // performance.now() when the last chunk arrived

    let tapNode: AudioWorkletNode | null = null;
    let tapped: GainNode | null = null; // cbAudio's capture bus while connected

    function pushSamples(chunk: Float32Array) {
        for (let i = 0; i < chunk.length; i++) ring[(ringWrite + i) % ring.length] = chunk[i];
        ringWrite += chunk.length;
        lastSampleAt = performance.now();
    }
    function resetRing() {
        ringWrite = 0;
        state.lastEnd = 0;
    }
    function readLast(n: number) {
        const out = new Float32Array(n);
        const start = ringWrite - n;
        for (let i = 0; i < n; i++) out[i] = ring[(((start + i) % ring.length) + ring.length) % ring.length];
        return out;
    }

    // Inside a user gesture: starts the shared context and the worklet.
    async function ensureAudio() {
        const ctx = await cbAudio.context();
        if (!ctx) return false;
        if (!tapNode || tapNode.context !== ctx) {
            const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "text/javascript" }));
            await ctx.audioWorklet.addModule(url);
            URL.revokeObjectURL(url);
            tapNode = new AudioWorkletNode(ctx, "cb-stt-tap", { numberOfOutputs: 0 });
            tapNode.port.onmessage = (ev: MessageEvent<Float32Array>) => {
                if (state.enabled) pushSamples(ev.data);
            };
        }
        return true;
    }

    // The worklet hangs off cbAudio's capture bus once; the shared graph
    // rebinds the bus to whatever <video> and track the player has, so a pan
    // change or a swapped element never touches this connection. A new
    // element or source restarts the ring, since its audio is discontinuous.
    function retap() {
        const video: SttVideo | null = cbAudio.playerVideo();
        if (!tapNode || !video) return;
        if (!tapped) {
            const bus = cbAudio.openCapture();
            if (!bus) return;
            bus.connect(tapNode);
            tapped = bus;
            resetRing();
            logger.info("tapped capture bus {src}", { src: video.currentSrc.slice(0, 60) });
        }
        if (!video.__cbSttSeekHook) {
            video.__cbSttSeekHook = true;
            video.addEventListener("seeking", resetRing);
            video.addEventListener("seeking", resetMedia);
        }
    }
    function untap() {
        if (!tapped) return;
        try {
            tapped.disconnect(tapNode!);
        } catch (error) {
            // already disconnected
        }
        cbAudio.closeCapture();
        tapped = null;
    }
    cbAudio.onVideoChange((video, change) => {
        if (!tapped) return;
        logger.info("capture restarts on new {change} {src}", { change, src: video.currentSrc.slice(0, 60) });
        resetRing();
        retap();
    });

    // ---- buffer tap: audio ahead of the playhead -------------------------
    // The player feeds fragmented MP4 parts (audio muxed with video, about
    // one second each) into a SourceBuffer some seconds before playback
    // reaches them. Each part is decoded on its own, prefixed with the init
    // segment, and filed under its media time (tfdt + timestampOffset), so a
    // window can be transcribed before it is heard and shown on time. This
    // path needs no AudioContext and ignores volume, mute and pan. Where no
    // parts arrive (native HLS, a player without MSE), the realtime tap above
    // takes over.
    const u32 = (u8: Uint8Array, p: number) => ((u8[p] << 24) | (u8[p + 1] << 16) | (u8[p + 2] << 8) | u8[p + 3]) >>> 0;
    const fourcc = (u8: Uint8Array, p: number) => String.fromCharCode(u8[p], u8[p + 1], u8[p + 2], u8[p + 3]);
    function* boxes(u8: Uint8Array, from = 0, to = u8.length): Generator<Mp4Box> {
        let p = from;
        while (p + 8 <= to) {
            let size = u32(u8, p);
            let head = 8;
            if (size === 1) {
                size = u32(u8, p + 8) * 2 ** 32 + u32(u8, p + 12);
                head = 16;
            } else if (size === 0) size = to - p;
            if (size < head || p + size > to) return;
            yield { type: fourcc(u8, p + 4), start: p + head, end: p + size };
            p += size;
        }
    }
    const child = (u8: Uint8Array, box: Mp4Box, type: string) => {
        for (const b of boxes(u8, box.start, box.end)) if (b.type === type) return b;
        return null;
    };
    // init segment → { trackId, timescale } of the first sound track
    function parseInit(u8: Uint8Array): SoundTrack | null {
        for (const moov of boxes(u8)) {
            if (moov.type !== "moov") continue;
            for (const trak of boxes(u8, moov.start, moov.end)) {
                if (trak.type !== "trak") continue;
                const tkhd = child(u8, trak, "tkhd");
                const mdia = child(u8, trak, "mdia");
                const mdhd = mdia && child(u8, mdia, "mdhd");
                const hdlr = mdia && child(u8, mdia, "hdlr");
                if (!tkhd || !mdhd || !hdlr || fourcc(u8, hdlr.start + 8) !== "soun") continue;
                return {
                    trackId: u32(u8, tkhd.start + 4 + (u8[tkhd.start] ? 16 : 8)),
                    timescale: u32(u8, mdhd.start + 4 + (u8[mdhd.start] ? 16 : 8)),
                };
            }
        }
        return null;
    }
    // media part → decode time of the sound track in seconds, or null
    function audioDecodeTime(u8: Uint8Array, init: SoundTrack) {
        for (const moof of boxes(u8)) {
            if (moof.type !== "moof") continue;
            for (const traf of boxes(u8, moof.start, moof.end)) {
                if (traf.type !== "traf") continue;
                const tfhd = child(u8, traf, "tfhd");
                const tfdt = child(u8, traf, "tfdt");
                if (!tfhd || !tfdt || u32(u8, tfhd.start + 4) !== init.trackId) continue;
                const t = u8[tfdt.start] ? u32(u8, tfdt.start + 4) * 2 ** 32 + u32(u8, tfdt.start + 8) : u32(u8, tfdt.start + 4);
                return t / init.timescale;
            }
        }
        return null;
    }

    const media: {
        inits: WeakMap<SourceBuffer, (SoundTrack & { bytes: Uint8Array }) | null>;
        chunks: MediaChunk[];
        lastAt: number;
        decodeErrors: number;
    } = { inits: new WeakMap(), chunks: [], lastAt: 0, decodeErrors: 0 };
    let decodeCtx: OfflineAudioContext | null = null;
    function addChunk(start: number, pcm: Float32Array) {
        const end = start + pcm.length / SR;
        // A re-append of the same span (quality switch) replaces the old one.
        media.chunks = media.chunks.filter((c) => c.end <= start + 0.02 || c.start >= end - 0.02);
        media.chunks.push({ start, end, pcm });
        media.chunks.sort((a, b) => a.start - b.start);
        const video = cbAudio.playerVideo();
        // Played audio is kept for a while: a short 타임머신 rewind lands in
        // the player's buffer, and the player appends nothing for it again.
        const keepFrom = (video?.currentTime ?? end) - KEEP_BEHIND_SEC;
        media.chunks = media.chunks.filter((c) => c.end > keepFrom).slice(-120);
        media.lastAt = performance.now();
    }
    function onAppend(sourceBuffer: SourceBuffer, data: BufferSource) {
        const u8 = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        if (u8.length < 16) return;
        const type = fourcc(u8, 4);
        if (type === "ftyp" || type === "moov") {
            const info = parseInit(u8);
            media.inits.set(sourceBuffer, info && { ...info, bytes: u8.slice() });
            return;
        }
        const init = media.inits.get(sourceBuffer);
        if (!init || !state.enabled) return;
        const t = audioDecodeTime(u8, init);
        if (t === null) return;
        const start = t + sourceBuffer.timestampOffset;
        const both = new Uint8Array(init.bytes.length + u8.length);
        both.set(init.bytes);
        both.set(u8, init.bytes.length);
        decodeCtx ??= new OfflineAudioContext(1, 1, SR);
        decodeCtx.decodeAudioData(both.buffer).then(
            (buf) => {
                const pcm = new Float32Array(buf.length);
                for (let c = 0; c < buf.numberOfChannels; c++) {
                    const d = buf.getChannelData(c);
                    for (let i = 0; i < d.length; i++) pcm[i] += d[i] / buf.numberOfChannels;
                }
                addChunk(start, pcm);
            },
            (error) => {
                if (media.decodeErrors++ < 3) logger.warn("part decode failed {error}", { error });
            }
        );
    }
    {
        const proto = pageWindow.SourceBuffer?.prototype;
        const append = proto?.appendBuffer;
        if (append)
            proto.appendBuffer = function (this: SourceBuffer, data: BufferSource) {
                try {
                    onAppend(this, data);
                } catch (error) {
                    sttError("buffer tap failed", "appendBuffer", error);
                }
                return append.call(this, data);
            };
    }
    // The buffer tap serves only while parts arrive and some part lies
    // within reach of the playhead; after a seek into audio the tap never
    // saw (or already dropped), the realtime tap takes over until it does.
    const mediaActive = () => {
        if (state.config.source === "realtime" || performance.now() - media.lastAt >= MSE_STALE_MS) return false;
        const ct = cbAudio.playerVideo()?.currentTime;
        if (ct === undefined) return true;
        const next = media.chunks.find((c) => c.end > ct);
        return !!next && next.start <= ct + MAX_AHEAD_SEC;
    };
    // End of the decoded audio that runs on without a gap from `t` (or from
    // the first part after `t`, right after the tap starts).
    function coveredEnd(t: number) {
        let end = -Infinity;
        for (const c of media.chunks) {
            if (c.end <= t) continue;
            if (end === -Infinity) end = c.end;
            else if (c.start <= end + 0.1) end = Math.max(end, c.end);
            else break;
        }
        return end;
    }
    function readMedia(from: number, to: number) {
        const out = new Float32Array(Math.round((to - from) * SR));
        for (const c of media.chunks) {
            if (c.end <= from || c.start >= to) continue;
            const off = Math.round((c.start - from) * SR);
            const a = Math.max(0, -off);
            const b = Math.min(c.pcm.length, out.length - off);
            if (b > a) out.set(c.pcm.subarray(a, b), off + a);
        }
        return out;
    }

    // ---- engine ----------------------------------------------------------
    const state: {
        enabled: boolean;
        phase: string;
        statusText: string;
        text: string;
        textAt: number;
        delayS: number | null;
        delayId?: number;
        textEpoch: number;
        inferFails: number;
        worker: SttPort | null;
        workerModel: string;
        busy: boolean;
        lastEnd: number;
        lastEndMedia: number;
        lastCt: number;
        due: DueItem[];
        runId: number;
        config: { source: string; minHopSec: number; pace: string; decode: string; streamHopSec: number; vad: boolean; engine: boolean };
        tent: string;
        history: string;
        peak: number;
        workerRaf: boolean | null;
    } = {
        enabled: false,
        phase: "off", // off | loading | ready | error
        statusText: "",
        text: "",
        textAt: 0,
        // Processing time: seconds the worker spent on the window, excluding
        // any wait for the playhead (see procS).
        delayS: null,
        // Bumped when text starts again after the overlay cleared, so the
        // subtitle lines start over on a fresh line.
        textEpoch: 0,
        inferFails: 0,
        worker: null,
        workerModel: "",
        busy: false,
        lastEnd: 0, // realtime tap: ring sample count at the last window
        lastEndMedia: -Infinity, // buffer tap: media time of the last window
        lastCt: 0,
        due: [], // buffer-tap results waiting for the playhead
        runId: 0,
        // source: "auto" (buffer tap when the player feeds one) | "realtime"
        // pace: "raf" slices GPU work per display frame; "free" runs flat out
        // (both still under the duty cycle below).
        // decode: "stream" (incremental, rolling 8–16 s context) | "window"
        // (independent 6 s windows merged by text); vad: gate on speech.
        config: { source: "auto", minHopSec: MIN_HOP_SEC, pace: "raf", decode: "stream", streamHopSec: 1, vad: true, engine: getSetting<unknown>("dev.engine", true) !== false },
        tent: "", // streaming: the last few tokens, still open to revision
        history: "", // streaming: text whose audio left the context
        peak: 0,
        workerRaf: null,
    };
    // Optional stages between the audio and the model, each
    // (pcm: Float32Array @16 kHz) => Float32Array | Promise<Float32Array>,
    // run in order; voice/BGM separation would go here.
    const preprocess: ((pcm: Float32Array<ArrayBuffer>, sr: number) => Float32Array<ArrayBuffer> | Promise<Float32Array<ArrayBuffer>>)[] = [];
    // e2eS: seconds from a word being heard to its text on screen, sampled
    // at nine points across the audio each window added (negative = early).
    const stats: {
        stream: { hops: number; decoded: number; gated: number; confirmS: number[]; hopMs: ResultMessage[]; vadMs: number[] };
        vad: { speech: number; nonspeech: number; nonspeechEmitted: number; speechEmitted: number };
        windows: number;
        silent: number;
        skippedHops: number;
        inferMs: number[];
        latencyMs: number[];
        rtf: number[];
        e2eS: number[];
        waitS: number[];
        renderMs: number[];
        readyMarginS: number[];
        bySource: { media: number; realtime: number };
        load: (ReadyMessage & { wallMs: number }) | null;
        log: Record<string, unknown>[];
        workerRaf?: boolean;
        pace?: PaceStats;
        aborted?: number;
        lines?: typeof lines;
    } = { stream: { hops: 0, decoded: 0, gated: 0, confirmS: [], hopMs: [], vadMs: [] }, vad: { speech: 0, nonspeech: 0, nonspeechEmitted: 0, speechEmitted: 0 }, windows: 0, silent: 0, skippedHops: 0, inferMs: [], latencyMs: [], rtf: [], e2eS: [], waitS: [], renderMs: [], readyMarginS: [], bySource: { media: 0, realtime: 0 }, load: null, log: [] };
    pageWindow.ChzzkBestStt = { stats, state, media, preprocess, mergeTranscript };

    function startWorker() {
        const name = modelName();
        if (state.worker && state.workerModel === name) return;
        if (!rafLast) requestAnimationFrame(rafLoop);
        state.worker?.terminate();
        const hub = cbSplit.frame ? (cbSplit.hub() as SttHub<WorkerMessage, PageMessage> | null) : null;
        if (cbSplit.frame && !hub) return fail("split view: the top page has no subtitle hub");
        const t0 = performance.now();
        const worker: SttPort = hub ? hub.connect(cbSplit.col, name, (m) => onWorker(m, t0)) : makeWorker(name);
        state.worker = worker;
        state.workerModel = name;
        state.phase = "loading";
        state.busy = false;
        setStatus(`${name} 모델 준비 중…`);
        if (!hub) {
            (worker as Worker).onmessage = (ev: MessageEvent<WorkerMessage>) => onWorker(ev.data, t0);
            (worker as Worker).onerror = (ev) => fail(`worker error: ${ev.message} (${ev.filename}:${ev.lineno}:${ev.colno})`);
        }
        if (!hub) postLoad(worker, name);
        navigator.storage?.persist?.().catch(() => {});
    }

    // Same fields the worker sends, for the errors raised on this side.
    const sttError = (what: string, stage: string, error: unknown) =>
        logger.error("{what} at {stage}: {name}: {message} {stack}", { what, stage, name: (error as ErrorLike)?.name ?? typeof error, message: (error as ErrorLike)?.message ?? String(error), stack: (error as ErrorLike)?.stack ?? "" });

    // Stops the worker for good: nothing restarts it until the user turns
    // subtitles off and on, so a GPU fault is never retried in a loop.
    function fail(message: string) {
        logger.error("{message}", { message });
        state.worker?.terminate();
        state.worker = null;
        state.phase = "error";
        state.busy = false;
        setStatus(`자막 오류: ${message.split("\n")[0].slice(0, 120)}`);
    }

    const mb = (n: number) => (n / 1048576).toFixed(0);
    function onWorker(m: WorkerMessage, t0: number) {
        if (m.type === "caps") {
            state.workerRaf = m.workerRaf;
            stats.workerRaf = m.workerRaf;
        } else if (m.type === "progress") {
            const pct = m.total ? Math.min(100, (100 * m.loaded) / m.total).toFixed(0) : "?";
            setStatus(
                m.fromNet > 0
                    ? `${state.workerModel} 모델 다운로드 ${pct}% (${mb(m.loaded)}/${mb(m.total)} MB)`
                    : `${state.workerModel} 캐시에서 불러오는 중 ${pct}%`
            );
        } else if (m.type === "status") {
            setStatus(m.text);
        } else if (m.type === "ready") {
            stats.load = { ...m, wallMs: performance.now() - t0 };
            logger.info("ready {load}", { load: stats.load });
            state.phase = "ready";
            setStatus("");
        } else if (m.type === "result") {
            state.busy = false;
            state.inferFails = 0;
            const job = pending.get(m.id);
            pending.delete(m.id);
            if (!job || m.id !== state.runId) return;
            if (m.speech !== undefined && m.speechSec !== null) {
                const emitted = m.stream ? !!m.grew : !!m.text;
                if (m.speech) {
                    stats.vad.speech++;
                    if (emitted) stats.vad.speechEmitted++;
                } else {
                    stats.vad.nonspeech++;
                    if (emitted) stats.vad.nonspeechEmitted++;
                }
            }
            if (m.vadMs != null) stats.stream.vadMs.push(m.vadMs);
            if (m.stream) {
                stats.stream.hops++;
                if (!m.decoded && !m.aborted) stats.stream.gated++;
                if (m.decoded || m.aborted) {
                    stats.inferMs.push(m.totalMs);
                    stats.stream.hopMs.push(m);
                    if (m.pace) stats.pace = m.pace;
                }
                if (m.aborted) stats.aborted = (stats.aborted || 0) + 1;
                if (!m.decoded && !m.final && !m.hist) return;
                const res = { ...job, ...m, text: (m.conf ?? "") + (m.tent ?? ""), ms: m.totalMs };
                if (job.source === "media") {
                    state.due.push(res as DueItem);
                    showDue();
                } else showStream(res as StreamShown, (performance.now() - job.capturedAt) / 1000);
                return;
            }
            if (m.skipped) return;
            stats.inferMs.push(m.totalMs);
            if (m.pace) stats.pace = m.pace;
            if (m.aborted) stats.aborted = (stats.aborted || 0) + 1;
            stats.rtf.push(m.totalMs / 1000 / job.sec);
            if (!m.text) return;
            if (job.source === "media") {
                const video = cbAudio.playerVideo();
                if (video) stats.readyMarginS.push(job.end - SHOW_LEAD_SEC - video.currentTime);
                state.due.push({ ...job, id: m.id, text: m.text, tokens: m.tokens, ms: m.totalMs });
                showDue();
            } else {
                stats.latencyMs.push(performance.now() - job.capturedAt);
                show({ ...job, id: m.id, text: m.text, tokens: m.tokens, ms: m.totalMs }, (frac) => (performance.now() - job.capturedAt) / 1000 + (1 - frac) * job.newSec);
            }
        } else if (m.type === "error") {
            if (m.id && ++state.inferFails < 3) {
                state.busy = false;
                logger.error("inference failed at {stage}: {name}: {message} {stack}", { stage: m.stage, name: m.name, message: m.message, stack: m.stack });
            } else {
                logger.error("load failed at {stage}: {name}: {message} {stack}", { stage: m.stage, name: m.name, message: m.message, stack: m.stack });
                fail(`${m.name}: ${m.message}`);
            }
        }
    }

    const procS = (shown: { ms: number }) => shown.ms / 1000;
    function show(job: WindowShown, e2eAt: (frac: number) => number) {
        const now = performance.now();
        const fresh = state.textAt && now - state.textAt < CLEAR_AFTER_MS;
        if (!fresh) state.textEpoch++;
        const text = mergeTranscript(fresh ? state.text : "", job.text).slice(-300);
        // A window that repeats what is shown is no new speech.
        if (!fresh || text !== state.text) state.textAt = now;
        state.text = text;
        state.tent = "";
        state.delayS = procS(job);
        state.delayId = job.id;
        for (let i = 1; i <= 9; i++) stats.e2eS.push(e2eAt(i / 10));
        if (stats.log.length < 400) stats.log.push({ t: Math.round(now), src: job.source, raw: job.text, tokens: job.tokens, ms: Math.round(job.ms) });
        render();
        if (!document.hidden) requestAnimationFrame(() => stats.renderMs.push(performance.now() - now));
    }
    // Streaming results replace the open text instead of merging into it.
    // tokenBirth[i] remembers when token i (at that id) was first heard, so
    // the time to confirmation can be measured once it leaves the tail.
    let tokenBirth: { id: number; at: number; confirmed: boolean }[] = [];
    // The worker's context ended (a final, or a hop sent with reset): its
    // next text is new speech, so no carried tail applies to it.
    let streamEnded = false;
    function showStream(res: StreamShown, lagS: number) {
        const now = performance.now();
        const fresh = state.textAt && now - state.textAt < CLEAR_AFTER_MS;
        if (!fresh) state.history = "";
        if (!fresh && (res.decoded || res.final)) state.textEpoch++;
        if (res.hist) state.history = (state.history + " " + res.hist).trim().slice(-300);
        if (res.final) tokenBirth = [];
        if (res.final) streamEnded = true;
        else if (res.decoded && streamEnded) {
            streamEnded = false;
            lines.carry = "";
        }
        if (res.decoded) {
            const heardAt = now - (lagS + res.newSec / 2) * 1000; // mid-hop
            const ids = res.ids!;
            if (res.slidN) tokenBirth = tokenBirth.slice(res.slidN);
            // Only new or revised tokens count as speech for CLEAR_AFTER_MS,
            // not a hop that returned the same text.
            if (ids.length !== tokenBirth.length || ids.some((id, i) => tokenBirth[i].id !== id)) state.textAt = now;
            const birth: { id: number; at: number; confirmed: boolean }[] = [];
            for (let i = 0; i < ids.length; i++) {
                const old = tokenBirth[i];
                birth.push(old && old.id === ids[i] ? old : { id: ids[i], at: heardAt, confirmed: false });
            }
            const nConf = Math.max(0, ids.length - 5);
            for (let i = 0; i < nConf; i++)
                if (!birth[i].confirmed) {
                    birth[i].confirmed = true;
                    stats.stream.confirmS.push((now - birth[i].at) / 1000);
                }
            tokenBirth = birth;
            stats.stream.decoded++;
            for (let i = 1; i <= 9; i++) stats.e2eS.push(lagS + (1 - i / 10) * res.newSec);
        }
        if (res.conf !== null && res.conf !== undefined) {
            state.text = (state.history + " " + res.conf).trim().slice(-300);
            state.tent = res.tent!;
        } else if (res.final) {
            state.text = state.history;
            state.tent = "";
        }
        if (res.decoded || res.final) {
            state.delayS = procS(res);
            state.delayId = res.id;
        }
        if (stats.log.length < 400 && res.decoded) stats.log.push({ t: Math.round(now), src: res.source, conf: res.conf, tent: res.tent, hist: res.hist, tokens: res.tokens, ms: Math.round(res.ms) });
        render();
    }
    // Buffer-tap results wait for the playhead, then show in window order.
    function showDue() {
        const video = cbAudio.playerVideo();
        if (!video || !state.due.length) return;
        state.due.sort((a, b) => a.end - b.end);
        while (state.due.length && video.currentTime >= state.due[0].end - SHOW_LEAD_SEC) {
            const job = state.due.shift()!;
            const ct = video.currentTime;
            if (job.stream) showStream(job, ct - job.end);
            else show(job, (frac) => ct - (job.end - job.newSec * (1 - frac)));
        }
    }
    function resetMedia() {
        state.lastEndMedia = -Infinity;
        state.due.length = 0;
    }
    // A seek (타임머신 rewind, its return to live, a VOD jump) makes every
    // window, result and shown line from the old position wrong at the new
    // one. `from` is the playhead before the jump: lines still on screen
    // are recorded in the history at that moment, not at the new one.
    let seekReset = false; // the next streaming hop starts a fresh context
    function onSeek(video: HTMLVideoElement, from: number) {
        const to = video.currentTime;
        logger.info("seek {from} → {to}, captions restart", { from, to });
        state.lastCt = to;
        state.runId++; // drops the result still in flight
        seekReset = true;
        resetRing();
        resetMedia();
        resetLines({ media: from, shiftSec: to - from });
        state.text = "";
        state.tent = "";
        state.history = "";
        state.textAt = 0;
        state.textEpoch++;
        tokenBirth = [];
        render();
    }
    document.addEventListener(
        "seeking",
        (event) => {
            const video = cbAudio.playerVideo();
            if (!state.enabled || !video || event.target !== video) return;
            if (Math.abs(video.currentTime - state.lastCt) > SEEK_JUMP_SEC) onSeek(video, state.lastCt);
        },
        true
    );

    // A worker without its own requestAnimationFrame paces GPU work on the
    // page's frames: one tick per frame while a job runs.
    let rafLast = 0;
    const rafLoop = (t: number) => {
        if (state.workerRaf === false && state.busy) state.worker?.postMessage({ type: "tick" });
        rafLast = t;
        if (state.enabled) requestAnimationFrame(rafLoop);
        else rafLast = 0;
    };
    // A hidden tab's timers slow to 1 Hz once it is silent and to about once
    // a minute after 5 minutes; a worker's timers keep their rate, so while
    // hidden a worker clock drives scheduling.
    let clock: Worker | null = null;
    function syncClock() {
        const want = document.hidden && state.enabled;
        if (want === !!clock) return;
        if (!want) {
            clock!.terminate();
            clock = null;
            return;
        }
        try {
            const url = URL.createObjectURL(new Blob(["setInterval(() => postMessage(0), 100)"], { type: "text/javascript" }));
            clock = new Worker(url);
            URL.revokeObjectURL(url);
        } catch (error) {
            logger.warn("hidden-tab clock unavailable, page timers only {error}", { error });
            return;
        }
        clock.onmessage = () => {
            syncClock();
            schedule();
        };
    }
    document.addEventListener("visibilitychange", syncClock);

    const pending = new Map<number, Job>();
    // Always transcribe the newest audio; when inference is slower than the
    // audio arrives, the next window simply covers more new audio.
    function schedule() {
        if (!state.enabled) return;
        const video = cbAudio.playerVideo();
        if (video) {
            // A jump of the playhead the seeking event did not report (live
            // resync, a throttled tab) restarts the captions the same way.
            if (Math.abs(video.currentTime - state.lastCt) > SEEK_JUMP_SEC) onSeek(video, state.lastCt);
            state.lastCt = video.currentTime;
        }
        showDue();
        // A paused player would only feed the model silence.
        if (state.phase !== "ready" || state.busy || video?.paused) return;
        const stream = state.config.decode === "stream";
        const hop = stream ? state.config.streamHopSec : state.config.minHopSec;
        let pcm: Float32Array<ArrayBuffer>;
        let job: Job;
        let newPcm: Float32Array<ArrayBuffer> | null = null;
        let reset = false;
        if (stream) {
            // Only the audio since the last hop goes to the worker, which
            // keeps the rest. A gap (seek, stall, long skip) restarts it.
            if (video && mediaActive()) {
                const ct = video.currentTime;
                // Right after a seek the first hop starts at the playhead and
                // stays short, so the first words show within seconds instead
                // of after the whole look-ahead.
                const end = Math.min(coveredEnd(ct), ct + (seekReset ? SEEK_FIRST_HOP_SEC : MAX_AHEAD_SEC));
                if (!(end > ct - 1)) return;
                let from = state.lastEndMedia;
                if (seekReset) {
                    // At the live edge the player holds about a second
                    // ahead, so the first hop reaches back into audio just
                    // played instead of waiting for a full hop to buffer.
                    const start = Math.min(ct, end - SEEK_FIRST_HOP_SEC);
                    from = Math.max(start, media.chunks.find((c) => c.end > start)?.start ?? start);
                    if (end - from < 1) return;
                    reset = true;
                } else if (!(from > end - 8) || from > end) {
                    from = Math.max(end - 2, media.chunks.find((c) => c.end > end - 2)?.start ?? end - 2);
                    reset = true;
                }
                if (!reset && end - from < hop) return;
                state.lastEndMedia = end;
                pcm = readMedia(from, end);
                job = { source: "media", end, newSec: end - from, sec: end - from };
            } else {
                let n = ringWrite - state.lastEnd;
                if (ringWrite < SR || n < SR * hop) return;
                if (n > SR * 8) {
                    n = SR * 2;
                    reset = true;
                }
                state.lastEnd = ringWrite;
                pcm = readLast(n);
                job = { source: "realtime", capturedAt: lastSampleAt, newSec: n / SR, sec: n / SR };
            }
            if (seekReset) reset = true;
            seekReset = false;
            // No new caption text for CLEAR_AFTER_MS: the next words start
            // a new caption, so the worker drops its old context (as after a
            // seek) instead of re-sending it. A result still waiting for the
            // playhead may be that new text, so the check waits for it.
            if (state.textAt && performance.now() - state.textAt >= CLEAR_AFTER_MS && !state.due.length) {
                state.textAt = 0;
                reset = true;
            }
        } else if (video && mediaActive()) {
            const ct = video.currentTime;
            const end = Math.min(coveredEnd(ct), ct + MAX_AHEAD_SEC);
            if (!(end > ct - 1) || end - state.lastEndMedia < hop) return;
            const from = Math.max(end - WINDOW_SEC, media.chunks.find((c) => c.end > end - WINDOW_SEC)?.start ?? end - WINDOW_SEC);
            if (end - from < 1) return;
            const newSec = Math.min(end - from, end - state.lastEndMedia);
            if (state.lastEndMedia > -Infinity && newSec > hop + 1.2) stats.skippedHops++;
            state.lastEndMedia = end;
            pcm = readMedia(from, end);
            job = { source: "media", end, newSec, sec: end - from };
        } else {
            if (ringWrite < SR * 2) return;
            if (ringWrite - state.lastEnd < SR * hop) return;
            const n = Math.min(SR * WINDOW_SEC, ringWrite);
            const newSec = Math.min(n, ringWrite - state.lastEnd) / SR;
            stats.waitS.push(newSec);
            state.lastEnd = ringWrite;
            pcm = readLast(n);
            job = { source: "realtime", capturedAt: lastSampleAt, newSec, sec: n / SR };
        }
        stats.windows++;
        let sum = 0;
        let peak = 0;
        for (let i = 0; i < pcm.length; i++) {
            sum += pcm[i] * pcm[i];
            peak = Math.max(peak, Math.abs(pcm[i]));
        }
        const quiet = Math.sqrt(sum / pcm.length) < SILENCE_RMS;
        if (quiet) stats.silent++;
        if (quiet && !stream) return;
        stats.bySource[job.source]++;
        // Normalise so a quiet source (or player volume on the element
        // path) does not starve the model. The streaming path keeps a slow
        // peak so the level does not jump between hops of one block.
        state.peak = stream ? Math.max(peak, state.peak * 0.9) : peak;
        const gain = Math.min(20, 0.5 / Math.max(state.peak, 1e-4));
        for (let i = 0; i < pcm.length; i++) pcm[i] *= gain;
        if (!stream && job.newSec > 0) newPcm = pcm.slice(-Math.round(Math.min(job.newSec, job.sec) * SR));
        const id = ++state.runId;
        pending.set(id, job);
        state.busy = true;
        (async () => {
            if (!quiet) for (const stage of preprocess) pcm = await stage(pcm, SR);
            const common = { id, pcm, lang: "ko", pace: state.config.pace === "raf" && !document.hidden, vad: state.config.vad, engine: state.config.engine };
            if (stream && reset) streamEnded = true;
            if (stream) state.worker!.postMessage({ type: "stream", ...common, reset, quiet, maxTokens: Math.round(16 + 12 * job.newSec) }, [pcm.buffer]);
            else state.worker!.postMessage({ type: "run", ...common, newPcm, maxTokens: 96 }, [pcm.buffer]);
        })().catch((error) => {
            state.busy = false;
            pending.delete(id);
            sttError("preprocess failed", "preprocess", error);
        });
    }

    // ---- UI --------------------------------------------------------------
    const STYLE = `
        .cb-stt-overlay { position: absolute; left: 50%; transform: translateX(-50%); bottom: 64px; width: min(86%, 1100px); z-index: 30; pointer-events: none; display: flex; flex-direction: column; justify-content: flex-end; text-align: center; transition: bottom .2s; }
        .pzp-pc--controls .cb-stt-overlay { bottom: 96px; }
        .cb-stt-overlay[data-pos="top"], .pzp-pc--controls .cb-stt-overlay[data-pos="top"] { bottom: auto; top: 56px; justify-content: flex-start; }
        .cb-stt-overlay[hidden] { display: none; }
        .cb-stt-delay { align-self: center; margin-bottom: 3px; font: 500 calc(clamp(10px, .75vw, 13px) * var(--cb-stt-scale, 1))/1.3 system-ui, sans-serif; font-variant-numeric: tabular-nums; color: rgba(255,255,255,.85); background: rgba(0,0,0,var(--cb-stt-bg, .72)); padding: 0 .45em; border-radius: 3px; text-shadow: 0 0 2px #000, 0 1px 2px #000; }
        .cb-stt-delay[hidden] { display: none; }
        .cb-stt-lines { max-height: calc(2 * 1.45em); overflow: hidden; display: flex; flex-direction: column; justify-content: flex-end; font: 600 calc(clamp(16px, 2.1vw, 34px) * var(--cb-stt-scale, 1))/1.45 "Pretendard", "Noto Sans KR", system-ui, sans-serif; }
        .cb-stt-lines > div { white-space: nowrap; }
        .cb-stt-lines > .cb-stt-measure { position: absolute; left: 0; top: 0; visibility: hidden; }
        .cb-stt-lines span span { background: none; padding: 0; text-shadow: inherit; }
        .cb-stt-tent { opacity: .62; }
        .cb-stt-lines > div > span { background: rgba(0,0,0,var(--cb-stt-bg, .72)); color: #fff; padding: .05em .4em; border-radius: 4px; -webkit-box-decoration-break: clone; box-decoration-break: clone; text-shadow: 0 0 2px #000, 0 1px 2px #000; }
        .cb-stt-status { align-self: center; margin-top: 6px; font: 500 13px/1.4 system-ui, sans-serif; color: #fff; background: rgba(0,0,0,.72); padding: 3px 10px; border-radius: 999px; }
        .cb-stt-status:empty { display: none; }
    `;
    let overlay: HTMLDivElement | null = null;
    function ensureOverlay() {
        const host = document.querySelector(".pzp-pc");
        if (!host) return null;
        if (!document.getElementById("cb-stt-style")) {
            const style = document.createElement("style");
            style.id = "cb-stt-style";
            style.textContent = STYLE;
            document.head.append(style);
        }
        if (overlay?.parentElement === host) return overlay;
        overlay = document.createElement("div");
        overlay.className = "cb-stt-overlay";
        overlay.innerHTML = '<div class="cb-stt-delay" hidden></div><div class="cb-stt-lines"><div class="cb-stt-measure"><span></span></div></div><div class="cb-stt-status"></div>';
        host.append(overlay);
        render();
        return overlay;
    }
    function setStatus(text: string) {
        state.statusText = text;
        render();
    }
    // ---- subtitle lines ----------------------------------------------------
    // The text is laid out into lines as rendered (box width, size setting,
    // fullscreen). A line is complete once the next word would not fit on
    // it; from then on its text is frozen, whatever the decoder revises
    // later, and it leaves the box whole when a new line needs the room.
    // Only the last, open line follows the decoder. The frozen lines are a
    // committed prefix of the transcript: each new hypothesis is aligned
    // against the committed tail (afterCommitted) and only what follows it
    // is laid out, so a sliding decoder window never rewrites shown lines.
    const VISIBLE_LINES = 2;
    const lines: { frozen: string[]; open: Word[]; key: string; layout: string; misses: number; dom: string; epoch: number; base: number; recorded: number; carry: string } = { frozen: [], open: [], key: "", layout: "", misses: 0, dom: "", epoch: 0, base: 0, recorded: 0, carry: "" };
    stats.lines = lines;
    // History records each line as it freezes. `recorded` counts the
    // leading open words already recorded (a reflow reopens the tail of a
    // recorded line), which the next record skips.
    // `at` stamps a line that was on screen before a seek with the moment
    // it was shown: its media time, and how far the seek moved the playhead.
    type LineStamp = { media: number; shiftSec: number };
    function recordLine(words: string[], at?: LineStamp) {
        const text = words.slice(lines.recorded).join(" ");
        lines.recorded = Math.max(0, lines.recorded - words.length);
        captionLog.append(text, at ? at.media : (cbAudio.playerVideo()?.currentTime ?? null), at?.shiftSec ?? 0);
    }
    // A seek (`at`) leaves nothing to align against; a stale gap keeps
    // the committed tail in `carry` for commitPoint.
    function resetLines(at?: LineStamp) {
        // The confirmed part of the open line was the end of what was
        // shown; it never freezes, so it is recorded as it leaves.
        const confirmed = lines.open.filter((w) => !w.tent).map((w) => w.t);
        recordLine(confirmed, at);
        const done = [...lines.frozen.slice(-6), ...confirmed].join(" ");
        if (at) lines.carry = "";
        else if (done) lines.carry = done;
        lines.recorded = 0;
        lines.frozen = [];
        lines.open = [];
        lines.key = "";
        lines.base = 0;
    }
    // Returns whether any line is shown.
    function composeLines(fresh: number | boolean) {
        const box = overlay!.querySelector(".cb-stt-lines") as HTMLElement;
        if (!fresh || lines.epoch !== state.textEpoch) resetLines();
        lines.epoch = state.textEpoch;
        const conf = fresh ? state.text : "";
        const tent = fresh && state.config.decode === "stream" ? state.tent : "";
        const hyp = conf + tent;
        const meas = box.querySelector(".cb-stt-measure > span") as HTMLElement;
        const width = box.clientWidth;
        const layout = `${width}|${getComputedStyle(box).fontSize}`;
        const fits = (words: Word[]) => {
            meas.textContent = words.map((w) => w.t).join(" ");
            return meas.getBoundingClientRect().width <= width;
        };
        if (width > 0 && layout !== lines.layout && lines.frozen.length) {
            // Size, scale or fullscreen changed: reflow the recent frozen
            // lines; the last of them reopens and is matched again below.
            const words = lines.frozen.slice(-4).join(" ").split(/\s+/).filter(Boolean).map((t) => ({ t, tent: false }));
            const out: string[] = [];
            let cur: Word[] = [];
            for (const w of words) {
                if (cur.length && !fits([...cur, w])) (out.push(cur.map((x) => x.t).join(" ")), (cur = []));
                cur.push(w);
            }
            lines.frozen = [...lines.frozen.slice(0, -4), ...out];
            lines.recorded = cur.length;
            lines.key = "";
        }
        lines.layout = layout;
        const key = `${hyp}\u0000${layout}`;
        if (width > 0 && key !== lines.key) {
            lines.key = key;
            const p = commitPoint(lines.frozen, lines.carry, hyp);
            if (p < 0) lines.misses++;
            else {
                const words: Word[] = [];
                const re = /\S+/g;
                const rest = hyp.slice(p);
                for (let m; (m = re.exec(rest)); ) words.push({ t: m[0], tent: p + m.index >= conf.length });
                let cur: Word[] = [];
                for (const w of words) {
                    if (cur.length && !fits([...cur, w])) {
                        lines.frozen.push(cur.map((x) => x.t).join(" "));
                        recordLine(cur.map((x) => x.t));
                        cur = [];
                    }
                    cur.push(w);
                }
                lines.open = cur;
                if (lines.frozen.length > 40) {
                    lines.base += lines.frozen.length - 20;
                    lines.frozen = lines.frozen.slice(-20);
                }
            }
        }
        const openConf = lines.open.filter((w) => !w.tent).map((w) => w.t).join(" ");
        const openTent = lines.open.filter((w) => w.tent).map((w) => w.t).join(" ");
        const shown = lines.frozen.slice(-(VISIBLE_LINES - (lines.open.length ? 1 : 0)));
        const dom = JSON.stringify([shown, openConf, openTent]);
        if (dom !== lines.dom || box.children.length !== shown.length + (lines.open.length ? 1 : 0) + 1) {
            lines.dom = dom;
            for (const el of [...box.children]) if (!el.classList.contains("cb-stt-measure")) el.remove();
            for (const t of shown) {
                const div = document.createElement("div");
                div.append(document.createElement("span"));
                (div.firstChild as HTMLSpanElement).textContent = t;
                box.append(div);
            }
            if (lines.open.length) {
                const div = document.createElement("div");
                div.innerHTML = '<span><span class="cb-stt-conf"></span><span class="cb-stt-tent"></span></span>';
                (div.querySelector(".cb-stt-conf") as HTMLElement).textContent = openConf && openTent ? openConf + " " : openConf;
                (div.querySelector(".cb-stt-tent") as HTMLElement).textContent = openTent;
                box.append(div);
            }
        }
        return shown.length > 0 || lines.open.length > 0;
    }
    function render() {
        if (!overlay) return;
        overlay.hidden = !state.enabled;
        const fresh = state.textAt && performance.now() - state.textAt < CLEAR_AFTER_MS;
        applyLook();
        const shown = composeLines(fresh);
        const delay = overlay.querySelector(".cb-stt-delay") as HTMLElement;
        delay.hidden = !shown || !Number.isFinite(state.delayS);
        if (!delay.hidden) delay.textContent = `${state.delayS!.toFixed(1)}s`;
        (overlay.querySelector(".cb-stt-status") as HTMLElement).textContent = state.statusText;
    }

    const sttButton = registerPlayerButton({
        className: "cb-stt-button",
        label: "실시간 자막",
        icon: ICON,
        order: 20,
        // A split column the host has not given subtitles gets no button.
        when: () => (LIVE_RE.test(location.pathname) || VIDEO_RE.test(location.pathname)) && cbSplit.sttHere(),
        pressed: () => state.enabled,
        onClick: () => setEnabled(!state.enabled),
    });
    function ensureButton() {
        sttButton.refresh();
    }

    // ---- 자막 설정: 플레이어 설정 패널 안의 네이티브 항목 ----------------
    // A "자막" row in the player's own settings menu opens a sub-pane built
    // from the player's classes (home rows, pane header with back button,
    // checked pane items), so it reuses the player's styling, sizing and
    // fullscreen rules. The player rebuilds its menu, so tick() re-inserts.
    // The pane is shown by a data attribute on .pzp-pc, which the player's
    // class bindings leave alone; it only counts while the menu is open.
    const KEY_LOOK = "stt.look";
    const SIZES: [string, string, number][] = [["s", "작게", 0.75], ["m", "보통", 1], ["l", "크게", 1.3], ["xl", "아주 크게", 1.65]];
    const BGS: [number, string][] = [[0, "없음"], [0.4, "반투명"], [0.72, "기본"], [1, "불투명"]];
    const POSITIONS: [string, string][] = [["bottom", "아래"], ["top", "위"]];
    const look: Look = { size: "m", bg: 0.72, pos: "bottom" };
    try {
        const saved = getSetting(KEY_LOOK, null) as Partial<Look> | null;
        if (saved && typeof saved === "object") {
            if (SIZES.some((s) => s[0] === saved.size)) look.size = saved.size!;
            if (BGS.some((b) => b[0] === saved.bg)) look.bg = saved.bg!;
            if (POSITIONS.some((p) => p[0] === saved.pos)) look.pos = saved.pos!;
        }
    } catch (error) {
        sttError("look load failed", "settings", error);
    }
    function applyLook() {
        if (!overlay) return;
        overlay.style.setProperty("--cb-stt-scale", String((SIZES.find((s) => s[0] === look.size) ?? SIZES[1])[2]));
        overlay.style.setProperty("--cb-stt-bg", String(look.bg));
        overlay.dataset.pos = look.pos;
    }
    function setLook<K extends keyof Look>(key: K, value: Look[K]) {
        look[key] = value;
        GM_setValue(KEY_LOOK, { ...look });
        applyLook();
        renderPanel();
    }
    const labelOf = (list: readonly (readonly [unknown, string, ...unknown[]])[], v: unknown) => (list.find((x) => x[0] === v) ?? list[0])[1];

    // Icons are copied from the player's own markup the first time it shows.
    const icons = { chevron: "", back: "", check: "", toggle: "" };
    function harvestIcons() {
        const grab = (key: keyof typeof icons, sel: string) => {
            if (icons[key]) return;
            const el = document.querySelector(sel);
            if (el) icons[key] = el.outerHTML;
        };
        grab("chevron", ".pzp-pc__settings .pzp-setting-intro-playbackrate .pzp-ui-setting-home-item__icon");
        grab("back", ".pzp-pc > [class*='-pane'] .pzp-ui-setting-pane-header__icon");
        grab("check", ".pzp-pc .pzp-ui-setting-pane-item__icon-wrapper");
        grab("toggle", ".pzp-pc__settings .pzp-ui-toggle");
    }

    const esc = (s: unknown) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
    type Page = [title: string, options: (readonly [string | number, string])[], current: () => string | number, set: (v: string) => void];
    const PAGES: Record<string, Page> = {
        size: ["글자 크기", SIZES.map(([v, l]) => [v, l]), () => look.size, (v) => setLook("size", v)],
        bg: ["배경", BGS.map(([v, l]) => [v, l]), () => look.bg, (v) => setLook("bg", Number(v))],
        pos: ["위치", POSITIONS, () => look.pos, (v) => setLook("pos", v)],
        model: ["모델", Object.keys(MODELS).map((k) => [k, k === "1.7B" ? `${k} (기본)` : k]), modelName, (v) => {
            GM_setValue(KEY_MODEL, v);
            if (state.enabled) startWorker();
            renderPanel();
        }],
    };
    const homeRow = (attrs: string, label: string, right: string) =>
        `<div role="menuitem" tabindex="0" class="pzp-ui-setting-home-item" ${attrs}><div class="pzp-ui-setting-home-item__top"><div class="pzp-ui-setting-home-item__left"><span class="pzp-ui-setting-home-item__label">${label}</span></div><div class="pzp-ui-setting-home-item__right">${right}</div></div></div>`;
    const valueRight = (value: string) => `<span class="pzp-ui-setting-home-item__value">${esc(value)}</span>${icons.chevron}`;

    function paneHtml(page: string) {
        const back = (title: string) =>
            `<div tabindex="0" role="button" class="pzp-ui-setting-pane-header pzp-setting-playbackrate-pane__header cb-sub-back" aria-label="backward"><div class="pzp-ui-setting-pane-header__container">${icons.back} <strong class="pzp-ui-setting-pane-header__title">${title}</strong></div><div class="pzp-ui-setting-pane-header__option"></div></div>`;
        if (page === "main") {
            const toggle = icons.toggle.replace(/<input /, `<input ${state.enabled ? "checked " : ""}`);
            return (
                back("자막") +
                '<div class="pzp-setting-playbackrate-pane__list-container">' +
                homeRow('data-cb-act="toggle" role="menuitemcheckbox" aria-checked="' + state.enabled + '"', "실시간 자막", toggle) +
                Object.entries(PAGES)
                    .map(([key, [title, options, current]]) => homeRow(`data-cb-page="${key}" expandable="true"`, title, valueRight(labelOf(options, current()))))
                    .join("") +
                homeRow('data-cb-act="history" role="menuitem"', "자막 기록", "") +
                homeRow('data-cb-act="clear-cache" role="menuitem"', `캐시 삭제 (${cacheMB ?? "…"} MB)`, "") +
                "</div>"
            );
        }
        const [title, options, current] = PAGES[page];
        return (
            back(title) +
            '<div class="pzp-setting-playbackrate-pane__list-container"><ul role="menu" class="pzp-setting-playbackrate-pane__list">' +
            options
                .map(([v, l]) => `<li role="menuitemradio" tabindex="0" aria-checked="${v === current()}" data-cb-value="${esc(v)}" class="pzp-ui-setting-pane-item${v === current() ? " pzp-ui-setting-pane-item--checked" : ""}">${icons.check}<div class="pzp-ui-setting-pane-item__slot"><span class="pzp-ui-setting-pane-item__value"><span>${esc(l)}</span></span></div></li>`)
                .join("") +
            "</ul></div>"
        );
    }

    let panelPage = "main";
    // Sized from headers and completion records; reading 2 GB of bodies just
    // to size them is not worth it.
    let cacheMB: string | null = null;
    async function refreshCacheSize() {
        let bytes = 0;
        try {
            const cache = await caches.open(CACHE_NAME);
            // Streamed entries carry no content-length; their size lives in the
            // worker's completion record (doneKey), so prefer that.
            const DONE = "https://chzzkbest.invalid/complete?";
            const keys = (await cache.keys()).map((r) => r.url);
            const done = new Set(keys.filter((u) => u.startsWith(DONE)).map((u) => decodeURIComponent(u.slice(DONE.length))));
            for (const url of keys) {
                const res = await cache.match(url);
                if (url.startsWith(DONE)) bytes += Number(await res!.text()) || 0;
                else if (!done.has(url)) bytes += Number(res?.headers.get("content-length") || 0);
            }
        } catch (error) {
            sttError("cache size failed", "settings", error);
        }
        cacheMB = mb(bytes);
        renderPanel();
    }
    function openPage(host: HTMLElement, page: string) {
        if (page === "main") refreshCacheSize();
        panelPage = page;
        host.dataset.cbSub = page;
        renderPanel();
        host.querySelector<HTMLElement>(".cb-sub-pane [tabindex]")?.focus({ preventScroll: true });
    }
    function onPaneAction(event: Event) {
        const host = document.querySelector<HTMLElement>(".pzp-pc");
        if (!host) return;
        const back = (event.target as Element).closest(".cb-sub-back");
        const row = (event.target as Element).closest<HTMLElement>("[data-cb-act], [data-cb-page], [data-cb-value]");
        if (back) {
            if (panelPage === "main") delete host.dataset.cbSub;
            else openPage(host, "main");
        } else if (row?.dataset.cbAct === "toggle") {
            setEnabled(!state.enabled);
        } else if (row?.dataset.cbAct === "history") {
            captionLog.open();
        } else if (row?.dataset.cbAct === "clear-cache") {
            caches.delete(CACHE_NAME).then(refreshCacheSize, (error) => sttError("cache delete failed", "settings", error));
            setStatus(state.enabled ? state.statusText : "");
        } else if (row?.dataset.cbPage) {
            openPage(host, row.dataset.cbPage);
        } else if (row?.dataset.cbValue !== undefined) {
            PAGES[panelPage][3](row!.dataset.cbValue!);
        }
    }
    const PANEL_STYLE = `
        .pzp-pc > .cb-sub-pane.pzp-pc__setting-playbackrate-pane { display: none; }
        .pzp-pc.pzp-pc--setting[data-cb-sub] > .cb-sub-pane.pzp-pc__setting-playbackrate-pane { display: block; }
        .pzp-pc.pzp-pc--setting[data-cb-sub] > .pzp-pc__settings { display: none; }
        .cb-sub-pane .pzp-ui-setting-home-item { cursor: pointer; }
    `;
    function ensurePanel() {
        const host = document.querySelector<PlayerHost>(".pzp-pc");
        const menu = host?.querySelector(":scope > .pzp-pc__settings");
        if (!menu || !host) return;
        harvestIcons();
        if (!document.getElementById("cb-sub-style")) {
            const style = document.createElement("style");
            style.id = "cb-sub-style";
            style.textContent = PANEL_STYLE;
            document.head.append(style);
        }
        if (!host.classList.contains("pzp-pc--setting") && host.dataset.cbSub) delete host.dataset.cbSub;
        if (!host.cbSubWatched) {
            host.cbSubWatched = true;
            new MutationObserver(() => {
                if (!host.classList.contains("pzp-pc--setting") && host.dataset.cbSub) delete host.dataset.cbSub;
            }).observe(host, { attributes: true, attributeFilter: ["class"] });
        }
        if (!menu.querySelector(".cb-sub-intro")) {
            const holder = document.createElement("div");
            holder.innerHTML = homeRow('expandable="true" label="자막"', "자막", '<span class="pzp-ui-setting-home-item__value cb-sub-summary"></span>' + icons.chevron);
            const row = holder.firstElementChild as HTMLElement;
            row.classList.add("cb-sub-intro");
            const open = (event: Event) => {
                event.stopPropagation();
                openPage(host, "main");
            };
            row.addEventListener("click", open);
            row.addEventListener("keydown", (event) => (event.key === "Enter" || event.key === " ") && open(event));
            const anchor = menu.querySelector(".pzp-setting-intro-playbackrate") ?? menu.querySelector(".pzp-setting-intro-quality");
            anchor ? anchor.after(row) : menu.append(row);
        }
        if (!host.querySelector(":scope > .cb-sub-pane")) {
            const pane = document.createElement("div");
            pane.className = "pzp-setting-playbackrate-pane pzp-pc-setting-playbackrate-pane pzp-pc__setting-playbackrate-pane cb-sub-pane";
            pane.setAttribute("role", "dialog");
            pane.setAttribute("aria-label", "자막 설정");
            // The player reads clicks and arrow keys under it as its own
            // commands (play toggle, seek, closing the menu).
            for (const type of ["pointerdown", "mousedown", "mouseup", "dblclick", "wheel"]) pane.addEventListener(type, (event) => event.stopPropagation());
            pane.addEventListener("click", (event) => {
                event.stopPropagation();
                onPaneAction(event);
            });
            pane.addEventListener("keydown", (event) => {
                event.stopPropagation();
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onPaneAction(event);
                } else if (event.key === "Escape" || event.key === "Backspace") {
                    if (panelPage === "main") delete host.dataset.cbSub;
                    else openPage(host, "main");
                }
            });
            host.append(pane);
            refreshCacheSize();
        }
        renderPanel();
    }
    let panelKey = "";
    function renderPanel() {
        const host = document.querySelector(".pzp-pc");
        const summary = host?.querySelector(".cb-sub-summary");
        if (summary) summary.textContent = state.enabled ? labelOf(SIZES, look.size) : "끔";
        const pane = host?.querySelector(":scope > .cb-sub-pane");
        if (!pane) return;
        const key = [panelPage, state.enabled, look.size, look.bg, look.pos, modelName(), cacheMB, icons.back.length].join("|");
        if (key === panelKey && pane.childElementCount) return;
        panelKey = key;
        pane.innerHTML = paneHtml(PAGES[panelPage] ? panelPage : "main");
    }

    async function setEnabled(on: boolean) {
        state.enabled = on;
        GM_setValue(KEY_ENABLED, on);
        ensureButton();
        ensureOverlay();
        render();
        renderPanel();
        if (!on) return untap();
        startWorker();
        if (!(await ensureAudio())) setStatus("화면을 한 번 클릭하면 자막이 시작돼요");
        retap();
    }

    // A split column the host has not given subtitles: no button, no
    // overlay, no hub connection (the worker drops this column's state).
    // Given them later, it picks up the saved on/off setting.
    let parked = !cbSplit.sttHere();
    function park() {
        document.querySelector(".pzp-pc .cb-stt-button")?.remove();
        overlay?.remove();
        overlay = null;
        if (state.enabled || state.worker) {
            state.enabled = false;
            untap();
            state.worker?.terminate();
            state.worker = null;
            state.phase = "off";
            state.busy = false;
            pending.clear();
            resetLines();
            setStatus("");
        }
        parked = true;
    }
    function unpark() {
        parked = false;
        if (!getSetting(KEY_ENABLED, false)) return;
        state.enabled = true;
        startWorker();
        ensureAudio().then((ok) => ok && state.enabled && retap());
    }

    const onPlayerPage = () => LIVE_RE.test(location.pathname) || VIDEO_RE.test(location.pathname);
    function tick() {
        try {
            if (!onPlayerPage()) return;
            // The SPA swaps pages in place; a new live/video path starts a new history.
            captionLog.setPath((LIVE_RE.exec(location.pathname) ?? VIDEO_RE.exec(location.pathname))![0]);
            if (!cbSplit.sttHere()) return park();
            if (parked) unpark();
            ensureButton();
            ensureOverlay();
            ensurePanel();
            if (state.enabled) {
                if (mediaActive()) untap();
                else retap();
            }
            render();
        } catch (error) {
            sttError(`tick failed (${location.href})`, "tick", error);
        }
    }

    // A persisted "on" resumes after the first gesture on the page.
    if (getSetting(KEY_ENABLED, false) && cbSplit.sttHere()) {
        state.enabled = true;
        const resume = () => {
            if (!onPlayerPage() || !state.enabled) return;
            ensureAudio().then((ok) => {
                if (!ok) return;
                window.removeEventListener("pointerdown", resume, true);
                window.removeEventListener("keydown", resume, true);
                if (state.statusText.startsWith("화면을")) setStatus("");
                retap();
            });
            startWorker();
        };
        window.addEventListener("pointerdown", resume, true);
        window.addEventListener("keydown", resume, true);
        setTimeout(() => state.enabled && onPlayerPage() && !cbAudio.running() && setStatus("화면을 한 번 클릭하면 자막이 시작돼요"), 3000);
    }

    setInterval(schedule, 100);
    setInterval(() => {
        tick();
    }, 1000);
    let scheduled = false;
    new MutationObserver(() => {
        if (scheduled) return;
        scheduled = true;
        setTimeout(() => {
            scheduled = false;
            tick();
        }, 300);
    }).observe(document, { subtree: true, childList: true });
})();
