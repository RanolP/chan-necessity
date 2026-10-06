// Messages between the page (index.ts, or the split-view hub) and the
// STT worker (worker/main.ts). Replies to a column's request carry its col.
import type { LogLevel } from "@logtape/logtape";

export interface EngineAssets {
    manifest: ArrayBuffer;
    qknorm: ArrayBuffer;
}

export interface TickMessage {
    type: "tick";
}
export interface DropMessage {
    type: "drop";
    col: string | null;
}
export interface LoadMessage {
    type: "load";
    repo: string;
    rev: string;
    cacheName: string;
    engineAssets: EngineAssets | null;
    logLevel?: LogLevel;
    id?: undefined;
    col?: string | null;
}
interface JobCommon {
    id: number;
    pcm: Float32Array;
    lang: string;
    pace: boolean;
    vad: boolean;
    engine: boolean;
    maxTokens: number;
    col?: string | null;
}
export interface StreamMessage extends JobCommon {
    type: "stream";
    reset: boolean;
    quiet: boolean;
    /** Prefill every hop from scratch instead of reusing the cached prefix KV: the reference the sliding cache is checked against. */
    exactKv?: boolean;
}
export interface RunMessage extends JobCommon {
    type: "run";
    newPcm: Float32Array | null;
}
export type PageMessage = TickMessage | DropMessage | LoadMessage | StreamMessage | RunMessage;

export interface PaceStats {
    k: number;
    frameMs: number;
    frames: number;
}

export interface CapsMessage {
    type: "caps";
    workerRaf: boolean;
    col?: string;
}
export interface ProgressMessage {
    type: "progress";
    loaded: number;
    total: number;
    fromNet: number;
    col?: string;
}
export interface StatusMessage {
    type: "status";
    text: string;
    col?: string;
}
export interface ReadyMessage {
    type: "ready";
    adapter: string;
    downloadMs: number;
    // Weight upload and pipeline creation for the encoder and the decoder.
    buildMs: number;
    fromNet: number;
    total: number;
    col?: string;
}
// One shape for both decode modes: a streaming hop sets `stream`, a
// window run sets `text`; the fields of the other mode stay absent.
export interface ResultMessage {
    type: "result";
    id: number;
    speechSec: number | null;
    vadMs: number | null;
    speech: boolean;
    totalMs: number;
    aborted?: boolean;
    tokens?: number;
    prefillMs?: number;
    pace?: PaceStats | null;
    decodeMs?: number;
    skipped?: "vad";
    text?: string;
    melMs?: number;
    encPrefillMs?: number;
    stream?: boolean;
    decoded?: boolean;
    hist?: string;
    conf?: string | null;
    tent?: string | null;
    ids?: number[];
    final?: boolean;
    grew?: boolean;
    slidN?: number;
    melEncMs?: number;
    encMs?: number;
    prefill?: number;
    /** Prompt rows in the KV cache after this hop's prefill, of which `cached` were reused from earlier hops. */
    ctx?: number;
    cached?: number;
    audioTokens?: number;
    /** Closed blocks evicted from the cached prefix this hop (engine.shiftKV), and the time the shifts took. */
    slides?: number;
    shiftMs?: number;
    /** Set on a quiet hop that recomputed the prefix whole. */
    rebuildMs?: number;
    rebuildTokens?: number;
    col?: string;
}
export interface ErrorMessage {
    type: "error";
    stage: string;
    name: string;
    message: string;
    stack?: string;
    id?: number;
    col?: string;
}
export type WorkerMessage = CapsMessage | ProgressMessage | StatusMessage | ReadyMessage | ResultMessage | ErrorMessage;
