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
export interface GateMessage {
    type: "gate";
    open: boolean;
    col?: string | null;
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
    gapMs: number;
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
}
export interface RunMessage extends JobCommon {
    type: "run";
    newPcm: Float32Array | null;
}
export type PageMessage = TickMessage | GateMessage | DropMessage | LoadMessage | StreamMessage | RunMessage;

export interface PaceStats {
    k: number;
    frameMs: number;
    frames: number;
}
export type EngineInfo = { on: true; loadMs: number } | { on: false; reason: string };

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
    f16: boolean;
    engine: EngineInfo;
    downloadMs: number;
    sessionMs: number;
    fromNet: number;
    total: number;
    warmed: number;
    warmMs: number;
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
    via?: "engine" | "ort";
    engineMs?: number;
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
    audioTokens?: number;
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
