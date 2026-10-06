// ---- 스플릿 뷰: where this copy of the script runs -----------------------
// The split page (/cbsplit) hosts up to three /live/<id> iframes, and the
// script runs again inside each. A column frame learns its place from the
// dataset of its <iframe> element (same origin): cbCol (channel id), cbPan
// (the position's starting pan), cbPanMin/cbPanMax (the range the column's
// slider may move in) and cbStt ("1" in the one column that may run
// subtitles). The host may rewrite these at any time; readers poll them.
// Published by captions/index.ts on the top window; the captions module
// owns the full shape.
export interface SttHub<Reply = unknown, Request extends object = object> {
    drop(col: string): void;
    stats(): { workersCreated: number; alive: boolean; model: string; clients: (string | null)[] };
    connect(col: string | null, name: string, fn: (m: Reply) => void): { postMessage(m: Request, transfer?: Transferable[]): void; terminate(): void };
}
declare global {
    interface Window {
        ChzzkBestSttHub?: SttHub;
    }
}

export interface PanRange {
    min: number;
    max: number;
    def: number;
}

export interface SplitContext {
    HOST_PATH: string;
    frame: boolean;
    col: string | null;
    isHost(): boolean;
    panRange(): PanRange | null;
    sttHere(): boolean;
    hub(): SttHub | null;
}

export const cbSplit: SplitContext = (() => {
    "use strict";
    const HOST_PATH = "/cbsplit";
    let el: HTMLElement | null = null;
    try {
        el = window.frameElement as HTMLElement | null;
    } catch {
        el = null;
    }
    const frame = !!el?.dataset?.cbCol;
    return {
        HOST_PATH,
        frame,
        col: frame ? (el?.dataset.cbCol ?? null) : null,
        isHost: () => !frame && window.top === window && location.pathname === HOST_PATH,
        panRange: () => {
            if (!frame || !el) return null;
            const num = (v: string | undefined, fallback: number) => {
                const n = Number(v);
                return v !== undefined && v !== "" && Number.isFinite(n) ? n : fallback;
            };
            const min = num(el.dataset.cbPanMin, -1);
            const max = Math.max(min, num(el.dataset.cbPanMax, 1));
            return { min, max, def: Math.max(min, Math.min(max, num(el.dataset.cbPan, 0))) };
        },
        sttHere: () => !frame || el?.dataset.cbStt === "1",
        hub: () => {
            try {
                return window.top?.ChzzkBestSttHub ?? null;
            } catch {
                return null;
            }
        },
    };
})();
