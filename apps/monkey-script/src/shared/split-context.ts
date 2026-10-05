// ---- 스플릿 뷰: where this copy of the script runs -----------------------
// The split page (/cbsplit) hosts up to three /live/<id> iframes, and the
// script runs again inside each. A column frame learns its place from the
// dataset of its <iframe> element (same origin): cbCol (channel id), cbPan
// and cbStt ("1" in the one column that may run subtitles). The host may
// rewrite cbPan and cbStt at any time; readers poll them.
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

export interface SplitContext {
    HOST_PATH: string;
    frame: boolean;
    col: string | null;
    isHost(): boolean;
    pan(): number | null;
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
        pan: () => (frame ? Number(el?.dataset.cbPan) || 0 : null),
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
