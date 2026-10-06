// Timestamped history of finalized subtitle lines, shown in a popup window
// the opener builds and keeps up to date (no URL, no Document PiP, so it
// works in Firefox too). The history lives in memory for one live/video
// page; the caller resets it when the page changes.
import type { Logger } from "@logtape/logtape";

export interface CaptionEntry {
    // On live pages, the broadcast moment being watched (wall-clock ms).
    wall: number;
    media: number | null;
    // The broadcast's open time (ms); null on VOD pages or until known.
    open: number | null;
    text: string;
}

const pad = (n: number) => String(n).padStart(2, "0");
const duration = (sec: number) => {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    return h ? `${h}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
};
// <pure> Live pages show the time since the broadcast opened, or the local
// wall clock (HH:MM:SS) of the watched moment while the open time is
// unknown; VOD pages show the media time. Durations read H:MM:SS, or M:SS
// under an hour.
export function formatStamp(entry: CaptionEntry, live: boolean): string {
    if (live) {
        if (entry.open !== null) return duration((entry.wall - entry.open) / 1000);
        const d = new Date(entry.wall);
        return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }
    if (entry.media === null || !Number.isFinite(entry.media)) return "--:--";
    return duration(entry.media);
}

const POPUP_NAME = "chan-necessity-captions";
const POPUP_STYLE = `
    :root { color-scheme: light dark; --fg: #1a1a1a; --bg: #fff; --dim: #777; --line: #e6e6e6; }
    @media (prefers-color-scheme: dark) { :root { --fg: #e8e8e8; --bg: #141517; --dim: #8b8f96; --line: #26282c; } }
    html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, sans-serif; }
    #cb-hist-list { height: 100%; overflow-y: auto; box-sizing: border-box; padding: 8px 12px; }
    .cb-hist-row { display: flex; gap: 10px; padding: 3px 0; border-bottom: 1px solid var(--line); }
    .cb-hist-time { flex: none; color: var(--dim); font-variant-numeric: tabular-nums; }
    .cb-hist-text { overflow-wrap: anywhere; }
    .cb-hist-empty { color: var(--dim); }
`;
// The list counts as followed when its bottom is this close to the view.
const STICK_PX = 40;

export function createCaptionHistory(opts: {
    cap: number;
    live: () => boolean;
    logger: Logger;
    // Live pages only: the open time (ms) of the page's broadcast, null when
    // it cannot be known; asked once per page and never awaited by append.
    broadcastOpen: (path: string) => Promise<number | null>;
    // Live pages only: the broadcast moment being watched (wall-clock ms).
    watched: (open: number | null) => number;
}) {
    const entries: CaptionEntry[] = [];
    let popup: Window | null = null;
    let path = "";
    let open: number | null = null;

    // A popup navigated elsewhere turns cross-origin and throws on access.
    const list = () => {
        try {
            return popup && !popup.closed ? popup.document.getElementById("cb-hist-list") : null;
        } catch {
            return null;
        }
    };
    function row(doc: Document, entry: CaptionEntry, live: boolean) {
        const div = doc.createElement("div");
        div.className = "cb-hist-row";
        const time = doc.createElement("span");
        time.className = "cb-hist-time";
        time.textContent = formatStamp(entry, live);
        const text = doc.createElement("span");
        text.className = "cb-hist-text";
        text.textContent = entry.text;
        div.append(time, text);
        return div;
    }
    function fill(box: HTMLElement) {
        const doc = box.ownerDocument;
        const live = opts.live();
        if (!entries.length) {
            const empty = doc.createElement("div");
            empty.className = "cb-hist-empty";
            empty.textContent = "아직 기록된 자막이 없어요";
            box.replaceChildren(empty);
        } else box.replaceChildren(...entries.map((e) => row(doc, e, live)));
        box.scrollTop = box.scrollHeight;
    }

    return {
        // Clears the history when the player page changed.
        setPath(next: string) {
            if (next === path) return;
            path = next;
            entries.length = 0;
            open = null;
            const box = list();
            if (box) fill(box);
            if (!opts.live()) return;
            // Lines recorded before the open time arrived get it now, and
            // their stamps switch from the wall clock to elapsed time.
            opts.broadcastOpen(next).then((ms) => {
                if (ms === null || path !== next) return;
                open = ms;
                for (const e of entries) e.open ??= ms;
                const box = list();
                box?.querySelectorAll(".cb-hist-time").forEach((el, i) => {
                    if (entries[i]) el.textContent = formatStamp(entries[i], opts.live());
                });
            });
        },
        // `shiftSec`: how far the playhead has moved since the line was
        // shown (a seek that ended it); the live stamp is moved back by it.
        append(text: string, media: number | null, shiftSec = 0) {
            if (!text) return;
            const live = opts.live();
            const entry = { wall: live ? opts.watched(open) - shiftSec * 1000 : Date.now(), media, open: live ? open : null, text };
            entries.push(entry);
            if (entries.length > opts.cap) entries.splice(0, entries.length - opts.cap);
            const box = list();
            if (!box) return;
            const follow = box.scrollHeight - box.scrollTop - box.clientHeight <= STICK_PX;
            if (box.querySelector(".cb-hist-empty")) box.replaceChildren();
            box.append(row(box.ownerDocument, entry, live));
            while (box.childElementCount > opts.cap) box.firstElementChild!.remove();
            if (follow) box.scrollTop = box.scrollHeight;
        },
        // An existing popup (also one left over from before a reload) is
        // reused and rebuilt from this page's history.
        open() {
            const win = window.open("", POPUP_NAME, "popup,width=420,height=640");
            if (!win) {
                opts.logger.warn("history popup blocked {path} {entries}", { path, entries: entries.length });
                return;
            }
            popup = win;
            try {
                const doc = win.document;
                doc.title = "자막 기록";
                let style = doc.getElementById("cb-hist-style");
                if (!style) {
                    style = doc.createElement("style");
                    style.id = "cb-hist-style";
                    style.textContent = POPUP_STYLE;
                    doc.head.append(style);
                }
                let box = doc.getElementById("cb-hist-list");
                if (!box) {
                    box = doc.createElement("div");
                    box.id = "cb-hist-list";
                    doc.body.replaceChildren(box);
                }
                fill(box);
            } catch (error) {
                opts.logger.error("history popup build failed {path} {error}", { path, error });
            }
            win.focus();
        },
    };
}
