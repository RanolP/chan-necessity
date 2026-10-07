import { getLogger } from "../shared/logtape.ts";
import { cbSplit } from "../shared/split-context.ts";

type Pos = "L" | "C" | "R";
interface MenuAction {
    label: string;
    cols: string[];
}
interface SplitStats {
    cols: string[];
    frames: { id: string; pos: string | undefined; live: boolean; pan: string | undefined; stt: string | undefined }[];
}
interface SplitApi {
    gesture(col: string | null): void;
    navigated(col: string, href: string): void;
    stats(): SplitStats;
    set(cols: string[]): void;
    menuActions(cols: string[], id: string): MenuAction[];
    loadLayout(): string[];
}
declare global {
    interface Window {
        ChzzkBestSplit?: SplitApi;
    }
}
interface StoredLayout {
    cols?: unknown;
}
interface ChannelContent {
    channelId?: string;
    channelName: string;
    channelImageUrl: string | null;
    openLive: boolean;
}
interface ChannelInfo {
    id: string;
    exists: boolean;
    name: string;
    image: string | null;
    live: boolean | null;
}
interface LiveNode {
    channel?: LiveNode;
    channelId?: unknown;
    channelName?: string;
    channelImageUrl?: string | null;
    liveInfo?: LiveNode;
    liveTitle?: string;
    concurrentUserCount?: number;
    openLive?: boolean;
    streamer?: { openLive?: boolean };
}
interface LiveItem {
    id: string;
    name: string;
    image: string | null | undefined;
    title: string;
    viewers: number | null;
    open: boolean;
}
interface Column {
    id: string;
    el: HTMLElement;
    body: HTMLDivElement;
    frame: HTMLIFrameElement | null;
    poll: number;
    pos: HTMLSpanElement;
    avatar: HTMLImageElement;
    name: HTMLAnchorElement;
    live: HTMLSpanElement;
    stt: HTMLSpanElement;
    close: HTMLButtonElement;
}
interface OpenMenu {
    el: HTMLDivElement;
    btn: HTMLButtonElement;
    outside: (event: PointerEvent) => void;
    key: (event: KeyboardEvent) => unknown;
}

const logger = getLogger(["split-view"]);

// ---- 스플릿 뷰: 최대 세 채널을 칸으로 나눠 보기 ---------------------------
// A top-level live page becomes the split page: Chzzk boots its own shell at
// /cbsplit (see the boot below), its header and sidebar stay, and a grid
// over #layout-body holds one same-origin /live/<id> iframe per column. The
// address names the columns: /live/<id1>?cb-split=<id2>,<id3>. The script
// runs again inside each frame (see cbSplit at the top): the frame restyles
// itself to the player and its chat, takes its pan range from its position
// (L -1..0 starting at -1, C -0.5..0.5 starting at 0, R 0..1 starting at
// +1; the column's slider moves freely inside it until the position
// changes), and only the leftmost column may run subtitles, through the top page's one model.
// Columns are reordered with CSS order, never by moving nodes, because a
// moved iframe reloads.
(() => {
    "use strict";
    const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;
    const LIVE_RE = /^\/live\/([0-9a-f]{32})/i;
    const ID_RE = /([0-9a-f]{32})/i;
    const KEY_LAYOUT = "split.layout";
    const MAX = 3;
    const FRAME_WIDE = 900;
    const SPLIT_PARAM = "cb-split";
    const POS: Record<number, Pos[]> = { 1: ["C"], 2: ["L", "R"], 3: ["L", "C", "R"] };
    const PAN = { L: { min: -1, max: 0, def: -1 }, C: { min: -0.5, max: 0.5, def: 0 }, R: { min: 0, max: 1, def: 1 } };
    function writePan(frame: HTMLIFrameElement, p: Pos) {
        const { min, max, def } = PAN[p];
        frame.dataset.cbPan = String(def);
        frame.dataset.cbPanMin = String(min);
        frame.dataset.cbPanMax = String(max);
    }
    const positions = (n: number) => POS[n] ?? [];
    const currentId = () => location.pathname.match(LIVE_RE)?.[1]?.toLowerCase() ?? null;
    const ICON =
        '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M9 5v14M15 5v14"/></svg>';

    function addStyle(css: string) {
        const style = document.createElement("style");
        style.textContent = css;
        if (document.head) return document.head.append(style);
        // At document-start <head> may not exist yet. A style parked on <html>
        // meanwhile gets dropped once Chzzk's page loads, so wait for <head>.
        new MutationObserver((_, observer) => {
            if (!document.head) return;
            observer.disconnect();
            document.head.append(style);
        }).observe(document, { childList: true, subtree: true });
    }

    // ---- inside a column frame -----------------------------------------
    if (cbSplit.frame) {
        // Chzzk keeps its 950px desktop layout at any width, so the frame
        // restyles it: no header or sidebar and the player alone from
        // <main>. A narrow frame puts the chat panel under the player; a wide
        // one keeps Chzzk's chat column (353px) beside it, which from
        // FRAME_WIDE on still leaves the player over 540px wide.
        addStyle(`
            html, body { min-width: 0 !important; overflow: hidden !important; }
            #header, #sidebar { display: none !important; }
            div:has(> #layout-body) { min-width: 0 !important; }
            #layout-body { min-width: 0 !important; height: 100vh !important; padding: 0 !important; margin: 0 !important; }
            #layout-body > section, #layout-body > section > div {
                flex-direction: column !important; width: 100% !important; height: 100% !important;
                min-width: 0 !important; margin: 0 !important; padding: 0 !important;
            }
            #layout-body > section > div > main {
                flex: none !important; width: 100% !important; height: auto !important; min-height: 0 !important;
                max-width: none !important; min-width: 0 !important; overflow: hidden !important; padding: 0 !important; margin: 0 !important;
            }
            #layout-body main > :not(:first-child), #layout-body main > :first-child > :not(:first-child) { display: none !important; }
            #layout-body main > :first-child, #layout-body main > :first-child > :first-child { width: 100% !important; max-width: none !important; }
            /* Chzzk's wide view mode makes the player box 100vh tall: the video letterboxes inside it while the
               controls and captions anchor to the box, off the picture. The 16:9 box keeps them on it. */
            #layout-body main > :first-child > :first-child { height: auto !important; }
            #aside-chatting {
                position: relative !important; flex: 1 1 auto !important; width: 100% !important; max-width: none !important;
                height: auto !important; min-height: 0 !important; top: auto !important; right: auto !important; border-left: 0 !important;
            }
            @media (min-width: ${FRAME_WIDE}px) {
                #layout-body > section > div { flex-direction: row !important; }
                #layout-body > section > div > main {
                    flex: 1 1 0 !important; height: 100% !important; display: flex !important; flex-direction: column !important;
                    justify-content: center !important; background: #000 !important;
                }
                #layout-body main > :first-child { width: min(100%, calc(100vh * 16 / 9)) !important; margin: 0 auto !important; }
                #aside-chatting { flex: none !important; width: 353px !important; height: 100% !important; }
            }
        `);
        // Chzzk links inside the frame (a channel name in chat, a raid) move
        // it off its column; the host turns that into a column change or a
        // top-level navigation.
        const own = `/live/${cbSplit.col}`;
        let reported = "";
        setInterval(() => {
            if (location.pathname.toLowerCase().startsWith(own) || location.href === reported) return;
            reported = location.href;
            try {
                window.top!.ChzzkBestSplit?.navigated(cbSplit.col!, location.href);
            } catch (error) {
                logger.error`navigation report failed ${cbSplit.col} ${location.href} ${error}`;
            }
        }, 500);
        // A click in one frame activates only that frame and the top page,
        // not its sibling columns; the top page re-dispatches it so every
        // column's audio context can start from the same gesture.
        const relay = (event: Event) => {
            if (!event.isTrusted) return;
            try {
                window.top!.ChzzkBestSplit?.gesture(cbSplit.col);
            } catch (error) {
                logger.error`gesture relay failed ${cbSplit.col} ${error}`;
            }
        };
        window.addEventListener("pointerdown", relay, true);
        window.addEventListener("keydown", relay, true);
        return;
    }
    if (window.top !== window) return;

    function colsFromUrl(url: URL | Location): string[] {
        if (url.pathname === cbSplit.HOST_PATH) return loadLayout();
        const first = url.pathname.match(LIVE_RE)?.[1];
        if (!first) return [];
        const rest = new URLSearchParams(url.search).get(SPLIT_PARAM)?.split(",") ?? [];
        return [...new Set([first, ...rest].map((id) => id.trim().toLowerCase()).filter((id) => /^[0-9a-f]{32}$/.test(id)))].slice(0, MAX);
    }
    const splitUrl = (cols: string[]) => (cols.length ? `/live/${cols[0]}${cols.length > 1 ? `?${SPLIT_PARAM}=${cols.slice(1).join(",")}` : ""}` : cbSplit.HOST_PATH);
    const here = () => location.pathname + location.search;

    // ---- booting a live page as the split host ---------------------------
    // Chzzk's router reads the address once at boot, then only on its own
    // navigations and popstate. Booted at /cbsplit it renders just the
    // header and sidebar; once #layout-body exists the real address comes
    // back through replaceState, keeping the router's history state.
    const rawPush = history.pushState.bind(history);
    const rawReplace = history.replaceState.bind(history);
    const boot = { active: cbSplit.isHost(), ready: false };
    if (boot.active) {
        const url = location.pathname === cbSplit.HOST_PATH ? splitUrl(loadLayout()) : location.pathname + location.search + location.hash;
        rawReplace(history.state, "", cbSplit.HOST_PATH);
        const restore = () => {
            if (!document.getElementById("layout-body")) return false;
            rawReplace(history.state, "", url);
            boot.ready = true;
            logger.info`booted the shell for ${url}`;
            return true;
        };
        new MutationObserver((_, observer) => restore() && observer.disconnect()).observe(document, { childList: true, subtree: true });
    }
    // A live page the router reaches by itself would render a player in
    // the top page, so it loads afresh and boots as a host instead. Not
    // while a boot is still pending: a boot that never lands cannot loop.
    for (const [name, raw, go] of [
        ["pushState", rawPush, (href: string) => location.assign(href)],
        ["replaceState", rawReplace, (href: string) => location.replace(href)],
    ] as const) {
        history[name] = (state: unknown, unused: string, url?: string | URL | null) => {
            if (url != null && (!boot.active || boot.ready)) {
                const next = new URL(url, location.href);
                // The router still believes it sits at /cbsplit.
                if (boot.ready && next.pathname === cbSplit.HOST_PATH) return raw(state, unused, here());
                if (next.origin === location.origin && LIVE_RE.test(next.pathname) && next.pathname + next.search !== here()) {
                    logger.info`router ${name} ${next.href}: reloading as a split host`;
                    return go(next.href);
                }
            }
            return raw(state, unused, url);
        };
    }
    // Back and forward between split addresses change columns in place; the
    // router would render a player page for them.
    window.addEventListener(
        "popstate",
        (event) => {
            if (!boot.ready || !cbSplit.isHost()) return;
            event.stopImmediatePropagation();
            if (host.mounted()) host.set(colsFromUrl(location), "none");
            else location.reload();
        },
        true,
    );
    // A live link in the shell (sidebar, search, header) opens as columns
    // without leaving the page; modified clicks and new tabs stay native.
    window.addEventListener(
        "click",
        (event) => {
            if (!host.mounted() || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
            const a = (event.target as Element | null)?.closest?.<HTMLAnchorElement>("a[href]");
            if (!a || (a.target && a.target !== "_self")) return;
            const url = new URL(a.href);
            if (url.origin !== location.origin || !LIVE_RE.test(url.pathname)) return;
            event.preventDefault();
            event.stopPropagation();
            host.set(colsFromUrl(url));
        },
        true,
    );

    function loadLayout(): string[] {
        try {
            const saved = GM_getValue<StoredLayout | null>(KEY_LAYOUT, null);
            const cols = Array.isArray(saved?.cols) ? (saved!.cols as unknown[]).filter((id): id is string => typeof id === "string" && /^[0-9a-f]{32}$/.test(id)) : [];
            return [...new Set(cols)].slice(0, MAX);
        } catch (error) {
            logger.error`layout load failed ${error}`;
            return [];
        }
    }
    const saveLayout = (cols: string[]) => GM_setValue(KEY_LAYOUT, { cols: cols.slice(0, MAX) });

    async function getJson(url: string): Promise<unknown> {
        const response = await fetch(url, { credentials: "include" });
        if (!response.ok) {
            const error: Error & { status?: number } = new Error(`HTTP ${response.status} ${url} ${(await response.text().catch(() => "")).slice(0, 200)}`);
            error.status = response.status;
            throw error;
        }
        return response.json();
    }
    const infoCache = new Map<string, { at: number; promise: Promise<ChannelInfo> }>(); // id -> { at, promise }
    function channelInfo(id: string, maxAge = 60000): Promise<ChannelInfo> {
        const hit = infoCache.get(id);
        if (hit && Date.now() - hit.at < maxAge) return hit.promise;
        const promise = getJson(`https://api.chzzk.naver.com/service/v1/channels/${id}`)
            .then((json): ChannelInfo => {
                const c = (json as { content?: ChannelContent } | null)?.content ?? ({} as ChannelContent);
                if (!c.channelId) return { id, exists: false, name: "알 수 없는 채널", image: null, live: false };
                return { id, exists: true, name: c.channelName, image: c.channelImageUrl, live: !!c.openLive };
            })
            .catch((error: unknown) => {
                logger.error`channel info failed ${id} ${error}`;
                infoCache.delete(id);
                return { id, exists: true, name: id.slice(0, 8), image: null, live: null };
            });
        infoCache.set(id, { at: Date.now(), promise });
        return promise;
    }
    // Live lists come in a few shapes (lives, followings); take every
    // object that names a channel, with the live fields beside it.
    function liveItems(json: unknown): LiveItem[] {
        const out: LiveItem[] = [];
        const seen = new Set<string>();
        const walk = (node: unknown): void => {
            if (!node || typeof node !== "object") return;
            if (Array.isArray(node)) return node.forEach(walk);
            const n = node as LiveNode;
            const ch = n.channel ?? n;
            const id = ch?.channelId;
            if (typeof id === "string" && /^[0-9a-f]{32}$/i.test(id) && ch.channelName) {
                if (seen.has(id)) return;
                seen.add(id);
                const live = n.liveInfo ?? n;
                const open = n.streamer?.openLive ?? live.openLive ?? ch.openLive;
                out.push({ id: id.toLowerCase(), name: ch.channelName, image: ch.channelImageUrl, title: live.liveTitle ?? "", viewers: live.concurrentUserCount ?? null, open: open !== false });
                return;
            }
            for (const value of Object.values(node)) walk(value);
        };
        walk((json as { content?: unknown } | null)?.content ?? json);
        return out;
    }

    // <pure> the sidebar menu: what each action does to the column list.
    // Positions: 1 column C; 2 columns L R; 3 columns L C R.
    function swapInto(cols: string[], i: number, id: string) {
        const next = [...cols];
        const j = next.indexOf(id);
        if (j >= 0) [next[i], next[j]] = [next[j], next[i]];
        else next[i] = id;
        return next;
    }
    function menuActions(cols: string[], id: string): MenuAction[] {
        const out: MenuAction[] = [];
        const at = cols.indexOf(id);
        positions(cols.length).forEach((p, i) => {
            if (i !== at) out.push({ label: `swap with ${p}`, cols: swapInto(cols, i, id) });
        });
        if (at < 0 && cols.length < MAX) {
            const adds: [string, number][] = cols.length === 0 ? [["center", 0]] : cols.length === 1 ? [["left", 0], ["right", 1]] : [["left", 0], ["center", 1], ["right", 2]];
            for (const [name, i] of adds) out.push({ label: `add ${name}`, cols: [...cols.slice(0, i), id, ...cols.slice(i)] });
        }
        return out;
    }
    // </pure>

    addStyle(`
        .cb-split-root, .cb-split-menu, .cb-split-dialog {
            --cbs-bg: #f5f6f8; --cbs-panel: #ffffff; --cbs-raised: #eef0f3; --cbs-line: #dfe1e6;
            --cbs-text: #141517; --cbs-sub: #697183; --cbs-accent: #00c785; --cbs-accent-ink: #ffffff; --cbs-live: #ff3b45;
            font-family: -apple-system, BlinkMacSystemFont, "Malgun Gothic", "맑은 고딕", Helvetica, Arial, sans-serif;
            color: var(--cbs-text); box-sizing: border-box;
        }
        html.theme_dark .cb-split-root, html.theme_dark .cb-split-menu, html.theme_dark .cb-split-dialog {
            --cbs-bg: #141517; --cbs-panel: #1d1e21; --cbs-raised: #2a2c30; --cbs-line: #2e3033;
            --cbs-text: #fbfbfe; --cbs-sub: #9da5b6; --cbs-accent: #00ffa3; --cbs-accent-ink: #141517; --cbs-live: #ff4a55;
        }
        .cb-split-root *, .cb-split-menu *, .cb-split-dialog * { box-sizing: border-box; }
        .cb-split-root [hidden] { display: none !important; }
        html.cb-split-on #layout-body > * { visibility: hidden !important; }
        .cb-split-root { position: fixed; right: 0; bottom: 0; z-index: 50; display: flex; flex-direction: column; background: var(--cbs-bg); min-width: 0; }
        .cb-split-bar { flex: none; display: flex; align-items: center; gap: 8px; height: 48px; padding: 0 12px; border-bottom: 1px solid var(--cbs-line); }
        .cb-split-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
        .cb-split-title svg { color: var(--cbs-accent); }
        .cb-split-count { font-size: 12px; color: var(--cbs-sub); font-weight: 500; }
        .cb-split-spacer { flex: 1; }
        .cb-split-btn { appearance: none; border: 0; border-radius: 8px; height: 32px; padding: 0 12px; font: inherit; font-size: 13px; font-weight: 600;
            background: var(--cbs-raised); color: var(--cbs-text); cursor: pointer; display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
        .cb-split-btn:hover:not(:disabled) { filter: brightness(1.15); }
        .cb-split-btn:disabled { opacity: .4; cursor: default; }
        .cb-split-btn--primary { background: var(--cbs-accent); color: var(--cbs-accent-ink); }
        .cb-split-grid { flex: 1; min-height: 0; display: flex; gap: 8px; padding: 8px; overflow-x: auto; scroll-snap-type: x mandatory; }
        .cb-split-col { flex: 1 1 0; min-width: min(100%, 320px); scroll-snap-align: start; display: flex; flex-direction: column; background: var(--cbs-panel); border: 1px solid var(--cbs-line); border-radius: 10px; overflow: hidden; }
        .cb-split-head { flex: none; display: flex; align-items: center; gap: 8px; height: 38px; padding: 0 6px 0 8px; border-bottom: 1px solid var(--cbs-line); }
        .cb-split-pos { flex: none; width: 22px; height: 22px; border-radius: 6px; display: grid; place-items: center; font-size: 11px; font-weight: 800; background: var(--cbs-raised); color: var(--cbs-sub); }
        .cb-split-avatar { flex: none; width: 24px; height: 24px; border-radius: 50%; object-fit: cover; background: var(--cbs-raised); }
        .cb-split-name { flex: 1; min-width: 0; font-size: 13px; font-weight: 700; color: inherit; text-decoration: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .cb-split-name:hover { text-decoration: underline; }
        .cb-split-tag { flex: none; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 5px; background: var(--cbs-raised); color: var(--cbs-sub); }
        .cb-split-tag--stt { color: var(--cbs-accent); }
        .cb-split-tag--live { background: var(--cbs-live); color: #fff; }
        .cb-split-icon { appearance: none; border: 0; background: none; color: var(--cbs-sub); width: 28px; height: 28px; border-radius: 6px; cursor: pointer; display: grid; place-items: center; font-size: 16px; line-height: 1; }
        .cb-split-icon:hover { background: var(--cbs-raised); color: var(--cbs-text); }
        .cb-split-body { flex: 1; min-height: 0; position: relative; }
        .cb-split-body > iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; display: block; }
        .cb-split-off { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; padding: 16px; text-align: center; color: var(--cbs-sub); font-size: 13px; }
        .cb-split-off img { width: 72px; height: 72px; border-radius: 50%; object-fit: cover; filter: grayscale(1); opacity: .7; }
        .cb-split-off strong { color: var(--cbs-text); font-size: 15px; }
        .cb-split-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--cbs-sub); font-size: 14px; }
        .cb-split-more { position: absolute; right: 4px; top: 50%; transform: translateY(-50%); z-index: 3; width: 24px; height: 24px; border: 0; border-radius: 6px;
            display: grid; place-items: center; padding: 0; cursor: pointer; background: #2a2c30; color: #fbfbfe; opacity: 0; transition: opacity .12s; }
        html:not(.theme_dark) .cb-split-more { background: #eef0f3; color: #141517; }
        .cb-split-more svg { width: 16px; height: 16px; }
        :hover > .cb-split-more, .cb-split-more:focus-visible, .cb-split-more[aria-expanded="true"] { opacity: 1; }
        .cb-split-more:disabled { cursor: not-allowed; }
        :hover > .cb-split-more:disabled { opacity: .35; }
        .cb-split-menu { position: fixed; z-index: 10000; min-width: 168px; padding: 6px; border-radius: 10px; background: var(--cbs-panel); border: 1px solid var(--cbs-line);
            box-shadow: 0 8px 28px rgba(0,0,0,.35); display: flex; flex-direction: column; gap: 2px; }
        .cb-split-menu-head { padding: 6px 8px 8px; font-size: 12px; color: var(--cbs-sub); border-bottom: 1px solid var(--cbs-line); margin-bottom: 4px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .cb-split-menu button { appearance: none; border: 0; background: none; color: var(--cbs-text); font: inherit; font-size: 13px; font-weight: 600; text-align: left; padding: 8px 10px; border-radius: 6px; cursor: pointer; }
        .cb-split-menu button:hover, .cb-split-menu button:focus-visible { background: var(--cbs-raised); outline: none; }
        .cb-split-menu-note { padding: 8px 10px; font-size: 12px; color: var(--cbs-sub); }
        .cb-split-scrim { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.5); display: grid; place-items: center; padding: 16px; }
        .cb-split-dialog { width: min(480px, 100%); max-height: min(640px, 100%); display: flex; flex-direction: column; background: var(--cbs-panel); border: 1px solid var(--cbs-line); border-radius: 14px; overflow: hidden; }
        .cb-split-dialog h3 { margin: 0; padding: 16px 16px 0; font-size: 16px; font-weight: 700; }
        .cb-split-dialog form { display: flex; gap: 8px; padding: 12px 16px 4px; }
        .cb-split-dialog input { flex: 1; min-width: 0; height: 36px; border-radius: 8px; border: 1px solid var(--cbs-line); background: var(--cbs-bg); color: var(--cbs-text); padding: 0 10px; font: inherit; font-size: 13px; }
        .cb-split-dialog input:focus { outline: 2px solid var(--cbs-accent); outline-offset: -1px; }
        .cb-split-error { min-height: 18px; padding: 0 16px; font-size: 12px; color: var(--cbs-live); }
        .cb-split-lists { overflow: auto; padding: 0 8px 12px; }
        .cb-split-lists h4 { margin: 10px 8px 4px; font-size: 12px; color: var(--cbs-sub); font-weight: 600; }
        .cb-split-item { width: 100%; appearance: none; border: 0; background: none; color: inherit; font: inherit; display: flex; align-items: center; gap: 10px; padding: 8px; border-radius: 8px; cursor: pointer; text-align: left; }
        .cb-split-item:hover:not(:disabled) { background: var(--cbs-raised); }
        .cb-split-item:disabled { opacity: .45; cursor: default; }
        .cb-split-item img { flex: none; width: 32px; height: 32px; border-radius: 50%; object-fit: cover; background: var(--cbs-raised); }
        .cb-split-item span { min-width: 0; display: flex; flex-direction: column; }
        .cb-split-item b { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .cb-split-item small { font-size: 12px; color: var(--cbs-sub); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .cb-split-hint { padding: 4px 8px; font-size: 12px; color: var(--cbs-sub); }
    `);

    const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] => {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text !== undefined) node.textContent = text;
        return node;
    };

    // ---- the split page --------------------------------------------------
    const host = (() => {
        let cols: string[] = [];
        let root: HTMLDivElement | null = null;
        let grid: HTMLDivElement | null = null;
        let countEl: HTMLSpanElement | null = null;
        let addBtn: HTMLButtonElement | null = null;
        let exitBtn: HTMLButtonElement | null = null;
        let emptyEl: HTMLDivElement | null = null;
        const columns = new Map<string, Column>(); // id -> { id, el, body, frame, poll, ... }

        function mount() {
            cols = colsFromUrl(location);
            if (splitUrl(cols) !== here()) rawReplace(history.state, "", splitUrl(cols));
            root = el("div", "cb-split-root");
            const bar = el("div", "cb-split-bar");
            const title = el("div", "cb-split-title");
            title.innerHTML = ICON;
            title.append(el("span", "", "스플릿 뷰"));
            countEl = el("span", "cb-split-count");
            title.append(countEl);
            addBtn = el("button", "cb-split-btn cb-split-btn--primary", "+ 채널 추가");
            addBtn.type = "button";
            addBtn.addEventListener("click", openDialog);
            exitBtn = el("button", "cb-split-btn", "나가기");
            exitBtn.type = "button";
            exitBtn.title = "왼쪽 칸만 남겨요";
            exitBtn.addEventListener("click", () => set(cols.slice(0, 1)));
            bar.append(title, el("div", "cb-split-spacer"), addBtn, exitBtn);
            grid = el("div", "cb-split-grid");
            emptyEl = el("div", "cb-split-empty");
            emptyEl.append(el("div", "", "보고 싶은 채널을 최대 세 개까지 추가하세요."));
            const emptyAdd = el("button", "cb-split-btn cb-split-btn--primary", "+ 채널 추가");
            emptyAdd.type = "button";
            emptyAdd.addEventListener("click", openDialog);
            emptyEl.append(emptyAdd);
            root.append(bar, grid, emptyEl);
            document.body.append(root);
            document.documentElement.classList.add("cb-split-on");
            place();
            render();
            logger.info`mounted ${cols}`;
        }
        function unmount() {
            for (const id of [...columns.keys()]) dispose(id);
            root?.remove();
            root = null;
            document.documentElement.classList.remove("cb-split-on");
            logger.info("unmounted");
        }
        function place() {
            if (!root) return;
            const header = document.querySelector("#header");
            const sidebar = document.querySelector("#sidebar");
            const top = header ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
            const sr = sidebar?.getBoundingClientRect();
            const left = sr && sr.width > 0 && getComputedStyle(sidebar!).display !== "none" ? Math.max(0, sr.right) : 0;
            root.style.top = `${top}px`;
            root.style.left = `${left}px`;
        }
        // Every change is a history entry unless it came from one.
        function set(next: string[], entry: "push" | "none" = "push") {
            cols = [...new Set(next)].slice(0, MAX);
            saveLayout(cols);
            if (entry === "push" && splitUrl(cols) !== here()) rawPush(history.state, "", splitUrl(cols));
            render();
        }
        function retitle() {
            const names = cols.map((id) => columns.get(id)?.name.textContent ?? id.slice(0, 8));
            document.title = `${names.length ? names.join(" · ") : "스플릿 뷰"} - CHZZK`;
        }
        // A column frame left its channel: another channel takes its
        // column, anything else is a page for the whole tab.
        function navigated(col: string, href: string) {
            if (!cols.includes(col)) return;
            const url = new URL(href, location.href);
            const id = url.origin === location.origin ? url.pathname.match(LIVE_RE)?.[1]?.toLowerCase() : undefined;
            logger.info`column ${col} went to ${href}`;
            if (id) return set(cols.map((x) => (x === col ? id : x)));
            location.assign(url.href);
        }
        function render() {
            if (!root) return;
            const pos = positions(cols.length);
            for (const id of [...columns.keys()]) if (!cols.includes(id)) dispose(id);
            cols.forEach((id, i) => {
                const c = columns.get(id) ?? create(id);
                const p = pos[i];
                c.el.style.order = String(i);
                c.el.dataset.pos = p;
                c.pos.textContent = p;
                c.pos.title = { L: "왼쪽 귀", C: "가운데", R: "오른쪽 귀" }[p];
                c.stt.hidden = i !== 0;
                c.close.hidden = cols.length < 2;
                if (c.frame) syncFrame(c, p, i === 0);
            });
            countEl!.textContent = `${cols.length}/${MAX}`;
            addBtn!.disabled = cols.length >= MAX;
            exitBtn!.hidden = cols.length < 2;
            retitle();
            grid!.hidden = cols.length === 0;
            emptyEl!.hidden = cols.length > 0;
        }
        function syncFrame(c: Column, p: Pos, stt: boolean) {
            c.frame!.dataset.cbStt = stt ? "1" : "0";
            // Re-renders keep the column's own pan; only a new position
            // resets it, pushed now rather than on the frame's next tick.
            if (c.frame!.dataset.cbPanMin === String(PAN[p].min) && c.frame!.dataset.cbPanMax === String(PAN[p].max)) return;
            writePan(c.frame!, p);
            try {
                c.frame!.contentWindow?.ChzzkBestPan?.set(PAN[p].def);
            } catch (error) {
                logger.error`pan push failed ${c.id} ${error}`;
            }
        }
        function create(id: string) {
            const c = { id, el: el("section", "cb-split-col"), body: el("div", "cb-split-body"), frame: null, poll: 0 } as Column;
            c.el.dataset.id = id;
            const head = el("div", "cb-split-head");
            c.pos = el("span", "cb-split-pos");
            c.avatar = el("img", "cb-split-avatar");
            c.avatar.alt = "";
            c.name = el("a", "cb-split-name", id.slice(0, 8));
            c.name.href = `/live/${id}`;
            c.name.target = "_blank";
            c.name.rel = "noopener";
            c.name.title = "새 탭에서 열기";
            c.live = el("span", "cb-split-tag", "확인 중");
            c.stt = el("span", "cb-split-tag cb-split-tag--stt", "자막");
            c.stt.title = "실시간 자막은 왼쪽 칸에서 돌아요";
            c.close = el("button", "cb-split-icon", "×");
            c.close.type = "button";
            c.close.title = "칸 닫기";
            c.close.setAttribute("aria-label", "칸 닫기");
            c.close.addEventListener("click", () => set(cols.filter((x) => x !== id)));
            head.append(c.pos, c.avatar, c.name, c.live, c.stt, c.close);
            c.el.append(head, c.body);
            columns.set(id, c);
            grid!.append(c.el);
            check(c);
            return c;
        }
        async function check(c: Column) {
            const info = await channelInfo(c.id, 0);
            if (columns.get(c.id) !== c) return;
            c.name.textContent = info.name;
            retitle();
            if (info.image) c.avatar.src = info.image;
            if (!info.exists) return offline(c, "존재하지 않는 채널이에요", "주소나 ID를 다시 확인해 주세요.", false);
            if (info.live === false) return offline(c, "오프라인", "방송이 시작되면 이 칸에서 자동으로 열려요.", true, info);
            c.live.textContent = "LIVE";
            c.live.className = "cb-split-tag cb-split-tag--live";
            if (!c.frame) attach(c);
        }
        function offline(c: Column, title: string, text: string, poll: boolean, info?: ChannelInfo) {
            c.live.textContent = title === "오프라인" ? "OFF" : "없음";
            c.live.className = "cb-split-tag";
            c.body.textContent = "";
            const card = el("div", "cb-split-off");
            if (info?.image) {
                const img = el("img");
                img.src = info.image;
                img.alt = "";
                card.append(img);
            }
            card.append(el("strong", "", title), el("div", "", text));
            c.body.append(card);
            clearInterval(c.poll);
            if (poll) c.poll = setInterval(() => check(c), 30000);
        }
        function attach(c: Column) {
            clearInterval(c.poll);
            c.body.textContent = "";
            const frame = el("iframe");
            frame.dataset.cbCol = c.id;
            const i = cols.indexOf(c.id);
            const p = positions(cols.length)[i] ?? "C";
            writePan(frame, p);
            frame.dataset.cbStt = i === 0 ? "1" : "0";
            frame.allow = "autoplay; fullscreen; picture-in-picture; clipboard-write";
            frame.allowFullscreen = true;
            frame.title = c.name.textContent ?? "";
            frame.src = `/live/${c.id}`;
            c.frame = frame;
            c.body.append(frame);
        }
        // Frees the column's player before its frame goes: media detached,
        // its subtitle state dropped from the shared worker.
        function dispose(id: string) {
            const c = columns.get(id);
            if (!c) return;
            columns.delete(id);
            clearInterval(c.poll);
            if (c.frame) {
                try {
                    for (const media of c.frame.contentDocument?.querySelectorAll<HTMLMediaElement>("video, audio") ?? []) {
                        media.pause();
                        media.removeAttribute("src");
                        media.load();
                    }
                } catch (error) {
                    logger.error`media release failed ${id} ${error}`;
                }
                pageWindow.ChzzkBestSttHub?.drop(id);
                c.frame.src = "about:blank";
                c.frame = null;
            }
            c.el.remove();
        }
        function gesture(fromCol: string | null) {
            for (const c of columns.values()) {
                if (!c.frame || c.id === fromCol) continue;
                try {
                    const w = c.frame.contentWindow as Window & typeof globalThis;
                    w.dispatchEvent(new w.PointerEvent("pointerdown"));
                } catch (error) {
                    logger.error`gesture forward failed ${c.id} ${error}`;
                }
            }
        }

        // ---- add dialog ----
        function openDialog() {
            if (cols.length >= MAX || document.querySelector(".cb-split-scrim")) return;
            const scrim = el("div", "cb-split-scrim");
            const dialog = el("div", "cb-split-dialog");
            dialog.setAttribute("role", "dialog");
            dialog.setAttribute("aria-label", "채널 추가");
            const form = el("form");
            const input = el("input");
            input.placeholder = "채널 주소 또는 ID 붙여넣기";
            input.spellcheck = false;
            const submit = el("button", "cb-split-btn cb-split-btn--primary", "추가");
            submit.type = "submit";
            form.append(input, submit);
            const error = el("div", "cb-split-error");
            const lists = el("div", "cb-split-lists");
            dialog.append(el("h3", "", "채널 추가"), form, error, lists);
            scrim.append(dialog);
            document.body.append(scrim);
            input.focus();
            const close = () => {
                scrim.remove();
                document.removeEventListener("keydown", onKey, true);
            };
            const onKey = (event: KeyboardEvent) => {
                if (event.key === "Escape") close();
            };
            document.addEventListener("keydown", onKey, true);
            scrim.addEventListener("click", (event) => event.target === scrim && close());
            const add = (id: string) => {
                if (cols.includes(id)) return (error.textContent = "이미 열려 있는 채널이에요.");
                if (cols.length >= MAX) return (error.textContent = "칸은 세 개까지예요.");
                close();
                set([...cols, id]);
            };
            form.addEventListener("submit", async (event) => {
                event.preventDefault();
                const id = input.value.match(ID_RE)?.[1]?.toLowerCase();
                if (!id) return (error.textContent = "chzzk.naver.com/live/… 주소나 32자리 채널 ID를 넣어 주세요.");
                submit.disabled = true;
                const info = await channelInfo(id);
                submit.disabled = false;
                if (!info.exists) return (error.textContent = "존재하지 않는 채널이에요.");
                add(id);
            });
            const section = (title: string) => {
                lists.append(el("h4", "", title));
                const box = el("div");
                box.append(el("div", "cb-split-hint", "불러오는 중…"));
                lists.append(box);
                return box;
            };
            const fill = (box: HTMLDivElement, items: LiveItem[], emptyText: string) => {
                box.textContent = "";
                if (!items.length) return box.append(el("div", "cb-split-hint", emptyText));
                for (const item of items) {
                    const row = el("button", "cb-split-item");
                    row.type = "button";
                    const img = el("img");
                    img.alt = "";
                    if (item.image) img.src = item.image;
                    const text = el("span");
                    text.append(el("b", "", item.name));
                    const viewers = item.viewers != null ? `${Number(item.viewers).toLocaleString()}명 · ` : "";
                    text.append(el("small", "", cols.includes(item.id) ? "이미 열려 있어요" : `${viewers}${item.title}`));
                    row.append(img, text);
                    row.disabled = cols.includes(item.id);
                    row.addEventListener("click", () => add(item.id));
                    box.append(row);
                }
            };
            const following = section("팔로잉 라이브");
            getJson("https://api.chzzk.naver.com/service/v1/channels/followings/live")
                .then((json) => fill(following, liveItems(json).filter((x) => x.open), "지금 방송 중인 팔로잉 채널이 없어요."))
                .catch((err: Error & { status?: number }) => {
                    following.textContent = "";
                    following.append(el("div", "cb-split-hint", err.status === 401 ? "로그인하면 팔로잉 라이브가 여기에 나와요." : "팔로잉 목록을 불러오지 못했어요."));
                    if (err.status !== 401) logger.error`followings failed ${err}`;
                });
            const popular = section("인기 라이브");
            getJson("https://api.chzzk.naver.com/service/v1/lives?size=20&sortType=POPULAR")
                .then((json) => fill(popular, liveItems(json), "목록이 비어 있어요."))
                .catch((err: unknown) => {
                    popular.textContent = "";
                    popular.append(el("div", "cb-split-hint", "인기 라이브를 불러오지 못했어요."));
                    logger.error`popular failed ${err}`;
                });
        }

        window.addEventListener("pointerdown", (event) => event.isTrusted && gesture(null), true);
        window.addEventListener("keydown", (event) => event.isTrusted && gesture(null), true);
        window.addEventListener("resize", place);
        return {
            mount,
            unmount,
            place,
            set,
            gesture,
            navigated,
            mounted: () => !!root,
            cols: () => [...cols],
            stats: () => ({ cols: [...cols], frames: [...columns.values()].map((c) => ({ id: c.id, pos: c.el.dataset.pos, live: !!c.frame, pan: c.frame?.dataset.cbPan, stt: c.frame?.dataset.cbStt })) }),
        };
    })();

    // ---- the sidebar menu ----------------------------------------------
    let menu: OpenMenu | null = null;
    function menuContext(): { where: "host" | "live" | "other"; cols: string[] } {
        if (cbSplit.isHost()) return { where: "host", cols: host.cols() };
        const cur = currentId();
        if (cur) return { where: "live", cols: [cur] };
        return { where: "other", cols: loadLayout() };
    }
    function commit(where: "host" | "live" | "other", cols: string[]) {
        if (where === "host") return host.set(cols);
        saveLayout(cols);
        location.assign(splitUrl(cols));
    }
    function closeMenu() {
        if (!menu) return;
        menu.el.remove();
        menu.btn.setAttribute("aria-expanded", "false");
        document.removeEventListener("pointerdown", menu.outside, true);
        document.removeEventListener("keydown", menu.key, true);
        menu = null;
    }
    function openMenu(btn: HTMLButtonElement, id: string) {
        if (menu?.btn === btn) return closeMenu();
        closeMenu();
        const ctx = menuContext();
        const box = el("div", "cb-split-menu");
        box.setAttribute("role", "menu");
        const head = el("div", "cb-split-menu-head", "스플릿 뷰");
        box.append(head);
        channelInfo(id).then((info) => {
            if (box.isConnected) head.textContent = `${info.name} · 스플릿 뷰`;
        });
        const actions = menuActions(ctx.cols, id);
        for (const action of actions) {
            const item = el("button", "", action.label);
            item.type = "button";
            item.setAttribute("role", "menuitem");
            item.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                closeMenu();
                commit(ctx.where, action.cols);
            });
            box.append(item);
        }
        if (!actions.length) box.append(el("div", "cb-split-menu-note", "이미 이 자리에 있어요."));
        document.body.append(box);
        const anchor = btn.parentElement!.getBoundingClientRect();
        const r = box.getBoundingClientRect();
        box.style.left = `${Math.min(anchor.right + 6, innerWidth - r.width - 8)}px`;
        box.style.top = `${Math.max(8, Math.min(anchor.top, innerHeight - r.height - 8))}px`;
        btn.setAttribute("aria-expanded", "true");
        const outside = (event: PointerEvent) => {
            if (!box.contains(event.target as Node | null) && event.target !== btn && !btn.contains(event.target as Node | null)) closeMenu();
        };
        const key = (event: KeyboardEvent) => event.key === "Escape" && closeMenu();
        document.addEventListener("pointerdown", outside, true);
        document.addEventListener("keydown", key, true);
        menu = { el: box, btn, outside, key };
        box.querySelector("button")?.focus();
    }
    function decorateSidebar() {
        for (const link of document.querySelectorAll<HTMLAnchorElement>('#sidebar a[href^="/live/"]')) {
            const item = link.parentElement;
            const id = link.getAttribute("href")!.match(ID_RE)?.[1]?.toLowerCase();
            if (!item || !id) continue;
            const live = !!item.querySelector('[class*="_is_live_"]');
            let btn = item.querySelector<HTMLButtonElement>(":scope > .cb-split-more");
            if (!btn) {
                btn = el("button", "cb-split-more");
                btn.type = "button";
                btn.innerHTML = ICON;
                btn.setAttribute("aria-haspopup", "menu");
                btn.setAttribute("aria-expanded", "false");
                for (const type of ["pointerdown", "mousedown", "mouseup"]) btn.addEventListener(type, (event) => event.stopPropagation());
                btn.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (!btn!.disabled) openMenu(btn!, btn!.dataset.id!);
                });
                if (getComputedStyle(item).position === "static") item.style.position = "relative";
                item.append(btn);
            }
            if (btn.dataset.id !== id && menu?.btn === btn) closeMenu();
            btn.dataset.id = id;
            btn.disabled = !live;
            const label = live ? "스플릿 뷰 메뉴" : "오프라인 채널은 스플릿 뷰에 넣을 수 없어요";
            btn.title = label;
            btn.setAttribute("aria-label", label);
        }
    }

    function tick() {
        try {
            const isHost = cbSplit.isHost();
            if (isHost && boot.ready && !host.mounted() && document.body) host.mount();
            else if (!isHost && host.mounted()) host.unmount();
            if (isHost) host.place();
            decorateSidebar();
        } catch (error) {
            logger.error`tick failed ${location.href} ${error}`;
        }
    }
    pageWindow.ChzzkBestSplit = { gesture: (col: string | null) => host.gesture(col), navigated: (col: string, href: string) => host.navigated(col, href), stats: () => host.stats(), set: (cols: string[]) => host.set(cols), menuActions, loadLayout };
    setInterval(tick, 700);
    let scheduled = false;
    new MutationObserver(() => {
        if (scheduled) return;
        scheduled = true;
        setTimeout(() => {
            scheduled = false;
            tick();
        }, 200);
    }).observe(document, { subtree: true, childList: true });
})();
