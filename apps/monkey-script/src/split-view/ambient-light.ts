import { cbSplit } from "../shared/split-context.ts";
import { getLogger } from "../shared/logtape.ts";
import { videoRect } from "../shared/video-rect.ts";

const logger = getLogger(["split-view", "ambient-light"]);

// ---- 앰비언트 라이트: 칸의 빈 여백을 영상 색으로 채우기 ------------------
// The split page copies each column's <video> (same-origin frames) into a
// 32x18 canvas a few times a second and stretches that canvas, blurred,
// over the picture's box on one layer behind every column, so the light
// spills past the column into the gaps and the bands around the player,
// and neighbouring columns' light screen-blends where it overlaps. While
// it is on, the columns and the frames' pages behind the player turn
// transparent to let it through. Pixels are never read back, so a tainted
// canvas draws just the same. Inside each frame a second canvas fills the
// chat panel behind its messages, whose background drops to 60% so the
// glow shows through and the text stays readable. A row in the player
// settings menu turns it off for every column (saved, default on).
declare global {
    interface Window {
        ChzzkBestAmbient?: { stats(): { on: boolean; shown: boolean; fps: number; draws: number } };
    }
}

const KEY_ON = "split.ambient";
const FPS = 8;
const FPS_REDUCED = 2;
// How often a page with nothing to draw (hidden, paused, off) looks again.
const IDLE_MS = 1000;
const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const readOn = () => {
    try {
        return GM_getValue<boolean>(KEY_ON, true) !== false;
    } catch (error) {
        logger.error`setting load failed ${error}`;
        return true;
    }
};
function glowCanvas(className: string) {
    const c = Object.assign(document.createElement("canvas"), { className, width: 32, height: 18 });
    c.setAttribute("aria-hidden", "true");
    return c;
}
const playable = (video: HTMLVideoElement) => !document.hidden && !video.paused && !video.ended && video.readyState >= 2;

// ---- the split page: every column's light on one layer ------------------
(() => {
    "use strict";
    if (cbSplit.frame || window.top !== window) return;
    // The layer sits under the grid, clipped to it; a glow's box is its
    // picture's, scaled and blurred past it. The column header keeps 60%
    // of its panel so its text stays readable over the light.
    const STYLE = `
        .cb-ambient-layer { position: absolute; z-index: 0; overflow: hidden; pointer-events: none; isolation: isolate; }
        .cb-ambient-layer > canvas {
            position: absolute; mix-blend-mode: screen; filter: blur(56px) saturate(1.6) brightness(1.1); opacity: .8; transform: scale(1.12, 1.45);
        }
        html.cb-ambient-on .cb-split-grid { position: relative; z-index: 1; }
        html.cb-ambient-on .cb-split-col { background: transparent; }
        html.cb-ambient-on .cb-split-head { background: color-mix(in srgb, var(--cbs-panel) 60%, transparent); }
    `;
    let on = readOn();
    let draws = 0;
    let since = performance.now();
    let fps = 0;
    let layer: HTMLDivElement | null = null;
    const glows = new Map<HTMLIFrameElement, { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D }>();
    function drop() {
        layer?.remove();
        layer = null;
        glows.clear();
        document.documentElement.classList.remove("cb-ambient-on");
    }
    function ensureStyle() {
        if (document.getElementById("cb-ambient-style") || !document.head) return;
        const style = document.createElement("style");
        style.id = "cb-ambient-style";
        style.textContent = STYLE;
        document.head.append(style);
    }

    // Boxes are measured on every draw, so a resize, a scrolled grid or a
    // column added, moved or dropped is followed within one beat.
    function frame() {
        let wait = IDLE_MS;
        try {
            const root = document.querySelector<HTMLElement>(".cb-split-root");
            const grid = root?.querySelector<HTMLElement>(".cb-split-grid");
            if (!on || !root || !grid) drop();
            else {
                ensureStyle();
                document.documentElement.classList.add("cb-ambient-on");
                if (!layer || layer.parentElement !== root) {
                    layer ??= Object.assign(document.createElement("div"), { className: "cb-ambient-layer" });
                    root.prepend(layer);
                }
                const r = root.getBoundingClientRect();
                const g = grid.getBoundingClientRect();
                Object.assign(layer.style, { left: `${g.left - r.left}px`, top: `${g.top - r.top}px`, width: `${g.width}px`, height: `${g.height}px` });
                const frames = [...grid.querySelectorAll<HTMLIFrameElement>(".cb-split-body > iframe")];
                let drew = false;
                for (const f of frames) {
                    const video = f.contentDocument?.querySelector<HTMLVideoElement>("#layout-body main video");
                    const v = video && videoRect(video);
                    let glow = glows.get(f);
                    if (!video || !v?.width || !v.height) {
                        glow?.canvas.remove();
                        glows.delete(f);
                        continue;
                    }
                    if (!glow) {
                        const canvas = glowCanvas("cb-ambient");
                        glow = { canvas, ctx: canvas.getContext("2d", { alpha: false })! };
                        glows.set(f, glow);
                    }
                    if (glow.canvas.parentElement !== layer) layer.append(glow.canvas);
                    const fb = f.getBoundingClientRect();
                    Object.assign(glow.canvas.style, { left: `${fb.left + v.left - g.left}px`, top: `${fb.top + v.top - g.top}px`, width: `${v.width}px`, height: `${v.height}px` });
                    if (playable(video)) {
                        glow.ctx.drawImage(video, 0, 0, 32, 18);
                        drew = true;
                    }
                }
                for (const [f, glow] of glows) {
                    if (frames.includes(f)) continue;
                    glow.canvas.remove();
                    glows.delete(f);
                }
                if (drew) {
                    draws++;
                    wait = 1000 / (reduced.matches ? FPS_REDUCED : FPS);
                }
            }
        } catch (error) {
            logger.error`split draw failed ${error}`;
        }
        const now = performance.now();
        if (now - since >= 2000) {
            fps = Math.round((draws * 10000) / (now - since)) / 10;
            draws = 0;
            since = now;
        }
        timer = setTimeout(frame, wait);
    }
    let timer = setTimeout(frame, IDLE_MS);
    const wake = () => {
        clearTimeout(timer);
        timer = setTimeout(frame, 0);
    };
    // A column's play event stays in its frame; the frames' toggle reaches
    // here through the saved setting.
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("resize", wake);
    setInterval(() => {
        const saved = readOn();
        if (saved === on) return;
        on = saved;
        wake();
    }, 1000);
    pageWindow.ChzzkBestAmbient = { stats: () => ({ on, shown: glows.size > 0, fps, draws }) };
})();

// ---- inside a column frame ----------------------------------------------
(() => {
    "use strict";
    if (!cbSplit.frame) return;
    let on = readOn();
    let draws = 0;
    let since = performance.now();
    let fps = 0;

    // The page behind the player lets the split page's light through; the
    // player box itself stays black for a picture that does not fill it.
    const STYLE = `
        html.cb-ambient-on, html.cb-ambient-on body, html.cb-ambient-on #layout-body, html.cb-ambient-on #layout-body > section,
        html.cb-ambient-on #layout-body > section > div, html.cb-ambient-on #layout-body > section > div > main,
        html.cb-ambient-on #layout-body main > :first-child { background: transparent !important; }
        #aside-chatting:has(> .cb-ambient) {
            isolation: isolate; background-color: color-mix(in srgb, var(--sem-color-background-neutral-base) 60%, transparent) !important;
        }
        #aside-chatting > canvas.cb-ambient {
            position: absolute; inset: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none;
            filter: blur(56px) saturate(1.6) brightness(1.1); opacity: .8;
        }
        /* The header, ranking and input rows paint the panel colour themselves. The pinned message keeps the
           60% layer, since the list scrolls under it. */
        #aside-chatting:has(> .cb-ambient) > *, #aside-chatting:has(> .cb-ambient) > * > h2 { background-color: transparent !important; }
        #aside-chatting:has(> .cb-ambient) > * > [class*="_fixed_"] {
            background-color: color-mix(in srgb, var(--sem-color-background-neutral-base) 60%, transparent) !important;
        }
        .cb-ambient-item { cursor: pointer; }
        .pzp-pc:not(.pzp-pc--setting-home) .pzp-settings > .cb-ambient-item { display: none; }
    `;
    function ensureStyle() {
        if (document.getElementById("cb-ambient-style") || !document.head) return;
        const style = document.createElement("style");
        style.id = "cb-ambient-style";
        style.textContent = STYLE;
        document.head.append(style);
    }

    let chat: HTMLCanvasElement | null = null;
    let chatCtx: CanvasRenderingContext2D | null = null;
    function drop() {
        chat?.remove();
        chat = null;
        chatCtx = null;
    }

    function frame() {
        let wait = IDLE_MS;
        try {
            const main = document.querySelector<HTMLElement>("#layout-body main");
            const video = main?.querySelector("video");
            const aside = document.querySelector<HTMLElement>("#aside-chatting");
            document.documentElement.classList.toggle("cb-ambient-on", on);
            if (!on || !main || !video) drop();
            else {
                if (!aside) chat?.remove();
                else if (!chat || chat.parentElement !== aside) {
                    chat ??= glowCanvas("cb-ambient");
                    chatCtx = chat.getContext("2d", { alpha: false });
                    aside.append(chat);
                }
                if (aside && playable(video)) {
                    chatCtx!.drawImage(video, 0, 0, 32, 18);
                    draws++;
                    wait = 1000 / (reduced.matches ? FPS_REDUCED : FPS);
                }
            }
        } catch (error) {
            logger.error`draw failed ${error}`;
        }
        const now = performance.now();
        if (now - since >= 2000) {
            fps = Math.round((draws * 10000) / (now - since)) / 10;
            draws = 0;
            since = now;
        }
        timer = setTimeout(frame, wait);
    }
    let timer = setTimeout(frame, IDLE_MS);
    // A column that starts playing or comes back into view draws at once
    // instead of on the idle beat.
    const wake = () => {
        clearTimeout(timer);
        timer = setTimeout(frame, 0);
    };
    document.addEventListener("play", wake, true);
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("pagehide", () => {
        clearTimeout(timer);
        drop();
    });

    function setOn(value: boolean) {
        on = value;
        GM_setValue(KEY_ON, on);
        syncItem();
        wake();
    }
    function syncItem() {
        const item = document.querySelector<HTMLElement>(".pzp-pc .cb-ambient-item");
        if (!item) return;
        item.setAttribute("aria-checked", String(on));
        const box = item.querySelector<HTMLInputElement>(".pzp-ui-toggle__checkbox")!;
        if (box.checked !== on) box.checked = on;
    }

    // Sits under the player's own 선명한 화면 (Vivid screen) row, the other
    // picture setting, or at the end of the menu without it.
    function ensureItem() {
        const menu = document.querySelector<HTMLElement>(".pzp-pc .pzp-settings");
        if (!menu) return;
        let item = menu.querySelector<HTMLElement>(":scope > .cb-ambient-item");
        if (!item) {
            item = document.createElement("div");
            item.className = "pzp-ui-setting-home-item cb-ambient-item";
            item.setAttribute("role", "menuitemcheckbox");
            item.tabIndex = 0;
            item.innerHTML =
                '<div class="pzp-ui-setting-home-item__top"><div class="pzp-ui-setting-home-item__left"><span class="pzp-ui-setting-home-item__label">앰비언트 라이트</span></div>' +
                '<div class="pzp-ui-setting-home-item__right"><div role="switch" class="pzp-ui-toggle"><input type="checkbox" tabindex="-1" class="pzp-ui-toggle__checkbox"> <div class="pzp-ui-toggle__handle"></div></div></div></div>';
            // The player reads clicks in its menu as navigation and keys as
            // seek/volume shortcuts.
            for (const type of ["pointerdown", "mousedown", "mouseup", "dblclick"]) item.addEventListener(type, (event) => event.stopPropagation());
            item.addEventListener("click", (event) => {
                event.stopPropagation();
                event.preventDefault();
                setOn(!on);
            });
            item.addEventListener("keydown", (event) => {
                event.stopPropagation();
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                setOn(!on);
            });
        }
        const anchor = menu.querySelector(".pzp-setting-intro-filter");
        if (anchor ? anchor.nextElementSibling !== item : item.parentElement !== menu) anchor ? anchor.after(item) : menu.append(item);
        syncItem();
    }

    function tick() {
        try {
            ensureStyle();
            // Another column may have flipped the saved setting.
            const saved = readOn();
            if (saved !== on) {
                on = saved;
                wake();
            }
            ensureItem();
        } catch (error) {
            logger.error`tick failed ${location.href} ${error}`;
        }
    }
    pageWindow.ChzzkBestAmbient = { stats: () => ({ on, shown: !!chat?.isConnected, fps, draws }) };
    setInterval(tick, 1000);
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
