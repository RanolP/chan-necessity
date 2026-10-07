import { cbSplit } from "../shared/split-context.ts";
import { getLogger } from "../shared/logtape.ts";
import { videoRect } from "../shared/video-rect.ts";

const logger = getLogger(["split-view", "ambient-light"]);

// ---- 앰비언트 라이트: 칸의 빈 여백을 영상 색으로 채우기 ------------------
// Each column frame copies its <video> into a 32x18 canvas a few times a
// second and stretches that canvas, blurred, behind the picture, so the
// black bands of the column (above and below a centred 16:9 player) glow
// in the picture's colours. Pixels are never read back, so a tainted
// canvas draws just the same. The canvas sits inside <main>, under the
// player in z-order; a column whose player fills <main> has no bands and
// draws nothing. A row in the player settings menu turns it off for every
// column (saved, default on).
declare global {
    interface Window {
        ChzzkBestAmbient?: { stats(): { on: boolean; shown: boolean; fps: number; draws: number } };
    }
}

(() => {
    "use strict";
    if (!cbSplit.frame) return;
    const KEY_ON = "split.ambient";
    const FPS = 8;
    const FPS_REDUCED = 2;
    // How often a column with nothing to draw (hidden, paused, off, no
    // bands) looks again.
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
    let on = readOn();
    let draws = 0;
    let since = performance.now();
    let fps = 0;

    // The column's split CSS hides every <main> child but the first; the
    // canvas is let through by a more specific selector.
    const STYLE = `
        #layout-body main:has(> .cb-ambient) { position: relative !important; isolation: isolate; }
        #layout-body main > canvas.cb-ambient {
            display: block !important; position: absolute; z-index: -1; pointer-events: none;
            filter: blur(56px) saturate(1.6) brightness(1.1); opacity: .8; transform: scale(1.12, 1.45);
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

    let canvas: HTMLCanvasElement | null = null;
    let ctx: CanvasRenderingContext2D | null = null;
    function drop() {
        canvas?.remove();
        canvas = null;
        ctx = null;
    }

    // Places the canvas over the picture's box inside <main>; false when
    // the picture leaves no band of <main> to light.
    function place(main: HTMLElement, video: HTMLVideoElement) {
        const m = main.getBoundingClientRect();
        const v = videoRect(video);
        if (!v.width || !v.height || (v.top - m.top < 2 && m.bottom - v.bottom < 2 && v.left - m.left < 2 && m.right - v.right < 2)) return false;
        if (!canvas || canvas.parentElement !== main) {
            canvas ??= Object.assign(document.createElement("canvas"), { className: "cb-ambient", width: 32, height: 18 });
            canvas.setAttribute("aria-hidden", "true");
            ctx = canvas.getContext("2d", { alpha: false });
            main.append(canvas);
        }
        const s = canvas.style;
        s.left = `${v.left - m.left}px`;
        s.top = `${v.top - m.top}px`;
        s.width = `${v.width}px`;
        s.height = `${v.height}px`;
        return true;
    }

    function frame() {
        let wait = IDLE_MS;
        try {
            const main = document.querySelector<HTMLElement>("#layout-body main");
            const video = main?.querySelector("video");
            if (!on || !main || !video) drop();
            else if (!place(main, video)) canvas?.remove();
            else if (!document.hidden && !video.paused && !video.ended && video.readyState >= 2) {
                ctx!.drawImage(video, 0, 0, 32, 18);
                draws++;
                wait = 1000 / (reduced.matches ? FPS_REDUCED : FPS);
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
    pageWindow.ChzzkBestAmbient = { stats: () => ({ on, shown: !!canvas?.isConnected, fps, draws }) };
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
