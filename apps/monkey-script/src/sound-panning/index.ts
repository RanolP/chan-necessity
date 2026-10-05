import { cbSplit } from "../shared/split-context.ts";
import { cbAudio } from "../shared/audio.ts";
import { getLogger } from "../shared/logtape.ts";

const logger = getLogger(["sound-panning"]);

// ---- 좌우 밸런스 ---------------------------------------------------------
// A slider in the player settings menu. The saved value shows in the slider
// at once but reaches the audio only after a gesture has started the
// shared context: the slider itself, or for a saved non-centre value, the
// first click or key press on the page.
declare global {
    interface Window {
        ChzzkBestPan?: { get(): number; set(value: number): Promise<void>; audio: typeof cbAudio };
    }
}

(() => {
    "use strict";
    const LIVE_RE = /^\/live\/[0-9a-f]{32}/i;
    const VIDEO_RE = /^\/video\/\d+/;
    const KEY_PAN = "audio.pan";
    const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;

    const clampPan = (v: number) => (Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0);
    let pan = 0;
    try {
        pan = cbSplit.frame ? clampPan(cbSplit.pan() ?? 0) : clampPan(Number(GM_getValue(KEY_PAN, 0)));
    } catch (error) {
        logger.error("pan setting load failed {error}", { error });
    }

    function apply() {
        const video = cbAudio.playerVideo();
        if (!video) return;
        const graph = pan === 0 ? cbAudio.graphOf(video) : cbAudio.route(video);
        if (!graph) return;
        const ctx = graph.panner.context;
        graph.panner.pan.setTargetAtTime(pan, ctx.currentTime, 0.015);
    }

    async function setPan(value: number) {
        pan = Math.abs(value) < 0.04 ? 0 : clampPan(Math.round(value * 20) / 20);
        // A split column's pan comes from its position; the saved
        // single-player value stays as it was.
        if (!cbSplit.frame) GM_setValue(KEY_PAN, pan);
        syncUi();
        if (pan !== 0) await cbAudio.context();
        apply();
    }

    const label = (v: number) =>
        v === 0 ? "가운데" : `${v < 0 ? "왼쪽" : "오른쪽"} ${Math.round(Math.abs(v) * 100)}%`;

    const STYLE = `
        .cb-pan-item { cursor: default; }
        .pzp-pc:not(.pzp-pc--setting-home) .pzp-settings > .cb-pan-item { display: none; }
        .cb-pan-row { display: flex; align-items: center; gap: 8px; padding: 0 0 10px; font-size: 12px; color: rgba(255,255,255,.6); }
        .cb-pan-range { flex: 1; min-width: 0; height: 4px; accent-color: #00ffa3; cursor: pointer; margin: 0; }
    `;
    function syncUi() {
        const item = document.querySelector(".cb-pan-item");
        if (!item) return;
        const range = item.querySelector<HTMLInputElement>(".cb-pan-range")!;
        if (Number(range.value) !== pan && document.activeElement !== range) range.value = String(pan);
        item.querySelector(".cb-pan-value")!.textContent = label(pan);
        range.setAttribute("aria-valuetext", label(pan));
    }

    // Sits directly above the player's own 라디오 모드 row in the settings
    // home, at the end of the menu when that row is absent. The player
    // re-renders the menu, so every tick re-checks the position.
    const RADIO_SELECTOR = '.setting-radio-mode, [role="menuitem"][label="라디오 모드"]';
    function place(item: HTMLElement, menu: Element) {
        const radio = menu.querySelector(RADIO_SELECTOR);
        if (radio) {
            if (item.nextElementSibling !== radio) radio.before(item);
        } else if (item.parentElement !== menu) menu.append(item);
    }
    function ensureItem() {
        const menu = document.querySelector<HTMLElement>(".pzp-pc .pzp-settings");
        if (!menu) return;
        const existing = document.querySelector<HTMLElement>(".pzp-pc .cb-pan-item");
        if (existing) return place(existing, menu);
        if (!document.getElementById("cb-pan-style")) {
            const style = document.createElement("style");
            style.id = "cb-pan-style";
            style.textContent = STYLE;
            document.head.append(style);
        }
        const item = document.createElement("div");
        item.className = "pzp-ui-setting-home-item cb-pan-item";
        item.setAttribute("role", "group");
        item.setAttribute("aria-label", "좌우 밸런스");
        item.innerHTML =
            '<div class="pzp-ui-setting-home-item__top"><div class="pzp-ui-setting-home-item__left"><span class="pzp-ui-setting-home-item__label">좌우 밸런스</span></div>' +
            '<div class="pzp-ui-setting-home-item__right"><span class="pzp-ui-setting-home-item__value cb-pan-value"></span></div></div>' +
            '<div class="cb-pan-row"><span>L</span><input type="range" class="cb-pan-range" min="-1" max="1" step="0.05" aria-label="좌우 밸런스" title="더블클릭: 가운데"><span>R</span></div>';
        const range = item.querySelector<HTMLInputElement>(".cb-pan-range")!;
        // The player treats clicks in the menu as menu navigation and arrow
        // keys as seek/volume shortcuts; the slider keeps both to itself.
        for (const type of ["click", "pointerdown", "mousedown", "keydown", "wheel"])
            item.addEventListener(type, (event) => event.stopPropagation());
        range.addEventListener("input", () => setPan(Number(range.value)));
        range.addEventListener("dblclick", () => {
            range.value = "0";
            setPan(0);
        });
        place(item, menu);
        syncUi();
    }

    const onPlayerPage = () => LIVE_RE.test(location.pathname) || VIDEO_RE.test(location.pathname);
    function tick() {
        try {
            if (!onPlayerPage()) return;
            if (cbSplit.frame && cbSplit.pan() !== pan) setPan(cbSplit.pan() ?? 0);
            ensureItem();
            // A swapped <video> (SPA navigation, quality reload) gets the
            // current pan once the context is running.
            if (cbAudio.running()) apply();
        } catch (error) {
            logger.error("pan tick failed {href}", { href: location.href, error });
        }
    }

    const resume = async () => {
        // A split column's pan can change later, so it starts audio anyway.
        if ((pan === 0 && !cbSplit.frame) || !onPlayerPage()) return;
        if (!(await cbAudio.context())) return;
        window.removeEventListener("pointerdown", resume, true);
        window.removeEventListener("keydown", resume, true);
        apply();
    };
    window.addEventListener("pointerdown", resume, true);
    window.addEventListener("keydown", resume, true);

    pageWindow.ChzzkBestPan = { get: () => pan, set: setPan, audio: cbAudio };
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
