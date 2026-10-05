import { getLogger } from "../shared/logtape.ts";

const logger = getLogger(["deny-grid"]);

// Based on https://github.com/refracta/FUCK-CHZZK-GRID-CHROME (MIT)
// Copyright (c) 2025 bass9030
//
// The extension redirects 480p playlist requests to 1080p with a
// declarativeNetRequest rule. A userscript has no such API, so the same
// rewrite happens by wrapping fetch and XMLHttpRequest in the page world.
// Like the extension, the bypass is active only on Windows.

(() => {
    const IS_WINDOWS = /Windows/i.test(
        (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ?? navigator.userAgent
    );
    if (!IS_WINDOWS) {
        logger.info("non-Windows platform detected; bypass disabled");
        return;
    }

    // ---- network: rules.json -------------------------------------------
    const PLAYLIST_RE = /(.*)480p(.*\.m3u8.*)/;

    function rewriteUrl(url: string) {
        return url.replace(PLAYLIST_RE, "$11080p$2");
    }

    // With any @grant, Violentmonkey runs the script against a proxy
    // whose property writes stay local, so `window.fetch = ...` would no
    // longer reach the page. unsafeWindow is the page's real window.
    const pageWindow =
        typeof unsafeWindow === "undefined" ? window : unsafeWindow;

    const originalFetch = pageWindow.fetch;
    pageWindow.fetch = function (this: unknown, input: RequestInfo | URL, init?: RequestInit) {
        if (typeof input === "string" || input instanceof pageWindow.URL) {
            input = rewriteUrl(String(input));
        } else if (input instanceof pageWindow.Request) {
            const rewritten = rewriteUrl(input.url);
            if (rewritten !== input.url) {
                input = new pageWindow.Request(rewritten, input);
            }
        }
        return originalFetch.call(this, input, init);
    };

    const XHR = pageWindow.XMLHttpRequest;
    const originalOpen = XHR.prototype.open;
    XHR.prototype.open = function (this: XMLHttpRequest, method: string, url: string | URL, ...rest: unknown[]) {
        return (originalOpen as (...args: unknown[]) => void).call(this, method, rewriteUrl(String(url)), ...rest);
    };

    // ---- UI: inject.js ---------------------------------------------------
    const LIVE_URL_RE =
        /^https:\/\/chzzk\.naver\.com\/live\/[0-9a-z]+(?:[/?#].*)?$/i;
    const QUALITY_LIST_SELECTOR =
        "ul.pzp-setting-quality-pane__list-container > li";
    const QUALITY_CONTAINER_SELECTOR =
        "ul.pzp-setting-quality-pane__list-container";
    const SELECTED_QUALITY_SELECTOR =
        `${QUALITY_LIST_SELECTOR}.pzp-ui-setting-pane-item--checked`;
    const SETTING_BUTTON_SELECTOR =
        "button.pzp-setting-button[command='SettingCommands.Toggle']";

    let previousQualityText = "";
    let qualityListElement: HTMLUListElement | null = null;
    let previousRestoreAttempt = 0;

    function changeText() {
        const qualityItems = document.querySelectorAll(QUALITY_LIST_SELECTOR);
        if (qualityItems.length === 0) return;

        const qualityElement = (Array.from(qualityItems) as HTMLElement[]).find((element) =>
            element.innerText.trim().includes("480p")
        );
        const qualityLabel = qualityElement?.querySelector<HTMLElement>(
            "li > div:nth-child(2) > span > div"
        );

        if (
            qualityLabel &&
            !qualityLabel.innerText.trim().includes("ChzzkBest")
        ) {
            qualityLabel.innerHTML =
                '<span class="pzp-pc-ui-setting-quality-item__prefix">1080p&nbsp;<div class="pzp-ui-track-badge"><em style="vertical-align:super;" class="pzp-ui-track-badge__badge">with ChzzkBest</em> <!----></div></span>';
        }

        const currentQualityTextElement = document.querySelector<HTMLElement>(
            "div.pzp-setting-intro-quality > div > div:last-child > span.pzp-ui-setting-home-item__value"
        );
        const selectedQualityTextElement = document.querySelector<HTMLElement>(
            SELECTED_QUALITY_SELECTOR
        );

        if (!currentQualityTextElement || !selectedQualityTextElement) return;

        const selectedQualityText = selectedQualityTextElement.innerText.trim();
        if (selectedQualityText.includes("ChzzkBest")) {
            if (
                !currentQualityTextElement.innerText
                    .trim()
                    .includes("ChzzkBest")
            ) {
                currentQualityTextElement.innerHTML =
                    '1080p <div class="pzp-ui-track-badge"><em style="vertical-align:super;" class="pzp-ui-track-badge__badge">with ChzzkBest</em> <!----></div>';
                previousQualityText = currentQualityTextElement.innerText;
            }
            return;
        }

        if (previousQualityText !== currentQualityTextElement.innerText) {
            currentQualityTextElement.innerText = selectedQualityText;
            previousQualityText = currentQualityTextElement.innerText;
        }
    }

    // innerText of a hidden <li> is its textContent ("720pHD 60fps"), of a
    // shown one the rendered lines ("720p\nHD\n60fps"), so saved and live
    // labels are compared by this key, never by raw text. 0.3.2 compared raw
    // text and reopened the settings panel every second whenever they
    // differed. Keep in sync with quality-key.check.mjs.
    const CHZZKBEST_QUALITY = "1080p with ChzzkBest";
    function qualityKey(text: string | null | undefined) {
        const normalized = String(text ?? "").replace(/\s+/g, " ").trim();
        // "FUCK GRID" is the pre-0.1.1 label of the same menu item.
        if (/ChzzkBest|FUCK GRID/i.test(normalized)) return CHZZKBEST_QUALITY;
        return normalized.match(/^\d+p/)?.[0] ?? normalized;
    }

    const storedQuality = localStorage.getItem("quality-text");
    if (storedQuality != null && storedQuality !== qualityKey(storedQuality)) {
        localStorage.setItem("quality-text", qualityKey(storedQuality));
    }

    // One attempt per player instance: the quality list element and the
    // video source both change when the player is rebuilt.
    let restoredFor: { list: Element | null; src: string | null } = { list: null, src: null };
    let restoring = false;

    function restoreQuality() {
        const now = Date.now();
        if (now - previousRestoreAttempt < 1000) return;
        previousRestoreAttempt = now;

        const list = document.querySelector<HTMLUListElement>(QUALITY_CONTAINER_SELECTOR);
        const videoElement = document.querySelector<HTMLVideoElement>(
            "video.webplayer-internal-video"
        );
        const settingButton = document.querySelector<HTMLButtonElement>(SETTING_BUTTON_SELECTOR);
        const settingIntro = document.querySelector<HTMLElement>(
            "div.pzp-setting-intro-quality"
        );
        if (
            !list ||
            !videoElement ||
            videoElement.readyState < 3 ||
            !settingButton ||
            !settingIntro
        ) {
            return;
        }
        const src = videoElement.currentSrc || videoElement.src;
        if (restoredFor.list === list && restoredFor.src === src) return;
        // The user has the panel open; touching it now would yank them
        // into the quality pane, so wait until they close it.
        if (settingButton.getAttribute("aria-expanded") === "true") return;
        const wanted = qualityKey(localStorage.getItem("quality-text") ?? "360p");
        const items = Array.from(list.querySelectorAll(":scope > li"));
        const current = items.find((li) =>
            li.classList.contains("pzp-ui-setting-pane-item--checked")
        );
        if (!current) return;
        restoredFor = { list, src };
        const target = items.find((li) => qualityKey(li.textContent) === wanted);
        if (!target || current === target) return;

        // The player ignores input on a hidden pane, so the panel has to be
        // open while the item is picked; it is closed again either way.
        const later = (ms: number) => new Promise((r) => setTimeout(r, ms));
        (async () => {
            settingButton.click();
            await later(300);
            settingIntro.click();
            await later(400);
            // Only a bubbling Enter reaches the player's handler; .click()
            // is ignored. The list's own keydown listener would save the
            // still-checked old item, hence the flag.
            restoring = true;
            target.dispatchEvent(
                new KeyboardEvent("keydown", { key: "Enter", bubbles: true })
            );
            restoring = false;
            changeText();
            await later(300);
            if (settingButton.getAttribute("aria-expanded") === "true") {
                settingButton.click();
            }
        })();
    }

    function saveQuality() {
        if (restoring) return;
        const qualityList = document.querySelector(SELECTED_QUALITY_SELECTOR);
        if (!qualityList) return;

        localStorage.setItem("quality-text", qualityKey(qualityList.textContent));
    }

    function callback() {
        if (document.readyState !== "complete") return;
        if (!LIVE_URL_RE.test(location.href)) return;

        changeText();
        restoreQuality();

        const nextQualityListElement = document.querySelector<HTMLUListElement>(
            QUALITY_CONTAINER_SELECTOR
        );
        if (
            nextQualityListElement &&
            nextQualityListElement !== qualityListElement
        ) {
            qualityListElement?.removeEventListener("click", saveQuality);
            qualityListElement?.removeEventListener("keydown", saveQuality);
            qualityListElement = nextQualityListElement;

            qualityListElement.addEventListener("click", saveQuality);
            qualityListElement.addEventListener("keydown", saveQuality);
        }
    }

    // The extension re-injects on SPA navigation; here one observer on
    // <body> covers every page, and callback() gates on the live URL.
    function startObserver() {
        new MutationObserver(callback).observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["class"],
        });
        callback();
    }

    if (document.body) {
        startObserver();
    } else {
        document.addEventListener("DOMContentLoaded", startObserver, {
            once: true,
        });
    }
})();
