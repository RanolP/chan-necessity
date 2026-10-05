import { getLogger } from "../shared/logtape.ts";

const logger = getLogger(["auto-claim-logs"]);

// ---- 통나무 파워 auto-collect ------------------------------------------
// Matching and the React-handler click follow 치지직 통나무파워 자동 벌목
// v1.7.1 by 떱_ (MIT), https://greasyfork.org/scripts/585512. The scope to
// the chat aside and the ranking-button exclusion follow Cheese Spanner
// (https://github.com/neder2/Cheese-spanner): the weekly 통나무 파워 ranking
// controls sit in the same slot and carry the same words.
// Claims go through the on-screen button only, never the claim API.
(() => {
    const POWER_RE = /통나무\s*파워/;
    const AMOUNT_RE = /[\d,]+\s*(?:개\s*)?받기/;
    const RETRY_COOLDOWN_MS = 10_000;
    const SETTLE_MS = 120;
    const lastAttemptAt = new WeakMap<HTMLElement, number>();

    function isClaimButton(el: HTMLElement) {
        if (!el.closest("#aside-chatting")) return false;
        if (
            el.hasAttribute("aria-expanded") ||
            el.hasAttribute("aria-haspopup") ||
            /ranking|arrow_button/.test(el.className)
        ) {
            return false;
        }
        const text = el.textContent.replace(/\s+/g, " ");
        return POWER_RE.test(text) && AMOUNT_RE.test(text);
    }

    function isVisibleAndEnabled(el: HTMLElement) {
        if (
            !el.isConnected ||
            (el as HTMLButtonElement).disabled ||
            el.getAttribute("aria-disabled") === "true" ||
            el.closest('[hidden], [aria-hidden="true"]')
        ) {
            return false;
        }
        const style = getComputedStyle(el);
        return (
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            el.getClientRects().length > 0
        );
    }

    type ReactProps = Record<string, ((event: unknown) => void) | undefined>;
    interface ReactFiber {
        memoizedProps: ReactProps | null;
        return: ReactFiber | null;
    }

    // Chzzk binds the handler through React; calling it directly works
    // even when a synthetic .click() is filtered by isTrusted checks.
    function invokeReactHandler(el: HTMLElement) {
        const names = ["onClick", "onPointerDown", "onMouseDown"];
        const event = {
            type: "click",
            target: el,
            currentTarget: el,
            nativeEvent: new MouseEvent("click", { bubbles: true }),
            preventDefault() {},
            stopPropagation() {},
            persist() {},
        };
        const keys = Object.getOwnPropertyNames(el);
        const propsKey = keys.find((k) => k.startsWith("__reactProps$"));
        const fiberKey = keys.find((k) => k.startsWith("__reactFiber$"));
        const fields = el as unknown as Record<string, unknown>;
        let props = (propsKey ? fields[propsKey] : null) as ReactProps | null;
        let fiber = (fiberKey ? fields[fiberKey] : null) as ReactFiber | null;
        for (let depth = 0; depth < 6; depth++) {
            for (const name of names) {
                if (typeof props?.[name] === "function") {
                    props[name](event);
                    return true;
                }
            }
            if (!fiber) break;
            props = fiber.memoizedProps;
            fiber = fiber.return;
        }
        return false;
    }

    function claim(el: HTMLElement) {
        if (!isClaimButton(el) || !isVisibleAndEnabled(el)) return;
        const now = Date.now();
        if (now - (lastAttemptAt.get(el) ?? 0) < RETRY_COOLDOWN_MS) return;
        lastAttemptAt.set(el, now);
        if (!invokeReactHandler(el)) el.click();
        logger.info`통나무 파워 claimed: ${el.textContent.trim()}`;
    }

    function scan() {
        for (const el of document.querySelectorAll<HTMLElement>(
            '#aside-chatting button, #aside-chatting [role="button"]'
        )) {
            if (isClaimButton(el)) setTimeout(() => claim(el), SETTLE_MS);
        }
    }

    let scheduled = false;
    new MutationObserver(() => {
        if (scheduled) return;
        scheduled = true;
        setTimeout(() => {
            scheduled = false;
            scan();
        }, 250);
    // At document-start <html> may not exist yet; the document node does.
    }).observe(document, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true,
        attributeFilter: ["disabled", "aria-disabled", "hidden"],
    });
})();
