// ---- 플레이어 하단 바 버튼 ---------------------------------------------
// The one owner of the buttons features add to the player's bottom-right
// bar. A feature declares its button; this module builds it in the
// player's own button markup, keeps all registered buttons together at the
// start of the bar in `order`, puts them back when the player re-renders
// the bar, and hides them with the player's controls overlay.

export interface PlayerButtonSpec {
    /** Class that identifies this button, e.g. "cb-bm-button". */
    className: string;
    /** Tooltip text and aria-label. */
    label: string;
    /** Inline SVG; give it the native 36x36 size and `pzp-ui-icon__svg` class. */
    icon: string;
    /** Lower comes first (leftmost). */
    order: number;
    onClick(event: MouseEvent): void;
    onContextMenu?(event: MouseEvent): void;
    /** Pages the button belongs on; absent means every player page. */
    when?(): boolean;
    /** Drives aria-pressed and the accent icon color; absent means not a toggle. */
    pressed?(): boolean;
}

export interface PlayerButton {
    readonly element: HTMLButtonElement;
    /** Re-applies `when` and `pressed` now instead of on the next DOM change. */
    refresh(): void;
}

const BAR = ".pzp-pc__bottom-buttons-right";
const MARK = "cb-bar-button";
// The player sizes each native button by its own class to a 36px box, and
// fades them with the controls overlay: .pzp-pc loses .pzp-pc--controls
// when the mouse leaves, and each button animates opacity alone, staying
// visible and clickable. Ours take the same box and toggle only opacity, so
// the player's own `.pzp-pc__bottom-buttons .pzp-pc-ui-button` transition
// fades them in step; a visibility toggle would snap them away mid-fade.
const STYLE = `
    .${MARK} { width: 36px; height: 36px; }
    .pzp-pc:not(.pzp-pc--controls) .${MARK} { opacity: 0; }
    .${MARK}[aria-pressed="true"] .pzp-ui-icon { color: #00ffa3; }
`;

const entries: { spec: PlayerButtonSpec; element: HTMLButtonElement }[] = [];
let styled = false;
let observing = false;

function sync() {
    if (!styled && document.head) {
        const style = document.createElement("style");
        style.id = "cb-bar-button-style";
        style.textContent = STYLE;
        document.head.append(style);
        styled = true;
    }
    const bar = document.querySelector(BAR);
    let prev: Element | null = null;
    for (const { spec, element } of entries) {
        if (!bar || (spec.when && !spec.when())) {
            if (element.isConnected) element.remove();
            continue;
        }
        if (spec.pressed) {
            const pressed = String(spec.pressed());
            if (element.getAttribute("aria-pressed") !== pressed) element.setAttribute("aria-pressed", pressed);
        }
        // Idempotent when already in place, so our own insertions settle
        // the observer instead of re-triggering it forever.
        if (element.parentElement !== bar || element.previousElementSibling !== prev) {
            bar.insertBefore(element, prev ? prev.nextSibling : bar.firstChild);
        }
        prev = element;
    }
}

export function registerPlayerButton(spec: PlayerButtonSpec): PlayerButton {
    const element = document.createElement("button");
    element.type = "button";
    element.className = `${spec.className} ${MARK} pzp-button pzp-pc-ui-button`;
    element.setAttribute("aria-label", spec.label);
    const tooltip = document.createElement("span");
    tooltip.className = "pzp-button__tooltip pzp-button__tooltip--top";
    tooltip.textContent = spec.label;
    const icon = document.createElement("span");
    icon.className = "pzp-ui-icon";
    icon.innerHTML = spec.icon;
    element.append(tooltip, icon);
    element.addEventListener("click", (event) => {
        event.stopPropagation();
        spec.onClick(event);
    });
    const onContextMenu = spec.onContextMenu;
    if (onContextMenu) {
        element.addEventListener("contextmenu", (event) => {
            event.preventDefault();
            event.stopPropagation();
            onContextMenu(event);
        });
    }

    const at = entries.findIndex((e) => e.spec.order > spec.order);
    entries.splice(at < 0 ? entries.length : at, 0, { spec, element });

    if (!observing) {
        observing = true;
        new MutationObserver(sync).observe(document, { subtree: true, childList: true });
    }
    sync();
    return { element, refresh: sync };
}
