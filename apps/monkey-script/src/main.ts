// One instance per page: the installed userscript and a dev build injected
// over CDP can both land in the same tab, which doubled every player-bar
// button and ran two caption engines on one GPU.
import pkg from "../package.json" with { type: "json" };
import { claimInstance } from "./shared/instance-guard.ts";

const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;
function start() {
    const running = claimInstance(pageWindow, pkg.version);
    if (running) {
        console.warn(`[chan-necessity] ${pkg.version} not starting: ${running} already runs on this page`);
    } else {
        // Feature modules install themselves on import, so they load only past the guard.
        void import("./features.ts");
    }
}
// An injector without the header's @match (an agent-browser init script) also
// runs the script in a split column's initial about:blank document. Chromium
// keeps that window when the frame loads its same-origin /live page and may
// not inject again, so starting in the blank document would put the column's
// split CSS there and lose it. Wait for the real page in the same window.
const chzzk = () => location.origin === "https://chzzk.naver.com";
if (chzzk()) start();
else if (location.href === "about:blank" && window !== window.top) {
    const wait = setInterval(() => {
        if (!chzzk()) return;
        clearInterval(wait);
        start();
    }, 20);
}
