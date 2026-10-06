// One instance per page: the installed userscript and a dev build injected
// over CDP can both land in the same tab, which doubled every player-bar
// button and ran two caption engines on one GPU.
import pkg from "../package.json" with { type: "json" };
import { claimInstance } from "./shared/instance-guard.ts";

const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;
const running = claimInstance(pageWindow, pkg.version);
if (running) {
    console.warn(`[chan-necessity] ${pkg.version} not starting: ${running} already runs on this page`);
} else {
    // Feature modules install themselves on import, so they load only past the guard.
    void import("./features.ts");
}
