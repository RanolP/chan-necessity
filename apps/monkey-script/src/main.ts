// One instance per page: the installed userscript and a dev build injected
// over CDP can both land in the same tab, which doubled every player-bar
// button and ran two caption engines on one GPU. The marker lives on the DOM
// because userscript sandboxes may not share `window`.
import pkg from "../package.json" with { type: "json" };

const root = document.documentElement;
const running = root.dataset.chanNecessity;
if (running) {
    console.warn(`[chan-necessity] ${pkg.version} not starting: ${running} already runs on this page`);
} else {
    root.dataset.chanNecessity = pkg.version;
    // Feature modules install themselves on import, so they load only past the guard.
    void import("./features.ts");
}
