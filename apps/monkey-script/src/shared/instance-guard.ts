// Keyed on the page's own window rather than the DOM: at @run-at
// document-start `document.documentElement` can still be null, and the page
// window is the one object both the installed userscript (via unsafeWindow)
// and a dev build injected over CDP (main world) can see.
const KEY = "__chanNecessity";

/** Returns the version already running on `host`, or claims `host` for `version` and returns undefined. */
export function claimInstance(host: object, version: string): string | undefined {
    const slot = host as Record<string, unknown>;
    const running = slot[KEY];
    if (typeof running === "string") return running;
    slot[KEY] = version;
    return undefined;
}
