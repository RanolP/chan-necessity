// Clip editor on Firefox: the /clip-editor popup refuses Firefox by a
// react-device-detect UA-string check (not feature detection), while the
// clip itself is encoded server side. Only that popup sees a Chrome UA, and
// only the userAgent getter changes, since UAParser's sync path reads
// nothing else and Firefox has no navigator.userAgentData to contradict it.
(() => {
    "use strict";
    if (!location.pathname.startsWith("/clip-editor")) return;
    const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;
    const real = pageWindow.navigator.userAgent;
    if (!/Firefox\//.test(real)) return;
    const platform = /Windows/.test(real)
        ? "Windows NT 10.0; Win64; x64"
        : /Mac OS X/.test(real)
          ? "Macintosh; Intel Mac OS X 10_15_7"
          : "X11; Linux x86_64";
    const chrome = `Mozilla/5.0 (${platform}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36`;
    Object.defineProperty(pageWindow.Navigator.prototype, "userAgent", {
        configurable: true,
        enumerable: true,
        get: () => chrome,
    });
})();
