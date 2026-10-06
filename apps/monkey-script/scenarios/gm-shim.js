// Userscript-manager shim for agent-browser runs: lets dist/chan-necessity.user.js
// run as a plain init script, with subtitles on and the 1.7B model.
(() => {
  const P = "gmshim:";
  window.unsafeWindow = window;
  window.GM_getValue = (k, d) => { const v = localStorage.getItem(P + k); return v === null ? d : JSON.parse(v); };
  window.GM_setValue = (k, v) => localStorage.setItem(P + k, JSON.stringify(v));
  window.GM_xmlhttpRequest = (o) => {
    fetch(o.url, { method: o.method || "GET", headers: o.headers, body: o.data })
      .then(async (r) => {
        const response = o.responseType === "arraybuffer" ? await r.arrayBuffer() : o.responseType === "blob" ? await r.blob() : o.responseType === "json" ? await r.json() : await r.text();
        o.onload?.({ status: r.status, response, responseText: typeof response === "string" ? response : "", finalUrl: r.url, responseHeaders: [...r.headers].map(([k, v]) => `${k}: ${v}`).join("\r\n") });
      })
      .catch((e) => o.onerror?.(e));
    return { abort() {} };
  };
  if (localStorage.getItem(P + "stt.enabled") === null) {
    GM_setValue("stt.enabled", true);
    GM_setValue("stt.model", "1.7B");
  }
})();
