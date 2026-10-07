// Per column frame for the ambient-light scenario: Chzzk's view mode, <main>, the picture's box,
// the glow canvas's box, and the loop's own stats (on, shown, fps over the last 2 s).
(() => {
  const box = (e) => (e ? [e.getBoundingClientRect()].map((r) => [r.x, r.y, r.width, r.height].map(Math.round))[0] : null);
  return JSON.stringify(
    [...document.querySelectorAll(".cb-split-body > iframe")].map((f) => {
      const w = f.contentWindow, d = f.contentDocument, main = d.querySelector("#layout-body main");
      return {
        col: f.dataset.cbCol.slice(0, 8), frameW: Math.round(f.getBoundingClientRect().width), wide: /_is_large_/.test(main?.className ?? ""),
        main: box(main), video: box(d.querySelector("video")), canvas: box(d.querySelector("canvas.cb-ambient")), ambient: w.ChzzkBestAmbient?.stats() ?? null,
      };
    }),
  );
})()
