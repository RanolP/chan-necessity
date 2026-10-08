// For the ambient-light scenarios: the split page's light layer (its box against the grid's, the loop's stats, and each
// glow's box and blend mode), then per column whether its frame lets the light through (column, frame page and <main>
// backgrounds, transparent while on), Chzzk's view mode, the picture's box in page coordinates (each glow sits on one),
// and the frame's own loop, which now draws only the chat panel's canvas (a glow canvas in <main> is the old design).
(() => {
  const box = (e) => (e ? [e.getBoundingClientRect()].map((r) => [r.x, r.y, r.width, r.height].map(Math.round))[0] : null);
  const layer = document.querySelector(".cb-ambient-layer");
  return JSON.stringify({
    on: document.documentElement.classList.contains("cb-ambient-on"), host: window.ChzzkBestAmbient?.stats() ?? null,
    layer: box(layer), grid: box(document.querySelector(".cb-split-grid")),
    glows: [...(layer?.children ?? [])].map((c) => ({ box: box(c), blend: getComputedStyle(c).mixBlendMode })),
    cols: [...document.querySelectorAll(".cb-split-col")].map((c) => {
      const f = c.querySelector(".cb-split-body > iframe"), w = f?.contentWindow, d = f?.contentDocument, main = d?.querySelector("#layout-body main");
      const fr = f?.getBoundingClientRect(), vr = d?.querySelector("video")?.getBoundingClientRect();
      const bg = (e) => (e ? w.getComputedStyle(e).backgroundColor : null);
      return {
        col: c.dataset.id.slice(0, 8), colBg: getComputedStyle(c).backgroundColor, wide: /_is_large_/.test(main?.className ?? ""),
        video: fr && vr ? [fr.x + vr.x, fr.y + vr.y, vr.width, vr.height].map(Math.round) : null,
        frameOn: d?.documentElement.classList.contains("cb-ambient-on") ?? null,
        bg: d ? { html: bg(d.documentElement), body: bg(d.body), main: bg(main), aside: bg(d.querySelector("#aside-chatting")) } : null,
        mainCanvas: !!main?.querySelector(":scope > canvas"), frameLoop: w?.ChzzkBestAmbient?.stats() ?? null,
      };
    }),
  });
})()
