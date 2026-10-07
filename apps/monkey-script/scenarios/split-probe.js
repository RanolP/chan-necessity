// Split-host state for the split-view scenarios: address, columns, header
// controls, each column frame's layout and playback, and whether any
// player-page feature landed in the top shell.
(() => {
  const box = (e) => (e ? [e.getBoundingClientRect()].map((r) => [r.x, r.y, r.width, r.height].map(Math.round))[0] : null);
  const frames = [...document.querySelectorAll(".cb-split-body > iframe")].map((f) => {
    const w = f.contentWindow, d = f.contentDocument, v = d?.querySelector("video");
    return { col: f.dataset.cbCol.slice(0, 8), mark: w?.__splitMark ?? null, frame: box(f), video: box(v), aside: box(d?.querySelector("#aside-chatting")), t: v ? +v.currentTime.toFixed(1) : null, paused: v?.paused };
  });
  return JSON.stringify({
    url: location.pathname + location.search, title: document.title, sameDoc: window.__splitDoc === performance.timeOrigin,
    cols: window.ChzzkBestSplit?.stats().cols.map((id) => id.slice(0, 8)), count: document.querySelector(".cb-split-count")?.textContent,
    exitHidden: [...document.querySelectorAll(".cb-split-bar .cb-split-btn")].find((b) => b.textContent === "나가기")?.hidden,
    closeHidden: [...document.querySelectorAll(".cb-split-col .cb-split-icon")].map((b) => b.hidden),
    topPlayer: { videos: document.querySelectorAll("video").length, stt: !!document.querySelector(".cb-stt-button, .cb-stt-lines"), bookmarks: !!document.querySelector('[class*="cb-bm"]'), pan: !!document.querySelector('[class*="cb-pan"]') },
    frames,
  });
})()
