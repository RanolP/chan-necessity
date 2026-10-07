// Per column frame: Chzzk's view mode (wide = main has _is_large_), the player, the picture, control bar,
// caption overlay and chat boxes, and whether the overlays sit on the picture. Wide mode once made the
// player 100vh tall: the <video> element filled it but its picture letterboxed mid-frame, while controls
// and captions hugged the frame's bottom edge, off the picture. So "picture" is the letterboxed frame,
// not the element box, which covered the overlays either way.
(() => {
  const box = (e) => (e ? [e.getBoundingClientRect()].map((r) => [r.x, r.y, r.width, r.height].map(Math.round))[0] : null);
  const picture = (v) => {
    const [x, y, w, h] = box(v);
    const s = Math.min(w / v.videoWidth, h / v.videoHeight);
    return [x + (w - v.videoWidth * s) / 2, y + (h - v.videoHeight * s) / 2, v.videoWidth * s, v.videoHeight * s].map(Math.round);
  };
  const inside = (a, b) => !!a && !!b && a[1] >= b[1] - 1 && a[1] + a[3] <= b[1] + b[3] + 1;
  return JSON.stringify(
    [...document.querySelectorAll(".cb-split-body > iframe")].map((f) => {
      const d = f.contentDocument, v = d.querySelector("video");
      const pic = v?.videoWidth ? picture(v) : null, bar = box(d.querySelector(".pzp-pc__bottom")), stt = box(d.querySelector(".cb-stt-overlay"));
      return {
        col: f.dataset.cbCol.slice(0, 8), frameW: Math.round(f.getBoundingClientRect().width),
        wide: /_is_large_/.test(d.querySelector("#layout-body main")?.className ?? ""),
        player: box(d.querySelector("#layout-body main > :first-child > :first-child")), video: box(v), picture: pic, bar, stt,
        aside: box(d.querySelector("#aside-chatting")),
        barOnPicture: inside(bar, pic), sttOnPicture: stt ? inside(stt, pic) : null,
      };
    }),
  );
})()
