(() => { const s = ChzzkBestStt, v = document.querySelector(".pzp-pc video"), c = s.media.chunks;
  return JSON.stringify({ phase: s.state.phase, status: s.state.statusText, ct: +v.currentTime.toFixed(2), seekable: v.seekable.length ? [+v.seekable.start(0).toFixed(1), +v.seekable.end(0).toFixed(1)] : null,
    chunks: c.length, span: c.length ? [+c[0].start.toFixed(1), +c.at(-1).end.toFixed(1)] : null, lastEndMedia: s.state.lastEndMedia, due: s.state.due.length,
    bySource: s.stats.bySource, hops: s.stats.stream.hops, gate: s.gov.open, reasons: s.gov.reasons, fake: window.__fakeStt?.(), text: s.state.text.slice(-60),
    lines: [...document.querySelectorAll(".cb-stt-lines > div:not(.cb-stt-measure)")].map((d) => d.textContent) }); })()
