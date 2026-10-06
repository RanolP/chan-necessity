// Stands in for the STT worker where WebGPU has no adapter (headless
// Chrome), so the page side (hop scheduling, seek reset, lines, history)
// runs for real. Each streaming hop adds one word naming its context
// ("c2h5": context 2, hop 5); a hop sent with reset starts a new context,
// so text left over from before a seek shows up as a stale context number.
(() => {
  const Real = window.Worker;
  let ctx = 0;
  let hop = 0;
  let words = [];
  class FakeWorker {
    constructor(url, opts) {
      // The STT worker is the only module worker built from a blob; the
      // player's own workers must stay real.
      if (!String(url).startsWith("blob:") || opts?.type !== "module") return new Real(url, opts);
      this.onmessage = null;
      this.onerror = null;
    }
    reply(m, delay = 30) {
      setTimeout(() => this.onmessage?.({ data: m }), delay);
    }
    postMessage(m) {
      if (m.type === "load") this.reply({ type: "ready", ms: 1 });
      else if (m.type === "stream") {
        if (m.reset || !ctx) (ctx++, (hop = 0), (words = []));
        hop++;
        if (!m.quiet) words.push(`c${ctx}h${hop}`);
        this.reply({ type: "result", id: m.id, stream: true, decoded: true, conf: words.join(" "), tent: "", ids: words.map((_, i) => i), totalMs: 40, speech: true, speechSec: 1, vadMs: null, newSec: 1 }, 40);
      } else if (m.type === "run") this.reply({ type: "result", id: m.id, text: `c${ctx}h${++hop}`, totalMs: 40, speech: true, speechSec: 1, vadMs: null }, 40);
    }
    terminate() {}
    addEventListener() {}
    removeEventListener() {}
  }
  window.Worker = FakeWorker;
  window.__fakeStt = () => ({ ctx, hop, words: words.slice(-6) });
})();
