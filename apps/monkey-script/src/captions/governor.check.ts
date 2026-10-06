// Run: node src/captions/governor.check.ts
// Drives governor.ts the way index.ts does: govern() every 500 ms with a
// "stall" verdict while stalled(), "ok" otherwise.
import { backoff, mayRunAt, notePlaying, noteWaiting, rebase, stalled, step, type Backoff } from "./governor.ts";

const TICK = 500;
// First tick at or after `from` where the model may start a run.
function firstRun(b: Backoff, from: number, until = from + 60000) {
    for (let t = from; t <= until; t += TICK) {
        const verdict = stalled(b, t) ? "stall" : "ok";
        step(b, verdict, t);
        if (verdict === "ok" && t >= mayRunAt(b)) return t;
    }
    return Infinity;
}

// Catches a 타임머신 jump or channel switch pausing captions ~15-20 s: the player's own rebuffer "waiting" counted as a model-caused stall, drove the factor to 8 and re-armed the 10 s pause (seen live: no hop for 20 s after the jump while the model stayed loaded).
{
    const b = backoff();
    const seekAt = 100000;
    rebase(b, seekAt);
    noteWaiting(b, seekAt + 50, true); // waiting while the seek is still running
    noteWaiting(b, seekAt + 800, false); // rebuffer before playback resumes
    notePlaying(b, seekAt + 1200);
    noteWaiting(b, seekAt + 1500, false); // inside the resume grace
    const t = firstRun(b, seekAt);
    if (t - seekAt > 1000) throw new Error(`first run ${t - seekAt} ms after a seek; want ≤ 1000 (factor ${b.factor}, pause until +${b.pauseUntil - seekAt})`);
    if (b.factor !== 1) throw new Error(`factor ${b.factor} after a seek's own rebuffer; want 1`);
}
// Catches the rebase dropping escalation earned at the old position too late: a pause armed before the jump must not carry over.
{
    const b = backoff();
    for (let t = 0; t < 4000; t += TICK) step(b, "drops", t);
    if (b.pauseUntil <= 4000) throw new Error("setup: drops should have armed the pause");
    rebase(b, 4000);
    notePlaying(b, 4300);
    const t = firstRun(b, 4000);
    if (t - 4000 > 1000) throw new Error(`old-position pause carried across the seek: first run +${t - 4000} ms`);
}
// Catches the fix muting real stalls: a waiting after playback resumed past the grace still backs off.
{
    const b = backoff();
    rebase(b, 0);
    notePlaying(b, 500);
    noteWaiting(b, 5000, false);
    if (!stalled(b, 5100)) throw new Error("a stall after playback resumed was ignored");
    if (firstRun(b, 5000) - 5000 < 5000) throw new Error("a real stall did not hold the model back");
}
console.log("governor.check: ok");
