// Back-off half of the playback governor in index.ts: how long the model
// waits after a run, and when it pauses outright. Kept free of DOM so
// governor.check.ts can drive it with synthetic clocks.
//
// A "waiting" event reads as a stall the model caused, except right after
// a seek, a 타임머신 rewind or forward, or a channel switch: there the
// player rebuffers at the new position on its own. Counting that rebuffer
// once doubled the factor to 8 and re-armed a 10 s pause on every tick of
// the 5 s stall window, so the first caption after a jump came ~15-20 s late.
export const STALL_MS = 5000;
const PAUSE_MS = 10000;
const MAX_FACTOR = 8;
// After a jump, waiting is ignored until playback resumes (plus this
// grace), or at most SETTLE_MAX_MS when no "playing" event comes.
const RESUME_GRACE_MS = 1000;
const SETTLE_MAX_MS = 10000;

export type Verdict = "ok" | "paused" | "stall" | "buffer" | "drops" | "jank";

export interface Backoff {
    factor: number; // wait after a run = run time × (factor − 1)
    pauseUntil: number;
    nextRunAt: number;
    cleanSince: number;
    waitingAt: number;
    settleUntil: number;
}

export const backoff = (): Backoff => ({ factor: 1, pauseUntil: 0, nextRunAt: 0, cleanSince: 0, waitingAt: -1e9, settleUntil: -1e9 });

// The playhead jumped or the source changed: escalation earned at the old
// position says nothing about the new one.
export function rebase(b: Backoff, now: number) {
    b.factor = 1;
    b.pauseUntil = 0;
    b.nextRunAt = 0;
    b.cleanSince = now;
    b.waitingAt = -1e9;
    b.settleUntil = now + SETTLE_MAX_MS;
}
export function noteWaiting(b: Backoff, now: number, seeking: boolean) {
    if (seeking || now < b.settleUntil) return;
    b.waitingAt = now;
}
export function notePlaying(b: Backoff, now: number) {
    b.settleUntil = Math.min(b.settleUntil, now + RESUME_GRACE_MS);
}
export const stalled = (b: Backoff, now: number) => now - b.waitingAt < STALL_MS;

export function step(b: Backoff, verdict: Verdict, now: number) {
    if (verdict === "drops" || verdict === "jank" || verdict === "buffer" || verdict === "stall") {
        b.cleanSince = now;
        if (b.factor >= MAX_FACTOR) b.pauseUntil = now + PAUSE_MS;
        b.factor = Math.min(MAX_FACTOR, b.factor * 2);
    } else if (verdict === "ok" && now - b.cleanSince > 5000) {
        b.factor = Math.max(1, b.factor * 0.85);
        b.cleanSince = now - 4000;
    }
}
export const mayRunAt = (b: Backoff) => Math.max(b.nextRunAt, b.pauseUntil);
