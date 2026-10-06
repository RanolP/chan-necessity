// Pure text helpers shared by the page side and the worker.
export const STRIP_RE = /[\s.,!?·…~"'“”‘’()\[\]-]/u;
export function normalizeWithMap(text: string): { chars: string[]; map: number[] } {
    const chars: string[] = [];
    const map: number[] = [];
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (STRIP_RE.test(ch)) continue;
        chars.push(ch.toLowerCase());
        map.push(i);
    }
    return { chars, map };
}
// Overlapping windows transcribe the same speech twice. Align the head
// of the new window's text against the tail of what is already shown
// (longest common run of non-space characters) and splice there,
// taking the new window's wording from the match on: the new window
// saw the boundary word whole, the old one may have cut it.
export function mergeTranscript(prev: string, next: string, tailChars = 60, headChars = 60): string {
    next = next.trim();
    if (!next) return prev;
    if (!prev) return next;
    const a = normalizeWithMap(prev);
    const b = normalizeWithMap(next);
    const aStart = Math.max(0, a.chars.length - tailChars);
    const bEnd = Math.min(b.chars.length, headChars);
    let best = 0;
    let bestA = -1;
    let bestB = -1;
    let row = new Array<number>(bEnd + 1).fill(0);
    for (let i = aStart; i < a.chars.length; i++) {
        const cur = new Array<number>(bEnd + 1).fill(0);
        for (let j = 0; j < bEnd; j++) {
            if (a.chars[i] !== b.chars[j]) continue;
            cur[j + 1] = row[j] + 1;
            if (cur[j + 1] > best) {
                best = cur[j + 1];
                bestA = i - best + 1;
                bestB = j - best + 1;
            }
        }
        row = cur;
    }
    if (best < Math.min(3, b.chars.length)) return `${prev} ${next}`;
    // The new window repeats nothing new: keep what is shown.
    if (best === b.chars.length) return prev;
    return (prev.slice(0, a.map[bestA]) + next.slice(b.map[bestB])).trim();
}
// Streaming context bookkeeping. Each block is encoded on its own, so
// its attention starts at the block's first frame, a closed block's features never
// change and the oldest block can be cut off without touching the rest.
// `st` = { open: Float32Array, closed: [{ pcm, af, mark }], tokens, histIds }.
// A block that closes records how many tokens the text had then (minus
// the rollback); when the block falls out of the kept range those
// tokens leave the forced prefix and are returned; the caller decides
// when they enter histIds.
// `af` holds the block's encoder output once computed (the worker keeps it
// on the GPU); closeBlocks never reads it.
export interface ClosedBlock<F = unknown> {
    pcm: Float32Array;
    af: F | null;
    mark: number;
}
export interface BlockState<F = unknown> {
    open: Float32Array;
    closed: ClosedBlock<F>[];
    tokens: number[];
    histIds: number[];
}
// Byte-level BPE splits one Hangul syllable (3 UTF-8 bytes) across
// tokens, so a slide is pulled back to the last token boundary that
// ends a whole character: the slid text and the text left behind both
// decode without U+FFFD halves.
export function closeBlocks<F>(st: BlockState<F>, block: number, keep: number, rollback: number, bytesOf: (ids: readonly number[]) => Uint8Array): number[] {
    const slid: number[] = [];
    while (st.open.length >= block) {
        st.closed.push({ pcm: st.open.slice(0, block), af: null, mark: Math.max(0, st.tokens.length - rollback) });
        st.open = st.open.slice(block);
        while (st.closed.length > keep) {
            const gone = st.closed.shift()!;
            const n = charBoundary(st.tokens, Math.min(gone.mark, Math.max(0, st.tokens.length - rollback)), bytesOf);
            slid.push(...st.tokens.splice(0, n));
            for (const b of st.closed) b.mark = Math.max(0, b.mark - n);
        }
    }
    return slid;
}
// Largest n' <= n such that tokens[0, n') decodes to whole characters.
export function charBoundary(tokens: readonly number[], n: number, bytesOf: (ids: readonly number[]) => Uint8Array): number {
    const all = bytesOf(tokens);
    let at = bytesOf(tokens.slice(0, n)).length;
    while (n > 0 && utf8Cut(all, at) !== at) at -= bytesOf([tokens[--n]]).length;
    return n;
}
// Byte offset at or before `at` that starts a UTF-8 character.
export function utf8Cut(bytes: Uint8Array, at: number): number {
    at = Math.max(0, Math.min(at, bytes.length));
    while (at > 0 && at < bytes.length && (bytes[at] & 0xc0) === 0x80) at--;
    return at;
}

// A decoder stuck in a loop repeats one block of p tokens: "그그그…"
// (p = 1), "2020…" (p = 2), a whole phrase (p ≈ 8-20). The streaming
// decoder forces the previous hop's text and puts slid history in the
// prompt, so a loop spans history, forced text and new tokens and is
// only visible on the joined sequence.
// Fires when the tail is one p-block repeated over max(LOOP_MIN, 2p)
// tokens: 12 tokens for p <= 6 keeps the old short-period floor, so
// speech that repeats a short word ("네 네", "맞아요 맞아요") never
// fires; from p = 6 on it needs two whole copies of the phrase, the old
// equal-halves rule extended to every period up to LOOP_MAX_PERIOD (24 ≈
// one long Korean sentence). The smallest period that fires wins, so
// collapsing keeps the shortest repeating unit.
const LOOP_MIN = 12;
const LOOP_MAX_PERIOD = 24;
// Tokens a caller must keep in front of new tokens for loopStop to see a
// loop that started before them.
export const LOOP_SPAN = 2 * LOOP_MAX_PERIOD;
// The trailing loop of `t`: its period and where the periodic run starts.
export function trailingLoop(t: readonly number[]): { period: number; start: number } | null {
    const n = t.length;
    for (let p = 1; p <= LOOP_MAX_PERIOD && 2 * p <= n; p++) {
        let s = n - p;
        while (s > 0 && t[s - 1] === t[s - 1 + p]) s--;
        if (n - s >= Math.max(LOOP_MIN, 2 * p)) return { period: p, start: s };
    }
    return null;
}
export const loopStop = (gen: readonly number[]): boolean => trailingLoop(gen) !== null;
// `fixed` + `t` with any trailing loop cut to one copy, returned as the
// new `t`: `fixed` is text the user has already seen and is never cut.
// When `fixed` already holds a whole copy, every looping token of `t`
// goes.
export function trimLoop(fixed: readonly number[], t: readonly number[]): number[] {
    const loop = trailingLoop([...fixed, ...t]);
    if (!loop) return t.slice();
    return t.slice(0, Math.max(0, loop.start + loop.period - fixed.length));
}
