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
// Streaming context bookkeeping. The encoder attends within 8 s blocks
// anchored at the start of its input, so a closed block's features never
// change and the oldest block can be cut off without touching the rest.
// `st` = { open: Float32Array, closed: [{ pcm, af, mark }], tokens, histIds }.
// A block that closes records how many tokens the text had then (minus
// the rollback); when the block falls out of the kept range those
// tokens leave the forced prefix and become history.
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
export function closeBlocks<F>(st: BlockState<F>, block: number, keep: number, rollback: number, histKeep: number): number[] {
    const slid: number[] = [];
    while (st.open.length >= block) {
        st.closed.push({ pcm: st.open.slice(0, block), af: null, mark: Math.max(0, st.tokens.length - rollback) });
        st.open = st.open.slice(block);
        while (st.closed.length > keep) {
            const gone = st.closed.shift()!;
            const n = Math.min(gone.mark, Math.max(0, st.tokens.length - rollback));
            slid.push(...st.tokens.splice(0, n));
            for (const b of st.closed) b.mark = Math.max(0, b.mark - n);
        }
    }
    if (slid.length) st.histIds = [...st.histIds, ...slid].slice(-histKeep);
    return slid;
}
// Byte offset at or before `at` that starts a UTF-8 character.
export function utf8Cut(bytes: Uint8Array, at: number): number {
    at = Math.max(0, Math.min(at, bytes.length));
    while (at > 0 && at < bytes.length && (bytes[at] & 0xc0) === 0x80) at--;
    return at;
}

// A digit run such as "2020202020..." loops with a period of one or two
// tokens, far below the 8/12-token window check, and otherwise runs to
// maxTokens; stop once a short period has repeated over 12 tokens.
export function shortLoop(gen: readonly number[]): boolean {
    const n = gen.length;
    if (n < 12) return false;
    for (let p = 1; p <= 4; p++) {
        let ok = true;
        for (let i = n - 12; i < n - p && ok; i++) ok = gen[i] === gen[i + p];
        if (ok) return true;
    }
    return false;
}

// The two decode loops stop a decoder stuck repeating one phrase: the
// whole-window path compares 12-token halves, the stream path 8-token ones.
export const repeatStop = (half: number) => (gen: readonly number[]): boolean => {
    if (shortLoop(gen)) return true;
    if (gen.length < 2 * half) return false;
    return gen.slice(-2 * half, -half).join(",") === gen.slice(-half).join(",");
};
