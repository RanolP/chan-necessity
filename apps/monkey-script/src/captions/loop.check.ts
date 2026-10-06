// Run: node src/captions/loop.check.ts
// Synthetic token ids. `hop` mirrors the stream path in worker/main.ts:
// trim forced against seen history, stop on seen + forced + gen, trim the result.
import { loopStop, trimLoop, LOOP_SPAN } from "./merge.ts";

const ROLLBACK = 5;
function hop(seen: number[], tokens: number[], out: number[]) {
    const forced = trimLoop(seen, tokens.slice(0, Math.max(0, tokens.length - ROLLBACK)));
    const lead = [...seen, ...forced].slice(-LOOP_SPAN);
    const gen: number[] = [];
    for (const t of out) {
        gen.push(t);
        if (loopStop([...lead, ...gen])) break;
    }
    return { forced, gen, tokens: trimLoop(seen, [...forced, ...gen]) };
}
function eq(got: readonly number[], want: readonly number[], why: string) {
    if (got.join() !== want.join()) throw new Error(`${why}\n  got  [${got.join()}]\n  want [${want.join()}]`);
}
const rep = (x: number[], k: number) => Array.from({ length: k }, () => x).flat();
const G = 7; // "그"

// Catches a 1-token loop that crosses the forced/gen boundary going unstopped because each hop's gen alone is under 12 tokens ("…그그그그" then "그그그 스팍").
{
    const prev = [10, 11, 12, ...rep([G], 12)]; // forced keeps 7 그, gen adds 6: neither alone reaches 12
    const r = hop([], prev, [...rep([G], 6), 30, 31]);
    if (r.gen.length === 8) throw new Error("1-token loop across forced/gen did not stop");
    eq(r.tokens, [10, 11, 12, G], "1-token loop must collapse to one 그");
}
// Catches a loop already inside forced being re-fed as the answer's start, which primed the model to continue it.
eq(trimLoop([], [10, ...rep([G], 14)]), [10, G], "forced text must not carry a loop");

// Catches a 9-token phrase ("이분도 누군지 알아요? 그럼") repeated 3x across forced+gen, invisible to the old 8-token-halves check.
{
    const P = [40, 41, 42, 43, 44, 45, 46, 47, 48];
    const prev = [1, 2, ...P, ...P.slice(0, 7)];
    const r = hop([], prev, [...P.slice(2), ...P, ...P]);
    eq(r.tokens, [1, 2, ...P], "phrase loop must collapse to one copy");
}
// Catches a phrase ("이분이 죽었을 때 이십세기에는") that starts in slid history and is repeated in forced+gen: the copy already shown must not be repeated, and history is never cut.
{
    const P = [60, 61, 62, 63, 64, 65, 66];
    const seen = [3, 4, ...P];
    const r = hop(seen, [...P, ...P, 70, 71], [...P, ...P]);
    eq(r.forced, [], "forced must drop the copy history already holds");
    eq(r.tokens, [], "a hop that only continues the history loop adds nothing");
    eq(trimLoop([], [...seen, ...P, ...P]), seen, "history injected into the prompt collapses to one copy");
}
// Catches the detector cutting normal speech that repeats a short word ("네 네", "네 네 네") or a short phrase twice.
{
    const ne = [50, 4, 50, 4, 50];
    const twice = [80, 81, 82, 83, 80, 81, 82, 83];
    const speech = [20, 21, 22, ...ne, 23, 24, ...twice, 25];
    for (let i = 1; i <= speech.length; i++) if (loopStop(speech.slice(0, i))) throw new Error(`false loop at ${i}: [${speech.slice(0, i).join()}]`);
    eq(hop([20, 21], speech.slice(0, 10), speech.slice(10 - ROLLBACK)).tokens, speech, "normal speech must pass through unchanged");
}
console.log("loop.check ok");
