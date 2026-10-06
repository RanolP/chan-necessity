// Run: node src/captions/merge.check.ts
// Catches a block slide cutting a Hangul syllable between two tokens, which showed "어디서든 �" in history and "�낼" at the head of the next caption.
import { closeBlocks, commitPoint } from "./merge.ts";

// "A든낼" = 41 | EB 93 A0 | EB 82 BC, tokenized so token 2 holds the end of "든" plus the first byte of "낼"; a mark of 3 would cut inside "낼".
const pieces: Record<number, number[]> = { 1: [0xeb, 0x93], 2: [0xa0, 0xeb], 3: [0x82, 0xbc], 4: [0x20], 5: [0x41], 6: [0x42], 7: [0x43], 8: [0x44], 9: [0x45] };
const bytesOf = (ids: readonly number[]) => Uint8Array.from(ids.flatMap((t) => pieces[t]));
const dec = (ids: readonly number[]) => new TextDecoder("utf-8", { fatal: true }).decode(bytesOf(ids));
const st = { open: new Float32Array(2), closed: [{ pcm: new Float32Array(1), af: null, mark: 3 }], tokens: [5, 1, 2, 3, 4, 6, 7, 8, 9], histIds: [] as number[] };
const slid = closeBlocks(st, 2, 1, 5, bytesOf);
dec(slid);
dec(st.tokens);
if (slid.join() !== "5") throw new Error(`slid [${slid.join()}], want [5]: the cut must fall back to the last whole character`);

// Catches the 5:43:12 history burst: a hop slower than the page's 7 s clear timeout reset the shown lines, the worker re-sent its whole 24 s context starting mid-word ("을 정도임"), and with nothing committed to align against all of it was recorded again in one go.
const shown = [
    "무슨 말을 하는지 모르겠을 정도임 지금 파스를 좀 붙여야 될",
    "것 같은데 왜 왜 갑자기 이렇게 됐지 야 근데 이거 하나 하나",
    "보면서 하니까 진짜 시간이 엄청 많이 가긴 했다 지금 보니까",
    "그리고 원래는 이거 전부 다 볼 생각은 없었는데 하다 보니까",
    "다 보고 싶어져서 음 오랜만에 말을 많이 해서 그런가 그럴",
    "수도 있고 어쨌든 이백삼십 분이라고 하면은 이백삼십",
];
const carry = shown.join(" ");
const resent = "을 정도임 지금 파스를 좀 붙여야 될 것 같은데 왜 왜 갑자기 이렇게 됐지 야 근데 이거 하나 하나 보면서 하니까 진짜 시간이 엄청 많이 가긴 했다 지금 보니까 그리고 원래는 이거 전부 다 볼 생각은 없었는데 하다 보니까 다 보고 싶어져서 음 오랜만에 말을 많이 해서 그런가 그럴 수도 있고 어쨌든 이백삼십 분이라고 하면은 이백삼십";
const fresh = " 분 정도 걸렸네";
const rest = (frozen: string[], c: string, hyp: string) => hyp.slice(commitPoint(frozen, c, hyp)).trim();
if (rest([], carry, resent + fresh) !== fresh.trim()) throw new Error(`after a stale gap the re-sent context must align to the carried tail; shown "${rest([], carry, resent + fresh)}"`);
if (rest([], carry, resent) !== "") throw new Error(`a re-sent context with nothing new must show nothing new; shown "${rest([], carry, resent)}"`);
// Catches the carried tail eating new speech after a real pause: unrelated text must show whole.
const next = "자 그럼 다음 영상으로 넘어가 볼게요";
if (rest([], carry, next) !== next) throw new Error(`new speech after a stale gap lost words; shown "${rest([], carry, next)}"`);
console.log("merge.check ok");
