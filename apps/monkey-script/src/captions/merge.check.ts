// Run: node src/captions/merge.check.ts
// Catches a block slide cutting a Hangul syllable between two tokens, which showed "어디서든 �" in history and "�낼" at the head of the next caption.
import { closeBlocks } from "./merge.ts";

// "A든낼" = 41 | EB 93 A0 | EB 82 BC, tokenized so token 2 holds the end of "든" plus the first byte of "낼"; a mark of 3 would cut inside "낼".
const pieces: Record<number, number[]> = { 1: [0xeb, 0x93], 2: [0xa0, 0xeb], 3: [0x82, 0xbc], 4: [0x20], 5: [0x41], 6: [0x42], 7: [0x43], 8: [0x44], 9: [0x45] };
const bytesOf = (ids: readonly number[]) => Uint8Array.from(ids.flatMap((t) => pieces[t]));
const dec = (ids: readonly number[]) => new TextDecoder("utf-8", { fatal: true }).decode(bytesOf(ids));
const st = { open: new Float32Array(2), closed: [{ pcm: new Float32Array(1), af: null, mark: 3 }], tokens: [5, 1, 2, 3, 4, 6, 7, 8, 9], histIds: [] as number[] };
const slid = closeBlocks(st, 2, 1, 5, 24, bytesOf);
dec(slid);
dec(st.tokens);
if (slid.join() !== "5") throw new Error(`slid [${slid.join()}], want [5]: the cut must fall back to the last whole character`);
console.log("merge.check ok");
