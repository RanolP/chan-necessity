// Silero VAD parity: our createVad against the probabilities ORT produced for
// the same model on the same seeded frames (fixtures/vad-reference.json).
//
//   node scripts/vad-parity.mjs [silero_vad_v5.onnx path or URL]
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const reference = JSON.parse(await readFile(new URL("fixtures/vad-reference.json", import.meta.url), "utf8"));
const modelSource = process.argv[2] ?? reference.model;
const { createVad, parseSileroVadWeights } = await import("../src/captions/vad.ts");

const model = modelSource.startsWith("http")
    ? new Uint8Array(await (await fetch(modelSource)).arrayBuffer())
    : new Uint8Array(await readFile(modelSource));
const sha = createHash("sha256").update(model).digest("hex");
if (sha !== reference.modelSha256) throw new Error(`${modelSource}: sha256 ${sha}, the reference was frozen from ${reference.modelSha256}`);
const vad = createVad(parseSileroVadWeights(model));
const frame = new Float32Array(576);
const random = mulberry32(0x511e_70);
let maxAbsDiff = 0;
let frames = 0;

function mulberry32(seed) {
    return () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

for (let i = 0; i < reference.frames; i++) {
    for (let j = 0; j < frame.length; j++) {
        const time = (i * 512 + j) / 16000;
        const speech = i >= 32 && i < 128;
        frame[j] = speech
            ? 0.08 * Math.sin(2 * Math.PI * 180 * time) + 0.025 * Math.sin(2 * Math.PI * 620 * time)
            : (random() * 2 - 1) * 0.004;
    }
    const diff = Math.abs(vad.prob(frame) - reference.probs[i]);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    frames++;
}

console.log(JSON.stringify({ model: modelSource, frames, maxAbsProbDiff: maxAbsDiff }));
