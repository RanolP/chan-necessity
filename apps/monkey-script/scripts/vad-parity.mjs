import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/silero_vad_v5.onnx";
const modelSource = process.argv[2] ?? MODEL_URL;
const ort = require("onnxruntime-node");
const { createVad, parseSileroVadWeights } = await import("../src/captions/vad.ts");

const model = modelSource.startsWith("http")
    ? new Uint8Array(await (await fetch(modelSource)).arrayBuffer())
    : new Uint8Array(await readFile(modelSource));
const weights = parseSileroVadWeights(model);
const vad = createVad(weights);
const session = await ort.InferenceSession.create(model);
const frame = new Float32Array(576);
const referenceState = new Float32Array(2 * 128);
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

for (let i = 0; i < 188; i++) {
    for (let j = 0; j < frame.length; j++) {
        const time = (i * 512 + j) / 16000;
        const speech = i >= 32 && i < 128;
        frame[j] = speech
            ? 0.08 * Math.sin(2 * Math.PI * 180 * time) + 0.025 * Math.sin(2 * Math.PI * 620 * time)
            : (random() * 2 - 1) * 0.004;
    }
    const actual = vad.prob(frame);
    const input = frame.slice();
    const result = await session.run({
        input: new ort.Tensor("float32", input, [1, 576]),
        state: new ort.Tensor("float32", referenceState, [2, 1, 128]),
        sr: new ort.Tensor("int64", BigInt64Array.from([16000n]), []),
    });
    const expected = result.output.data[0];
    const nextState = result.stateN.data;
    referenceState.set(nextState);
    const diff = Math.abs(actual - expected);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    frames++;
}

console.log(JSON.stringify({ model: modelSource, frames, maxAbsProbDiff: maxAbsDiff }));
