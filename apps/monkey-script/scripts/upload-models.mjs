// Publishes the WebGPU engine's two assets to the GitHub release `models-v1`,
// which MODEL_ASSET_BASE in src/captions/assets.ts points at.
//
// Only manifest.json and qknorm.bin are uploaded. The engine also reads
// decoder_weights.q4f16.data, embed_tokens.int8.bin and embed_scales.f32.bin,
// but those are byte-identical to the files of the Hugging Face export
// (jiangzhuo9357/Qwen3-ASR-1.7B-ONNX@fcc238dfdc95cdcccaa9a7e2c7f5abc2f94f44a7)
// that the userscript already downloads into Cache Storage for the ORT
// path, so the engine reads them from there.
//
// Needs the GitHub CLI logged in with write access to RanolP/chan-necessity.
// Run: pnpm upload-models   (re-running replaces the assets in place)
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO = "RanolP/chan-necessity";
const TAG = "models-v1";
const dir = fileURLToPath(new URL("../engine-dev/model/", import.meta.url));
const files = {
    "manifest.json": "7946a4ece5013b7668111814976f621697ffb5b4c945e62bb56c1d1d7e6d77f1",
    "qknorm.bin": "e5a84f4af6ae6a3fdad2d52273e3dce92e97b80a36bc1e6cf2441a2201124850",
};

const paths = [];
for (const [name, want] of Object.entries(files)) {
    const path = dir + name;
    const got = createHash("sha256").update(readFileSync(path)).digest("hex");
    if (got !== want) throw new Error(`${path}: sha256 ${got}, expected ${want}; regenerate with engine-dev/tools/convert.py or update the hash`);
    paths.push(path);
}
const gh = (...args) => execFileSync("gh", args, { stdio: "inherit" });
let exists = true;
try {
    execFileSync("gh", ["release", "view", TAG, "-R", REPO], { stdio: "ignore" });
} catch {
    exists = false;
}
if (exists) gh("release", "upload", TAG, ...paths, "--clobber", "-R", REPO);
else gh("release", "create", TAG, ...paths, "-R", REPO, "--title", "Model assets v1", "--notes", "WebGPU decode engine assets for the live subtitles (Qwen3-ASR 1.7B q4f16). The userscript fetches these; nothing to install by hand.");
console.log(`uploaded ${Object.keys(files).join(", ")} to ${REPO} ${TAG}`);
