// Publishes the WebGPU engine's assets to the GitHub release `models-v1`,
// which MODEL_ASSET_BASE in src/captions/assets.ts points at.
//
// Per model only manifest.json and qknorm.bin are uploaded, under the names
// engineAssetFiles() in src/captions/assets.ts gives them. The engine also
// reads decoder_weights.q4f16.data, embed_tokens.int8.bin and
// embed_scales.f32.bin, but those are byte-identical to the files of the
// Hugging Face export that the userscript already downloads into Cache
// Storage, so the engine reads them from there.
//
// Only assets the release lacks are uploaded; an existing asset is never
// replaced, because released userscripts fetch it by URL. To change one,
// delete it on GitHub first.
//
// The 0.6B pair is not committed; build it first with
//   uv run --with onnx --with numpy python engine-dev/tools/convert.py <dir> jiangzhuo9357/Qwen3-ASR-0.6B-ONNX@4a01b95fafe2c9e3af77e33c18bbb7de349c62f6 engine-dev/model/0.6B
// where <dir> holds that revision's decoder_step.q4f16.onnx and config.json.
//
// Needs the GitHub CLI logged in with write access to RanolP/chan-necessity.
// Run: pnpm upload-models
import { createHash } from "node:crypto";
import { copyFileSync, mkdtempSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = "RanolP/chan-necessity";
const TAG = "models-v1";
const model = (sub) => fileURLToPath(new URL(`../engine-dev/model/${sub}`, import.meta.url));
// release asset name -> [local file, sha256]
const assets = {
    "manifest.json": [model("manifest.json"), "7946a4ece5013b7668111814976f621697ffb5b4c945e62bb56c1d1d7e6d77f1"],
    "qknorm.bin": [model("qknorm.bin"), "e5a84f4af6ae6a3fdad2d52273e3dce92e97b80a36bc1e6cf2441a2201124850"],
    "manifest-0.6B.json": [model("0.6B/manifest.json"), "b72fb52b984de79091687b40db895fb6d0e2c42f0854314c499c588c29ec9d44"],
    "qknorm-0.6B.bin": [model("0.6B/qknorm.bin"), "420ef8e03f8c0f0f04135af687a62aab0694c5300efe79178bb3c540d0f3d8dd"],
};

let existing = null;
try {
    existing = new Set(JSON.parse(execFileSync("gh", ["release", "view", TAG, "-R", REPO, "--json", "assets"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] })).assets.map((a) => a.name));
} catch (error) {
    if (!/release not found/i.test(String(error.stderr ?? error))) throw error;
}
const todo = Object.keys(assets).filter((name) => !existing?.has(name));
if (!todo.length) {
    console.log(`${REPO} ${TAG} already has ${Object.keys(assets).join(", ")}; nothing to upload`);
    process.exit(0);
}
const stage = mkdtempSync(join(tmpdir(), "upload-models-"));
const paths = todo.map((name) => {
    const [path, want] = assets[name];
    const got = createHash("sha256").update(readFileSync(path)).digest("hex");
    if (got !== want) throw new Error(`${path}: sha256 ${got}, expected ${want}; regenerate with engine-dev/tools/convert.py or update the hash`);
    copyFileSync(path, join(stage, name));
    return join(stage, name);
});
const gh = (...args) => execFileSync("gh", args, { stdio: "inherit" });
if (existing) gh("release", "upload", TAG, ...paths, "-R", REPO);
else gh("release", "create", TAG, ...paths, "-R", REPO, "--title", "Model assets v1", "--notes", "WebGPU decode engine assets for the live subtitles (Qwen3-ASR q4f16). The userscript fetches these; nothing to install by hand.");
console.log(`uploaded ${todo.join(", ")} to ${REPO} ${TAG}`);
