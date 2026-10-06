# chan-necessity

A userscript for [CHZZK](https://chzzk.naver.com/) that adds the 1080p option the grid requirement hides, collects 통나무 파워 automatically, keeps live bookmarks and shows them on the VOD timeline, transcribes live audio into subtitles in the browser, and plays up to three channels side by side in a split view with per-column left/right panning.

Install it with [Violentmonkey](https://violentmonkey.github.io/) from https://ranolp.github.io/chan-necessity/chan-necessity.user.js; the site at https://ranolp.github.io/chan-necessity/ has the usage docs (Korean).

## Layout

- `apps/monkey-script` builds the userscript with tsdown into one self-contained `dist/chan-necessity.user.js`. Each feature is its own module under `src/` (`deny-grid`, `auto-claim-logs`, `bookmarks`, `sound-panning`, `captions`, `split-view`, `clip-for-firefox`), and `src/shared/` holds the split-view context, the audio graph and the LogTape setup. The build type-checks with `tsc --noEmit` before bundling. The metadata block is generated from `userscript.config.ts`. The captions worker (`src/captions/worker/`) is bundled separately and inlined as a string, which the page starts from a Blob URL.

Every module logs through LogTape under the category `chan-necessity·<module>`, to the browser console at `info` and above. Set the script value `dev.logLevel` to `"debug"` (Violentmonkey: the script's Values tab) to see debug logs.
- `apps/website` is the Astro Starlight site deployed to GitHub Pages by `.github/workflows/pages.yml`. Its build copies the built userscript to the site root, where `@updateURL` and `@downloadURL` point.

## Build

```sh
pnpm install
pnpm build
```

Node and pnpm come from `mise.toml`; the `packageManager` field pins the same pnpm version.

## Subtitle model assets

The subtitles run Qwen3-ASR (`jiangzhuo9357/Qwen3-ASR-1.7B-ONNX`, or `jiangzhuo9357/Qwen3-ASR-0.6B-ONNX` from the settings menu) entirely on an in-house WebGPU engine (`src/captions/engine/`): the encoder, the prefill and decoder, and the Silero VAD (`src/captions/vad.ts`). There is no onnxruntime at runtime, and the `.onnx` files are downloaded only as weight containers that `engine/encoder-onnx.ts` parses itself. The engine reads the decoder weights and embeddings from the Hugging Face files and needs two extra files per model, `manifest.json` and `qknorm.bin` for 1.7B and `manifest-0.6B.json` and `qknorm-0.6B.bin` for 0.6B, which `engine-dev/tools/convert.py` builds and `pnpm upload-models` publishes to the `models-v1` GitHub release. Without them the worker fails to load, and there is no fallback path. The kernels read f16 weights as packed `u32`, so the GPU does not need `shader-f16`.

`apps/monkey-script/engine-dev/` holds the engine's conversion tools and its standalone test page.

## License

Apache-2.0. Parts are based on MIT-licensed work by others; see [LICENSE](LICENSE) for their notices.
