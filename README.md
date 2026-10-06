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

The subtitles run Qwen3-ASR (`jiangzhuo9357/Qwen3-ASR-1.7B-ONNX` or the 0.6B export) with onnxruntime-web on WebGPU. With the 1.7B model on a device with `shader-f16`, token decoding runs in a dedicated WebGPU engine (`src/captions/engine/`) instead of the ORT decode loop. The engine reads the decoder weights and embeddings from the same Hugging Face files the ORT path caches, and needs only two extra files, `manifest.json` and `qknorm.bin`, which are published to the `models-v1` GitHub release by `pnpm upload-models`. If they cannot be fetched, the subtitles keep working on the ORT path.

`apps/monkey-script/engine-dev/` holds the engine's conversion tools and its standalone test page.

## License

Apache-2.0. Parts are based on MIT-licensed work by others; see [LICENSE](LICENSE) for their notices.
