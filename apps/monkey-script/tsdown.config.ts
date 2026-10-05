import { build as rolldown, type Plugin } from "rolldown";
import { defineConfig } from "tsdown";
import { meta, renderMeta } from "./userscript.config.ts";

// The STT worker runs from a Blob URL, so it cannot load sibling files.
// It is bundled on its own as one ESM module and imported by the page side
// as a string through this virtual module.
const WORKER_ID = "virtual:captions-worker";
const sttWorker: Plugin = {
    name: "chan-necessity:captions-worker",
    resolveId(id) {
        return id === WORKER_ID ? "\0" + WORKER_ID : null;
    },
    async load(id) {
        if (id !== "\0" + WORKER_ID) return null;
        const out = await rolldown({
            input: "src/captions/worker/main.ts",
            platform: "browser",
            write: false,
            output: { format: "esm", minify: false, codeSplitting: false },
        });
        const chunks = out.output.filter((o) => o.type === "chunk");
        if (chunks.length !== 1) throw new Error(`captions worker: expected 1 chunk, got ${chunks.map((c) => c.fileName).join(", ")}`);
        for (const f of chunks[0].moduleIds) this.addWatchFile(f);
        return `export default ${JSON.stringify(chunks[0].code)};`;
    },
};

export default defineConfig({
    entry: { "chan-necessity": "src/main.ts" },
    format: "iife",
    platform: "browser",
    target: false,
    minify: false,
    hash: false,
    dts: false,
    clean: true,
    outputOptions: { entryFileNames: "[name].user.js" },
    banner: renderMeta(meta),
    deps: { alwaysBundle: [/.*/] },
    plugins: [sttWorker],
});
