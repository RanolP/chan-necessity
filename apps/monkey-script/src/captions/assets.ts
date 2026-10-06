// The WebGPU decode engine needs two small files per model besides the
// Hugging Face export: its manifest (tensor layout) and qknorm.bin. They are
// attached to the chan-necessity GitHub release `models-v1`; the decoder
// weights and the embeddings it also reads are byte-identical to the Hugging
// Face files the worker already caches, so they are not re-hosted.
import { getLogger } from "../shared/logtape.ts";
import type { EngineAssets } from "./protocol.ts";

const logger = getLogger(["captions", "assets"]);

export const MODEL_ASSET_BASE = "https://github.com/RanolP/chan-necessity/releases/download/models-v1/";
// Points the engine at another copy of the assets (e.g. a local server
// sending CORS headers); set it with GM_setValue in the Violentmonkey
// script's Values tab. An override is fetched every load, never cached.
export const KEY_ASSET_BASE = "dev.modelAssetBase";
// 1.7B keeps the unsuffixed names it shipped with in 0.6.0, whose URLs
// released copies still fetch; every other model is suffixed by its name.
export function engineAssetFiles(model: string) {
    const suffix = model === "1.7B" ? "" : `-${model}`;
    return { manifest: `manifest${suffix}.json`, qknorm: `qknorm${suffix}.bin` };
}
// A request can stall with no error (Firefox holds a page fetch to a local
// address while it waits on its local-network permission), and the model
// load awaits these files, so each one is given up on after this long.
const FETCH_TIMEOUT_MS = 30_000;

// Release downloads redirect to a host that sends no CORS headers, so the
// default base goes through GM_xmlhttpRequest (the @connect hosts).
function gmGet(url: string): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
        GM_xmlhttpRequest<ArrayBuffer>({
            url,
            method: "GET",
            responseType: "arraybuffer",
            timeout: FETCH_TIMEOUT_MS,
            onload: (r) => (r.status === 200 ? resolve(r.response) : reject(new Error(`${url} → HTTP ${r.status} ${String(r.statusText ?? "")}`))),
            onerror: (r) => reject(new Error(`${url} → network error ${r?.error ?? ""}`)),
            ontimeout: () => reject(new Error(`${url} → timeout`)),
        });
    });
}
async function plainGet(url: string): Promise<ArrayBuffer> {
    const res = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
    if (!res.ok) throw new Error(`${url} → HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    return res.arrayBuffer();
}

// Resolves to { manifest, qknorm } ArrayBuffers for `model`, or to { error }
// naming what could not be had; the worker then fails its load with that
// error, since it has no other decoder.
export async function loadEngineAssets(cacheName: string, getSetting: (key: string, fallback: string) => unknown, model: string): Promise<EngineAssets | { error: string }> {
    const override = String(getSetting(KEY_ASSET_BASE, "") || "").trim();
    const base = override || MODEL_ASSET_BASE;
    const files = engineAssetFiles(model);
    try {
        const cache = override ? null : await caches.open(cacheName);
        const out: Partial<EngineAssets> = {};
        for (const [key, f] of Object.entries(files) as [keyof EngineAssets, string][]) {
            const url = base + f;
            const hit = await cache?.match(url);
            let buf = hit ? await hit.arrayBuffer() : null;
            if (!buf) {
                buf = override ? await plainGet(url) : await gmGet(url);
                await cache?.put(url, new Response(buf.slice(0)));
            }
            out[key] = buf;
        }
        return out as EngineAssets;
    } catch (error) {
        logger.warn("engine assets unavailable, captions cannot load: {model} {base}", { model, base, error });
        return { error: `${model} ${files.manifest}, ${files.qknorm} from ${base}: ${error instanceof Error ? error.message : String(error)}` };
    }
}
