// The WebGPU decode engine needs two small files besides the Hugging Face
// export: its manifest (tensor layout) and qknorm.bin. They are attached to
// the chan-necessity GitHub release `models-v1`; the decoder weights and the
// embeddings it also reads are byte-identical to the Hugging Face files the
// ORT path already caches, so they are not re-hosted.
import { getLogger } from "../shared/logtape.ts";
import type { EngineAssets } from "./protocol.ts";

const logger = getLogger(["captions", "assets"]);

export const MODEL_ASSET_BASE = "https://github.com/RanolP/chan-necessity/releases/download/models-v1/";
// Points the engine at another copy of the assets (e.g. a local server
// sending CORS headers); set it with GM_setValue in the Violentmonkey
// script's Values tab. An override is fetched every load, never cached.
export const KEY_ASSET_BASE = "dev.modelAssetBase";
const FILES = ["manifest.json", "qknorm.bin"];
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

// Resolves to { manifest, qknorm } ArrayBuffers, or null when they cannot
// be had; the subtitles then decode on the ORT path, as before.
export async function loadEngineAssets(cacheName: string, getSetting: (key: string, fallback: string) => unknown): Promise<EngineAssets | null> {
    const override = String(getSetting(KEY_ASSET_BASE, "") || "").trim();
    const base = override || MODEL_ASSET_BASE;
    try {
        const cache = override ? null : await caches.open(cacheName);
        const out: Partial<EngineAssets> = {};
        for (const f of FILES) {
            const url = base + f;
            const hit = await cache?.match(url);
            let buf = hit ? await hit.arrayBuffer() : null;
            if (!buf) {
                buf = override ? await plainGet(url) : await gmGet(url);
                await cache?.put(url, new Response(buf.slice(0)));
            }
            out[f === "manifest.json" ? "manifest" : "qknorm"] = buf;
        }
        return out as EngineAssets;
    } catch (error) {
        logger.warn("engine assets unavailable, decoding on the ORT path: {base}", { base, error });
        return null;
    }
}
