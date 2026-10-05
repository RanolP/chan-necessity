declare module "virtual:captions-worker" {
    const source: string;
    export default source;
}

// Violentmonkey's GM_* API, for the grants in userscript.config.ts.
interface GMXhrResponse<T> {
    status: number;
    statusText: string;
    response: T;
    error?: string;
}
interface GMXhrDetails<T> {
    url: string;
    method?: "GET" | "POST" | "HEAD";
    responseType?: "arraybuffer" | "blob" | "json" | "text";
    timeout?: number;
    headers?: Record<string, string>;
    onload?: (r: GMXhrResponse<T>) => void;
    onerror?: (r: GMXhrResponse<T>) => void;
    ontimeout?: (r: GMXhrResponse<T>) => void;
}
declare function GM_getValue<T>(key: string, defaultValue: T): T;
declare function GM_getValue(key: string): unknown;
declare function GM_setValue(key: string, value: unknown): void;
declare function GM_xmlhttpRequest<T = unknown>(details: GMXhrDetails<T>): { abort(): void };
declare const unsafeWindow: Window & typeof globalThis;
