import { getLogger } from "./logtape.ts";

// ---- 오디오 그래프 (좌우 밸런스 · 실시간 자막 공용) -------------------------
// One AudioContext for the page and at most one graph per <video>:
// MediaElementSource → StereoPanner → destination. An element accepts a
// single MediaElementSource for life, and once rerouted it plays only
// through this context, so two rules follow:
// - the context is created and resumed only inside a user gesture; an
//   element routed into a suspended context (autoplay policy) goes silent;
// - an element is routed only when something needs its output (a pan other
//   than centre, or the subtitle fallback where captureStream is missing).
// The element's own volume and mute still apply on this path, so the
// player's volume controls keep working.
export interface AudioGraph {
    source: MediaElementAudioSourceNode;
    panner: StereoPannerNode;
}

const logger = getLogger(["shared", "audio"]);

export const cbAudio = (() => {
    "use strict";
    let ctx: AudioContext | null = null;
    const graphs = new WeakMap<HTMLMediaElement, AudioGraph>();

    // Call from a user-gesture handler only.
    async function context(): Promise<AudioContext | null> {
        ctx ??= new AudioContext();
        // resume() stays pending, not rejected, when the page lacks user
        // activation; a bounded wait keeps callers from hanging on it.
        if (ctx.state !== "running")
            await Promise.race([ctx.resume().catch(() => {}), new Promise((r) => setTimeout(r, 1000))]);
        return ctx.state === "running" ? ctx : null;
    }
    const running = (): AudioContext | null => (ctx?.state === "running" ? ctx : null);
    const playerVideo = (): HTMLVideoElement | null =>
        document.querySelector<HTMLVideoElement>(".pzp-pc video") ?? document.querySelector("video");

    function route(video: HTMLMediaElement): AudioGraph | null {
        let graph = graphs.get(video);
        if (graph) return graph;
        const ctx = running();
        if (!ctx) return null;
        const source = ctx.createMediaElementSource(video);
        const panner = new StereoPannerNode(ctx, { pan: 0 });
        source.connect(panner).connect(ctx.destination);
        graph = { source, panner };
        graphs.set(video, graph);
        logger.info("audio routed {src}", { src: video.currentSrc.slice(0, 60) });
        return graph;
    }

    return { context, running, playerVideo, route, graphOf: (video: HTMLMediaElement) => graphs.get(video) };
})();
