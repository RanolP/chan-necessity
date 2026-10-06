import { getLogger } from "./logtape.ts";

// ---- 오디오 그래프 (좌우 밸런스 · 실시간 자막 공용) -------------------------
// The one place that touches Web Audio for the player. Consumers never make
// contexts or nodes of their own; they set a pan value and read a capture
// bus. One AudioContext for the page, and per <video> at most one graph:
//
//   MediaElementSource ─→ StereoPanner ─→ destination      (what the viewer hears)
//   capture input ─→ captureBus                           (what subtitles hear)
//
// The capture input is the element's captureStream() audio track where it
// exists (Chrome): decoded audio before volume, mute and pan, taken without
// rerouting the element, so subtitles work while muted. Where it is missing
// (Firefox has only mozCaptureStream, which silences the element) it is the
// element source itself, before the panner, and so follows volume and mute.
// The bus is a fixed node; when the player swaps its <video> or reloads its
// source, the module rebinds the bus's input, so a consumer connects to the
// bus once. A pan change only moves the panner's AudioParam.
//
// Rules that follow from MediaElementSource:
// - an element accepts one MediaElementSource for life, and once rerouted it
//   plays only through this context, so the context is created and resumed
//   only inside a user gesture (a suspended context leaves it silent);
// - an element is routed only when something needs its output: a pan other
//   than centre, or capture where captureStream is missing. Once routed it
//   stays routed, and centre is then just pan 0 on its panner.
// The element's own volume and mute still apply on the routed path, so the
// player's volume controls keep working.
type CaptureVideo = HTMLVideoElement & { captureStream?: () => MediaStream };
export type VideoChange = "element" | "source";

interface Graph {
    source: MediaElementAudioSourceNode;
    panner: StereoPannerNode;
}

const logger = getLogger(["shared", "audio"]);
const SYNC_MS = 1000;
const PAN_SMOOTHING_SEC = 0.015;

export const cbAudio = (() => {
    "use strict";
    let ctx: AudioContext | null = null;
    let bus: GainNode | null = null;
    let captureUsers = 0;
    let pan = 0;
    let video: CaptureVideo | null = null;
    let captured: { video: CaptureVideo; key: string; node: AudioNode; owned: boolean } | null = null;
    const graphs = new WeakMap<HTMLMediaElement, Graph>();
    const streams = new WeakMap<HTMLMediaElement, MediaStream>();
    const listeners = new Set<(video: HTMLVideoElement, change: VideoChange) => void>();

    // Call from a user-gesture handler only.
    async function context(): Promise<AudioContext | null> {
        ctx ??= new AudioContext();
        // resume() stays pending, not rejected, when the page lacks user
        // activation; a bounded wait keeps callers from hanging on it.
        if (ctx.state !== "running")
            await Promise.race([ctx.resume().catch(() => {}), new Promise((r) => setTimeout(r, 1000))]);
        if (ctx.state !== "running") {
            logger.warn("audio context not running after resume {state}", { state: ctx.state });
            return null;
        }
        sync();
        return ctx;
    }
    const running = (): AudioContext | null => (ctx?.state === "running" ? ctx : null);
    const playerVideo = (): HTMLVideoElement | null =>
        document.querySelector<HTMLVideoElement>(".pzp-pc video") ?? document.querySelector("video");

    function route(el: HTMLMediaElement, ctx: AudioContext): Graph {
        let graph = graphs.get(el);
        if (graph) return graph;
        const source = ctx.createMediaElementSource(el);
        const panner = new StereoPannerNode(ctx, { pan });
        source.connect(panner).connect(ctx.destination);
        graph = { source, panner };
        graphs.set(el, graph);
        logger.info("audio routed {src} {pan}", { src: el.currentSrc.slice(0, 60), pan });
        return graph;
    }

    function applyPan(el: HTMLMediaElement, ctx: AudioContext) {
        const graph = pan === 0 ? graphs.get(el) : route(el, ctx);
        graph?.panner.pan.setTargetAtTime(pan, ctx.currentTime, PAN_SMOOTHING_SEC);
    }

    // captureStream() makes a new stream on every call; one per element is
    // kept. A source reload on the same element (Chrome) adds a new audio
    // track and leaves the old one "live" but silent, so the newest live
    // track is the one carrying sound.
    function captureTrack(el: CaptureVideo): MediaStreamTrack | null {
        let stream = streams.get(el);
        if (!stream) {
            stream = el.captureStream!();
            stream.addEventListener("addtrack", () => sync());
            streams.set(el, stream);
        }
        return stream.getAudioTracks().findLast((t) => t.readyState === "live") ?? null;
    }

    function uncapture() {
        if (!captured) return;
        try {
            captured.node.disconnect(bus!);
        } catch {
            // already disconnected
        }
        captured = null;
    }

    function bindCapture(el: CaptureVideo, ctx: AudioContext) {
        if (!bus || captureUsers === 0) return uncapture();
        let key: string;
        let track: MediaStreamTrack | null = null;
        if (el.captureStream) {
            try {
                track = captureTrack(el);
            } catch (error) {
                logger.warn("captureStream failed {src} {error}", { src: el.currentSrc.slice(0, 60), error });
                return;
            }
            // No audio track yet (source still loading): the next sync retries.
            if (!track) return;
            key = track.id;
        } else key = "element";
        if (captured?.video === el && captured.key === key) return;
        uncapture();
        const owned = !!track;
        const node = track ? ctx.createMediaStreamSource(new MediaStream([track])) : route(el, ctx).source;
        node.connect(bus);
        captured = { video: el, key, node, owned };
        logger.info("capture bound {via} {src}", { via: owned ? "captureStream" : "element", src: el.currentSrc.slice(0, 60) });
    }

    function watch(el: CaptureVideo) {
        const onSource = () => {
            if (el !== video) return;
            notify(el, "source");
            sync();
        };
        el.addEventListener("loadstart", onSource);
        el.addEventListener("emptied", onSource);
    }
    const watched = new WeakSet<HTMLMediaElement>();

    function notify(el: HTMLVideoElement, change: VideoChange) {
        for (const fn of listeners) {
            try {
                fn(el, change);
            } catch (error) {
                logger.error("video change listener failed {change} {src} {error}", { change, src: el.currentSrc.slice(0, 60), error });
            }
        }
    }

    // Follows the player's current <video>: a new element gets the current
    // pan and the capture bus. Runs on a timer, on the element's source
    // events, and whenever a consumer changes what it needs.
    function sync() {
        try {
            const el = playerVideo() as CaptureVideo | null;
            if (el !== video) {
                video = el;
                if (el) {
                    if (!watched.has(el)) {
                        watched.add(el);
                        watch(el);
                    }
                    logger.info("player video changed {src}", { src: el.currentSrc.slice(0, 60) });
                    notify(el, "element");
                }
            }
            const ctx = running();
            if (!ctx || !el) return;
            applyPan(el, ctx);
            bindCapture(el, ctx);
        } catch (error) {
            logger.error("audio sync failed {href} {pan} {error}", { href: location.href, pan, error });
        }
    }

    // Stores the pan and moves the current element's panner to it. Routing
    // waits for a running context (see context()).
    function setPan(value: number) {
        pan = value;
        const ctx = running();
        if (ctx && video) applyPan(video, ctx);
        else sync();
    }

    // The bus subtitles connect to, in the running context; null until a
    // gesture has started it. Every openCapture() needs one closeCapture().
    function openCapture(): GainNode | null {
        const ctx = running();
        if (!ctx) return null;
        bus ??= new GainNode(ctx);
        captureUsers++;
        sync();
        return bus;
    }
    function closeCapture() {
        captureUsers = Math.max(0, captureUsers - 1);
        if (captureUsers === 0) uncapture();
    }

    // Fires with the new element ("element") or when the current one starts
    // loading another source ("source"). Returns the unsubscribe.
    function onVideoChange(fn: (video: HTMLVideoElement, change: VideoChange) => void): () => void {
        listeners.add(fn);
        return () => listeners.delete(fn);
    }

    setInterval(sync, SYNC_MS);
    return { context, running, playerVideo, setPan, pan: () => pan, openCapture, closeCapture, onVideoChange, sync };
})();
