import { getLogger } from "../shared/logtape.ts";

const logger = getLogger(["bookmarks"]);

// ---- 라이브 북마크 → 다시보기 ------------------------------------------
// A bookmark records the wall-clock moment the viewer was watching. The
// replay of the same broadcast starts at liveOpenDate, so the replay
// offset is wallTime − openDate, minus the length of earlier parts when
// a long broadcast was split into several videos.
interface Bookmark {
    id: string;
    channelId: string;
    liveId: number;
    videoId: string | null;
    openDate: string;
    wallTime: number;
    timeSource: string;
    memo: string;
    liveTitle: string;
    createdAt: number;
}

interface LiveInfo {
    channelId: string;
    liveId: number;
    openDate: string;
    liveTitle: string;
    status: string;
    videoId: string | null;
}

interface LiveDetailResponse {
    liveId: number;
    openDate: string;
    liveTitle: string;
    status: string;
    livePlaybackJson: string;
}

interface VideoResponse {
    videoNo: number;
    videoId: string;
    liveOpenDate: string;
    publishDate: string;
    videoType: string;
    duration: number;
    channel?: { channelId: string };
}

interface VodInfo {
    videoNo: number;
    videoId: string;
    liveOpenDate: string;
    duration: number;
    channelId: string | undefined;
    earlierSec: number;
}

interface VideoListResponse {
    data?: { videoNo: number; videoType: string; publishDate: string }[];
}

interface PlayerRef {
    getProgramDateTime(): number;
}

interface FiberHook {
    memoizedState: unknown;
    next: FiberHook | null;
}

interface Fiber {
    memoizedState: FiberHook | null;
    sibling: Fiber | null;
    child: Fiber | null;
}

type Route = { kind: "none" | "live" | "video"; key: string };
type TimeSource = "player" | "openDate+currentTime" | "now";
interface Entry {
    bookmark: Bookmark;
    offset: number;
}

(() => {
    const STORAGE_KEY = "bookmarks";
    const API = "https://api.chzzk.naver.com/service";
    const LIVE_RE = /^\/live\/([0-9a-f]{32})/i;
    const VIDEO_RE = /^\/video\/(\d+)/;
    // Split parts each cover at most ~17h, so a gap this large between
    // open and publish beyond the video's own length means earlier parts.
    const EARLIER_PART_SLACK_SEC = 3600;
    const ICON =
        '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true" class="pzp-ui-icon__svg"><path d="M13 10.5C13 9.67 13.67 9 14.5 9h7c.83 0 1.5.67 1.5 1.5V26l-5-3.4-5 3.4V10.5Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>';

    const pageWindow =
        typeof unsafeWindow === "undefined" ? window : unsafeWindow;

    const load = (): Bookmark[] => {
        try {
            const value = GM_getValue<unknown>(STORAGE_KEY, []);
            return Array.isArray(value) ? (value as Bookmark[]) : [];
        } catch (error) {
            logger.error("bookmark load failed {error}", { error });
            return [];
        }
    };
    const store = (list: Bookmark[]) => GM_setValue(STORAGE_KEY, list);

    const kstToMs = (text: string | null | undefined) =>
        text ? Date.parse(text.replace(" ", "T") + "+09:00") : NaN;

    function formatClock(sec: number) {
        sec = Math.max(0, Math.floor(sec));
        const h = Math.floor(sec / 3600);
        const m = String(Math.floor(sec / 60) % 60).padStart(2, "0");
        const s = String(sec % 60).padStart(2, "0");
        return `${h}:${m}:${s}`;
    }

    async function getJson<T>(url: string): Promise<T> {
        const response = await pageWindow.fetch(url, {
            credentials: "include",
        });
        const body = await response.text();
        if (!response.ok) {
            throw new Error(
                `[ChzzkBest] ${url} → ${response.status}: ${body.slice(0, 200)}`
            );
        }
        try {
            return JSON.parse(body).content as T;
        } catch {
            throw new Error(
                `[ChzzkBest] ${url} → bad JSON: ${body.slice(0, 200)}`
            );
        }
    }

    async function fetchLive(channelId: string): Promise<LiveInfo> {
        const c = await getJson<LiveDetailResponse>(
            `${API}/v3/channels/${channelId}/live-detail`
        );
        let videoId: string | null = null;
        try {
            videoId = JSON.parse(c.livePlaybackJson).meta.videoId ?? null;
        } catch (error) {
            logger.warn("livePlaybackJson unreadable {error}", { error });
        }
        return {
            channelId,
            liveId: c.liveId,
            openDate: c.openDate,
            liveTitle: c.liveTitle,
            status: c.status,
            videoId,
        };
    }

    // Chzzk's player component exposes getProgramDateTime() through a React
    // imperative ref; it is startDate + currentTime of the core player, so
    // it already accounts for latency and 타임머신 rewind.
    function findPlayerRef(): PlayerRef | null {
        const host = document.querySelector("#root") as (Element & Record<string, unknown>) | null;
        const key = host && Object.keys(host).find((k) =>
            k.startsWith("__reactContainer$")
        );
        if (!key) return null;
        const stack = [host[key] as Fiber];
        for (let visited = 0; stack.length && visited < 50_000; visited++) {
            const fiber = stack.pop();
            if (!fiber) continue;
            let hook = fiber.memoizedState;
            for (
                let i = 0;
                hook && typeof hook === "object" && "next" in hook && i < 80;
                i++, hook = hook.next
            ) {
                const ref = hook.memoizedState;
                const current: PlayerRef | null =
                    ref && typeof ref === "object" && "current" in ref
                        ? (ref.current as PlayerRef | null)
                        : null;
                if (typeof current?.getProgramDateTime === "function") {
                    return current;
                }
            }
            if (fiber.sibling) stack.push(fiber.sibling);
            if (fiber.child) stack.push(fiber.child);
        }
        return null;
    }

    // The fiber walk is too costly for the timeline's periodic refresh, so
    // the ref is cached until the player element is replaced.
    let cachedRef: PlayerRef | null = null;
    let cachedFor: Element | null = null;
    function playerRef() {
        const player = document.querySelector(".pzp-pc");
        if (!cachedRef || cachedFor !== player) {
            cachedRef = findPlayerRef();
            cachedFor = cachedRef ? player : null;
        }
        return cachedRef;
    }

    function watchedWallTime(live: { openDate: string | null }): { wallTime: number; timeSource: TimeSource } {
        const now = Date.now();
        const openMs = kstToMs(live.openDate);
        const plausible = (t: number | undefined): t is number =>
            t !== undefined && Number.isFinite(t) && t <= now + 5_000 && !(t < openMs - 60_000);
        try {
            const t = playerRef()?.getProgramDateTime();
            if (plausible(t)) return { wallTime: t, timeSource: "player" };
        } catch (error) {
            logger.warn("getProgramDateTime failed {error}", { error });
        }
        cachedRef = null;
        const video = document.querySelector("video");
        const t = openMs + 1000 * (video?.currentTime ?? NaN);
        if (plausible(t)) {
            return { wallTime: Math.floor(t), timeSource: "openDate+currentTime" };
        }
        return { wallTime: now, timeSource: "now" };
    }

    // ---- UI ----------------------------------------------------------------
    const style = document.createElement("style");
    style.textContent = `
        .cb-bm-panel { position: absolute; top: 26px; left: 12px; z-index: 30;
            max-width: 320px; font: 12px/1.5 sans-serif; color: #fff;
            background: rgba(20, 20, 24, .82); border-radius: 8px; }
        .cb-bm-panel > button { all: unset; cursor: pointer; padding: 4px 10px;
            display: block; font-weight: 600; }
        .cb-bm-panel ul { list-style: none; margin: 0; padding: 0 6px 6px;
            max-height: 240px; overflow-y: auto; }
        .cb-bm-panel li { display: flex; gap: 6px; align-items: baseline;
            padding: 2px 4px; border-radius: 4px; }
        .cb-bm-panel li[data-seek]:hover { background: rgba(255,255,255,.12);
            cursor: pointer; }
        .cb-bm-panel time { font-variant-numeric: tabular-nums; color: #00ffa3; }
        .cb-bm-panel span { flex: 1; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; }
        .cb-bm-panel .cb-bm-del { all: unset; cursor: pointer; opacity: .6; }
        .cb-bm-panel .cb-bm-del:hover { opacity: 1; }
        .cb-bm-panel.cb-collapsed ul { display: none; }
        .cb-bm-marker { position: absolute; top: 50%; width: 4px; height: 12px;
            margin-left: -2px; transform: translateY(-50%); z-index: 5;
            background: #ffd23f; box-shadow: 0 0 0 1px rgba(0,0,0,.6);
            border-radius: 2px; cursor: pointer; }
        .cb-bm-tl { position: absolute; top: 0; left: 0; right: 0; height: 18px;
            z-index: 31; display: flex; align-items: center; gap: 8px;
            padding: 0 10px; pointer-events: none;
            background: linear-gradient(rgba(0,0,0,.55), rgba(0,0,0,0));
            font: 11px/1 sans-serif; color: #fff;
            font-variant-numeric: tabular-nums; text-shadow: 0 0 2px #000, 0 0 2px #000; }
        .cb-bm-panel, .cb-bm-tl, .cb-bm-button {
            transition: opacity .2s ease-in, visibility .2s ease-in; }
        .pzp-pc:not(.pzp-pc--controls) .cb-bm-panel,
        .pzp-pc:not(.pzp-pc--controls) .cb-bm-tl,
        .pzp-pc:not(.pzp-pc--controls) .cb-bm-button {
            opacity: 0; visibility: hidden; pointer-events: none; }
        .cb-bm-tl-track { position: relative; flex: 1; height: 4px; border-radius: 2px;
            background: rgba(255,255,255,.3); box-shadow: 0 0 0 1px rgba(0,0,0,.45); }
        .cb-bm-tl-fill { position: absolute; left: 0; top: 0; bottom: 0;
            border-radius: 2px; background: #00ffa3; }
        .cb-bm-tl-head { position: absolute; top: 50%; width: 3px; height: 12px;
            margin-left: -1.5px; transform: translateY(-50%); border-radius: 1px;
            background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.7); }
        .cb-bm-tl-mark { position: absolute; top: 50%; width: 4px; height: 12px;
            margin-left: -2px; transform: translateY(-50%); border-radius: 2px;
            background: #ffd23f; box-shadow: 0 0 0 1px rgba(0,0,0,.7);
            pointer-events: auto; cursor: default; }
        .cb-bm-tl-mark:hover { transform: translateY(-50%) scale(1.4); }
        .cb-bm-toast { position: absolute; left: 50%; bottom: 72px; z-index: 40;
            transform: translateX(-50%); padding: 6px 14px; border-radius: 6px;
            color: #fff; background: rgba(0, 0, 0, .8); font: 13px sans-serif;
            pointer-events: none; }
    `;

    let route: Route = { kind: "none", key: "" };
    let live: LiveInfo | null = null;
    let vod: VodInfo | null = null;
    let collapsed = true;
    let renderedSignature = "";

    function toast(text: string) {
        const host = document.querySelector(".pzp-pc") ?? document.body;
        const el = document.createElement("div");
        el.className = "cb-bm-toast";
        el.textContent = text;
        host.append(el);
        setTimeout(() => el.remove(), 2200);
    }

    async function addBookmark() {
        if (route.kind !== "live") return;
        const channelId = route.key;
        try {
            const watched = watchedWallTime(live ?? { openDate: null });
            const detail = await fetchLive(channelId);
            if (detail.status !== "OPEN") {
                toast("방송 중이 아니에요");
                return;
            }
            live = detail;
            if (watched.timeSource === "openDate+currentTime") {
                Object.assign(watched, watchedWallTime(detail));
            }
            const memo = prompt("북마크 메모 (비워도 돼요)", "");
            if (memo === null) return;
            const record: Bookmark = {
                id: crypto.randomUUID(),
                channelId,
                liveId: detail.liveId,
                videoId: detail.videoId,
                openDate: detail.openDate,
                wallTime: watched.wallTime,
                timeSource: watched.timeSource,
                memo: memo.trim(),
                liveTitle: detail.liveTitle,
                createdAt: Date.now(),
            };
            store([...load(), record]);
            renderedSignature = "";
            const offset = (record.wallTime - kstToMs(record.openDate)) / 1000;
            toast(`북마크 저장: ${formatClock(offset)}`);
            logger.info("bookmark saved {record}", { record });
        } catch (error) {
            logger.error("bookmark save failed {href} {error}", { href: location.href, error });
            toast("북마크 저장 실패 (콘솔 확인)");
        }
    }

    function ensureButton() {
        const bar = document.querySelector(".pzp-pc__bottom-buttons-right");
        if (!bar || bar.querySelector(".cb-bm-button")) return;
        const button = document.createElement("button");
        button.className = "cb-bm-button pzp-button pzp-pc-ui-button";
        button.setAttribute("aria-label", "북마크 (Alt+B)");
        button.innerHTML =
            '<span class="pzp-button__tooltip pzp-button__tooltip--top">북마크 (Alt+B)</span><span class="pzp-ui-icon">' +
            ICON +
            "</span>";
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            addBookmark();
        });
        bar.prepend(button);
    }

    // Each entry: { bookmark, offset (sec, replay offset or stream elapsed) }.
    function visibleEntries(): Entry[] {
        const all = load();
        const liveNow = live;
        const vodNow = vod;
        if (route.kind === "live" && liveNow) {
            const openMs = kstToMs(liveNow.openDate);
            return all
                .filter((b) => b.liveId === liveNow.liveId)
                .map((b) => ({ bookmark: b, offset: (b.wallTime - openMs) / 1000 }));
        }
        if (route.kind === "video" && vodNow) {
            return all
                .filter(
                    (b) =>
                        b.videoId === vodNow.videoId ||
                        (b.channelId === vodNow.channelId &&
                            b.openDate === vodNow.liveOpenDate)
                )
                .map((b) => ({
                    bookmark: b,
                    offset:
                        (b.wallTime - kstToMs(b.openDate)) / 1000 -
                        vodNow.earlierSec,
                }))
                .filter((e) => e.offset >= 0 && e.offset <= vodNow.duration);
        }
        return [];
    }

    // LIVE only: 0:00:00 (openDate) .. now, since the live seek bar only
    // covers the DVR window and cannot show the whole stream.
    let timelineSignature = "";
    function renderTimeline() {
        const player = document.querySelector<HTMLElement>(".pzp-pc");
        let bar = player?.querySelector<HTMLElement>(":scope > .cb-bm-tl");
        if (route.kind !== "live" || !live || live.status !== "OPEN" || !player) {
            document.querySelectorAll(".cb-bm-tl").forEach((t) => t.remove());
            timelineSignature = "";
            return;
        }
        if (!style.isConnected) document.head.append(style);
        if (!bar) {
            bar = document.createElement("div");
            bar.className = "cb-bm-tl";
            bar.innerHTML =
                '<span class="cb-bm-tl-start">0:00:00</span><div class="cb-bm-tl-track"><div class="cb-bm-tl-fill"></div><div class="cb-bm-tl-head"></div></div><span class="cb-bm-tl-end"></span>';
            player.append(bar);
            timelineSignature = "";
        }
        const openMs = kstToMs(live.openDate);
        const span = Math.max(1, Date.now() - openMs);
        const pct = (ms: number) => Math.min(100, Math.max(0, (100 * (ms - openMs)) / span));
        const watched = watchedWallTime(live);
        bar.querySelector(".cb-bm-tl-end")!.textContent = formatClock(span / 1000);
        bar.querySelector<HTMLElement>(".cb-bm-tl-fill")!.style.width = `${pct(watched.wallTime)}%`;
        const head = bar.querySelector<HTMLElement>(".cb-bm-tl-head")!;
        head.style.left = `${pct(watched.wallTime)}%`;
        head.dataset.wallTime = String(watched.wallTime);

        const track = bar.querySelector(".cb-bm-tl-track")!;
        const entries = visibleEntries();
        const signature = JSON.stringify(entries.map((e) => [e.bookmark.id, e.bookmark.memo]));
        if (signature !== timelineSignature) {
            timelineSignature = signature;
            track.querySelectorAll(".cb-bm-tl-mark").forEach((m) => m.remove());
            for (const { bookmark, offset } of entries) {
                const mark = document.createElement("div");
                mark.className = "cb-bm-tl-mark";
                mark.dataset.wallTime = String(bookmark.wallTime);
                mark.title = `${formatClock(offset)}${bookmark.memo ? ` ${bookmark.memo}` : ""}`;
                for (const type of ["click", "dblclick", "mousedown", "pointerdown"]) {
                    mark.addEventListener(type, (e) => e.stopPropagation());
                }
                track.append(mark);
            }
        }
        for (const mark of track.querySelectorAll<HTMLElement>(".cb-bm-tl-mark")) {
            mark.style.left = `${pct(Number(mark.dataset.wallTime))}%`;
        }
    }

    function seek(sec: number) {
        const video = document.querySelector("video");
        if (video && Number.isFinite(video.duration)) {
            video.currentTime = sec;
            setTimeout(() => {
                if (Math.abs(video.currentTime - sec) > 5) {
                    location.href = `${location.pathname}?currentTime=${Math.floor(sec)}`;
                }
            }, 1500);
            return;
        }
        location.href = `${location.pathname}?currentTime=${Math.floor(sec)}`;
    }

    function remove(id: string) {
        store(load().filter((b) => b.id !== id));
        renderedSignature = "";
    }

    function render() {
        const player = document.querySelector<HTMLElement>(".pzp-pc");
        const entries = route.kind === "none" ? [] : visibleEntries();
        const signature =
            route.kind + route.key + collapsed + JSON.stringify(entries.map((e) => [e.bookmark.id, e.offset]));
        let panel = player?.querySelector<HTMLElement>(":scope > .cb-bm-panel");
        const slider = document.querySelector<HTMLElement>(".pzp-pc__progress-slider");
        const markersPresent =
            route.kind !== "video" ||
            entries.length === 0 ||
            slider?.querySelector(".cb-bm-marker");
        if (signature === renderedSignature && panel && markersPresent) return;
        renderedSignature = signature;

        document.querySelectorAll(".cb-bm-marker").forEach((m) => m.remove());
        if (!player || route.kind === "none") {
            document.querySelectorAll(".cb-bm-panel").forEach((p) => p.remove());
            return;
        }
        if (!style.isConnected) document.head.append(style);
        if (!panel) {
            panel = document.createElement("div");
            panel.className = "cb-bm-panel";
            // Keep clicks and keys inside the panel away from the player.
            for (const type of ["click", "dblclick", "mousedown", "keydown"]) {
                panel.addEventListener(type, (e) => e.stopPropagation());
            }
            player.append(panel);
        }
        panel.classList.toggle("cb-collapsed", collapsed);
        panel.replaceChildren();
        const header = document.createElement("button");
        header.textContent = `${collapsed ? "▸" : "▾"} 북마크 ${entries.length}`;
        header.addEventListener("click", () => {
            collapsed = !collapsed;
            renderedSignature = "";
            render();
        });
        const list = document.createElement("ul");
        entries.sort((a, b) => a.offset - b.offset);
        for (const { bookmark, offset } of entries) {
            const item = document.createElement("li");
            const time = document.createElement("time");
            time.textContent = formatClock(offset);
            const memo = document.createElement("span");
            memo.textContent = bookmark.memo || "";
            memo.title = `${bookmark.liveTitle ?? ""} · ${bookmark.timeSource}`;
            const del = document.createElement("button");
            del.className = "cb-bm-del";
            del.textContent = "✕";
            del.title = "삭제";
            del.addEventListener("click", (e) => {
                e.stopPropagation();
                remove(bookmark.id);
                render();
            });
            item.append(time, memo, del);
            if (route.kind === "video") {
                item.dataset.seek = String(offset);
                item.addEventListener("click", () => seek(offset));
            }
            list.append(item);
        }
        panel.append(header, list);

        if (route.kind === "video" && slider && vod?.duration) {
            for (const { bookmark, offset } of entries) {
                const marker = document.createElement("div");
                marker.className = "cb-bm-marker";
                marker.style.left = `${(100 * offset) / vod.duration}%`;
                marker.title = `${formatClock(offset)} ${bookmark.memo || ""}`;
                for (const type of ["mousedown", "pointerdown"]) {
                    marker.addEventListener(type, (e) => e.stopPropagation());
                }
                marker.addEventListener("click", (e) => {
                    e.stopPropagation();
                    seek(offset);
                });
                slider.append(marker);
            }
        }
    }

    async function fetchVod(videoNo: string): Promise<VodInfo> {
        const c = await getJson<VideoResponse>(`${API}/v3/videos/${videoNo}`);
        const info: VodInfo = {
            videoNo: c.videoNo,
            videoId: c.videoId,
            liveOpenDate: c.liveOpenDate,
            duration: c.duration,
            channelId: c.channel?.channelId,
            earlierSec: 0,
        };
        const openMs = kstToMs(c.liveOpenDate);
        const publishMs = kstToMs(c.publishDate);
        const relevant = load().some(
            (b) =>
                b.videoId === info.videoId ||
                (b.channelId === info.channelId &&
                    b.openDate === info.liveOpenDate)
        );
        if (
            relevant &&
            c.videoType === "REPLAY" &&
            (publishMs - openMs) / 1000 > c.duration + EARLIER_PART_SLACK_SEC
        ) {
            const page = await getJson<VideoListResponse>(
                `${API}/v1/channels/${info.channelId}/videos?sortType=LATEST&pagingType=PAGE&page=0&size=50`
            );
            const candidates = (page?.data ?? []).filter((v) => {
                const t = kstToMs(v.publishDate);
                return (
                    v.videoType === "REPLAY" &&
                    v.videoNo < info.videoNo &&
                    t >= openMs &&
                    t < publishMs
                );
            });
            for (const v of candidates) {
                const part = await getJson<VideoResponse>(`${API}/v3/videos/${v.videoNo}`);
                if (part.liveOpenDate === info.liveOpenDate) {
                    info.earlierSec += part.duration;
                }
            }
            logger.info("earlier parts {count} {earlierSec}", {
                count: candidates.length,
                earlierSec: info.earlierSec,
            });
        }
        return info;
    }

    function onRoute() {
        const liveMatch = location.pathname.match(LIVE_RE);
        const videoMatch = location.pathname.match(VIDEO_RE);
        const next: Route = liveMatch
            ? { kind: "live", key: liveMatch[1] }
            : videoMatch
              ? { kind: "video", key: videoMatch[1] }
              : { kind: "none", key: "" };
        if (next.kind === route.kind && next.key === route.key) return;
        route = next;
        live = null;
        vod = null;
        renderedSignature = "";
        cachedRef = null;
        const current = next;
        const loader =
            next.kind === "live"
                ? fetchLive(next.key).then((d) => (live = d))
                : next.kind === "video"
                  ? fetchVod(next.key).then((d) => (vod = d))
                  : null;
        loader
            ?.then(() => {
                if (route === current) tick();
            })
            .catch((error) =>
                logger.error("bookmark route load failed {href} {error}", { href: location.href, error })
            );
    }

    function tick() {
        try {
            onRoute();
            if (route.kind === "live") ensureButton();
            render();
            renderTimeline();
        } catch (error) {
            logger.error("bookmark tick failed {href} {error}", { href: location.href, error });
        }
    }

    setInterval(() => {
        if (route.kind === "live") tick();
    }, 2000);

    window.addEventListener(
        "keydown",
        (event) => {
            if (
                route.kind !== "live" ||
                event.code !== "KeyB" ||
                !event.altKey ||
                event.ctrlKey ||
                event.metaKey ||
                event.shiftKey
            ) {
                return;
            }
            const target = event.target as HTMLElement | null;
            if (
                target?.isContentEditable ||
                /^(input|textarea|select)$/i.test(target?.tagName ?? "")
            ) {
                return;
            }
            event.preventDefault();
            event.stopPropagation();
            addBookmark();
        },
        true
    );

    let scheduled = false;
    new MutationObserver(() => {
        if (scheduled) return;
        scheduled = true;
        setTimeout(() => {
            scheduled = false;
            tick();
        }, 300);
    }).observe(document, { subtree: true, childList: true });
})();
