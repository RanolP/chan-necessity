// ---- 지금 보고 있는 방송 시각 ------------------------------------------
// Which moment of a live broadcast the viewer is watching, as a wall-clock
// time, and when that broadcast opened. Bookmarks store the moment; the
// caption history shows it as time elapsed since the open.
import { getLogger } from "./logtape.ts";

const logger = getLogger(["broadcast-time"]);

export const API = "https://api.chzzk.naver.com/service";

export type TimeSource = "player" | "openDate+currentTime" | "now";

export interface LiveInfo {
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

const pageWindow = typeof unsafeWindow === "undefined" ? window : unsafeWindow;

// Chzzk API dates are KST without a zone ("2026-01-01 09:00:00").
export const kstToMs = (text: string | null | undefined) => (text ? Date.parse(text.replace(" ", "T") + "+09:00") : NaN);

export async function getJson<T>(url: string): Promise<T> {
    const response = await pageWindow.fetch(url, {
        credentials: "include",
    });
    const body = await response.text();
    if (!response.ok) {
        throw new Error(`[ChzzkBest] ${url} → ${response.status}: ${body.slice(0, 200)}`);
    }
    try {
        return JSON.parse(body).content as T;
    } catch {
        throw new Error(`[ChzzkBest] ${url} → bad JSON: ${body.slice(0, 200)}`);
    }
}

// Every successful fetch refreshes the channel's cached open date, so a
// caller that always fetches (bookmarks, for the live status) keeps the
// cache current for the callers that only need the open date.
const openDates = new Map<string, Promise<string>>();

export async function fetchLive(channelId: string): Promise<LiveInfo> {
    const c = await getJson<LiveDetailResponse>(`${API}/v3/channels/${channelId}/live-detail`);
    let videoId: string | null = null;
    try {
        videoId = JSON.parse(c.livePlaybackJson).meta.videoId ?? null;
    } catch (error) {
        logger.warn("livePlaybackJson unreadable {channelId} {error}", { channelId, error });
    }
    openDates.set(channelId, Promise.resolve(c.openDate));
    return {
        channelId,
        liveId: c.liveId,
        openDate: c.openDate,
        liveTitle: c.liveTitle,
        status: c.status,
        videoId,
    };
}

// The open date of the channel's current broadcast, fetched once per
// channel; a failed fetch is forgotten so the next call retries.
export function liveOpenDate(channelId: string): Promise<string> {
    let pending = openDates.get(channelId);
    if (!pending) {
        pending = fetchLive(channelId).then((live) => live.openDate);
        openDates.set(channelId, pending);
        pending.catch(() => {
            if (openDates.get(channelId) === pending) openDates.delete(channelId);
        });
    }
    return pending;
}

// Chzzk's player component exposes getProgramDateTime() through a React
// imperative ref; it is startDate + currentTime of the core player, so
// it already accounts for latency and 타임머신 rewind.
function findPlayerRef(): PlayerRef | null {
    const host = document.querySelector("#root") as (Element & Record<string, unknown>) | null;
    const key = host && Object.keys(host).find((k) => k.startsWith("__reactContainer$"));
    if (!key) return null;
    const stack = [host[key] as Fiber];
    for (let visited = 0; stack.length && visited < 50_000; visited++) {
        const fiber = stack.pop();
        if (!fiber) continue;
        let hook = fiber.memoizedState;
        for (let i = 0; hook && typeof hook === "object" && "next" in hook && i < 80; i++, hook = hook.next) {
            const ref = hook.memoizedState;
            const current: PlayerRef | null = ref && typeof ref === "object" && "current" in ref ? (ref.current as PlayerRef | null) : null;
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

// Call when the page changed to a different broadcast.
export function forgetPlayerRef() {
    cachedRef = null;
}

// `openMs` is the broadcast's open time, NaN when not known yet; it bounds
// what counts as a plausible watched moment and anchors the fallback.
export function watchedWallTime(openMs: number): { wallTime: number; timeSource: TimeSource } {
    const now = Date.now();
    const plausible = (t: number | undefined): t is number => t !== undefined && Number.isFinite(t) && t <= now + 5_000 && !(t < openMs - 60_000);
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
