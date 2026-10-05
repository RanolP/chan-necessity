// The userscript metadata block, rendered into the bundle's banner at build time.
import pkg from "./package.json" with { type: "json" };

export const PAGES_URL = "https://ranolp.github.io/chan-necessity/";

export interface UserscriptMeta {
    name: string;
    namespace: string;
    version: string;
    description: string;
    match: string[];
    runAt: "document-start" | "document-end" | "document-idle";
    injectInto: "page" | "content" | "auto";
    grant: string[];
    connect: string[];
    homepageURL: string;
    supportURL: string;
    updateURL: string;
    downloadURL: string;
    license: string;
}

export const meta: UserscriptMeta = {
    name: "chan-necessity",
    namespace: "https://github.com/RanolP/chan-necessity",
    version: pkg.version,
    description:
        "치지직 그리드 요구 우회 + 통나무 파워 자동 수집 + 라이브 북마크 → 다시보기 연동 + 파이어폭스 클립 편집기 + 좌우 밸런스 + 실시간 자막(Qwen3-ASR, 스트리밍·VAD) + 스플릿 뷰(최대 3채널, 칸별 좌우 패닝)",
    match: ["https://chzzk.naver.com/*"],
    runAt: "document-start",
    injectInto: "page",
    grant: ["unsafeWindow", "GM_getValue", "GM_setValue", "GM_xmlhttpRequest"],
    // Release assets for the STT decode engine redirect through these hosts.
    connect: ["github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"],
    homepageURL: PAGES_URL,
    supportURL: "https://github.com/RanolP/chan-necessity/issues",
    updateURL: `${PAGES_URL}chan-necessity.user.js`,
    downloadURL: `${PAGES_URL}chan-necessity.user.js`,
    license: "Apache-2.0",
};

export function renderMeta(m: UserscriptMeta): string {
    const rows: [string, string][] = [
        ["name", m.name],
        ["namespace", m.namespace],
        ["version", m.version],
        ["description", m.description],
        ...m.match.map((v): [string, string] => ["match", v]),
        ["run-at", m.runAt],
        ["inject-into", m.injectInto],
        ...m.grant.map((v): [string, string] => ["grant", v]),
        ...m.connect.map((v): [string, string] => ["connect", v]),
        ["homepageURL", m.homepageURL],
        ["supportURL", m.supportURL],
        ["updateURL", m.updateURL],
        ["downloadURL", m.downloadURL],
        ["license", m.license],
    ];
    return ["// ==UserScript==", ...rows.map(([k, v]) => `// @${k.padEnd(12)} ${v}`), "// ==/UserScript==", ""].join("\n");
}
