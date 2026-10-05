import starlight from "@astrojs/starlight";
import { defineConfig } from "astro/config";

// Served from GitHub Pages at https://ranolp.github.io/chan-necessity/.
// Starlight does not prefix `base` onto hero actions or Markdown links,
// so content links are written with /chan-necessity/ in full.
export default defineConfig({
    site: "https://ranolp.github.io",
    base: "/chan-necessity",
    integrations: [
        starlight({
            title: "chan-necessity",
            description: "치지직을 더 편하게 보는 유저스크립트",
            defaultLocale: "root",
            // Korean only for now; English can be added later as an `en` locale.
            locales: { root: { label: "한국어", lang: "ko" } },
            social: [{ icon: "github", label: "GitHub", href: "https://github.com/RanolP/chan-necessity" }],
            customCss: ["./src/styles/custom.css"],
            sidebar: [
                { label: "시작하기", items: [{ label: "설치와 업데이트", slug: "install" }] },
                {
                    label: "기능",
                    items: [
                        { label: "1080p 화질", slug: "features/deny-grid" },
                        { label: "통나무 파워 자동 수집", slug: "features/auto-claim-logs" },
                        { label: "북마크와 다시보기 타임라인", slug: "features/bookmarks" },
                        { label: "실시간 자막", slug: "features/subtitles" },
                        { label: "스플릿 뷰", slug: "features/split-view" },
                    ],
                },
                { label: "도움말", items: [{ label: "자주 묻는 질문", slug: "faq" }] },
            ],
        }),
    ],
});
