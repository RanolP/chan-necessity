---
title: 설치
description: Violentmonkey에 chan-necessity 유저스크립트를 설치하고 업데이트하는 방법이에요.
---

chan-necessity는 치지직(`https://chzzk.naver.com/*`)에서 도는 유저스크립트예요. 브라우저에 유저스크립트 관리자를 깔고, 그 관리자에 스크립트를 추가하면 돼요.

## 준비물

유저스크립트 관리자는 [Violentmonkey](https://violentmonkey.github.io/)를 써요. 스크립트는 페이지 컨텍스트에 주입되고(`@inject-into page`), 문서가 만들어지는 순간부터(`@run-at document-start`) 돌아요.

## 설치하기

1. 브라우저에 Violentmonkey 확장을 설치해요.
2. 아래 주소를 열어요. Violentmonkey가 설치 화면을 띄워요.
3. 설치 화면에서 설치를 누르고, 치지직 탭을 새로고침해요.

```
https://ranolp.github.io/chan-necessity/chan-necessity.user.js
```

<!-- TODO: screenshot — Violentmonkey 설치 확인 화면에 chan-necessity 이름과 권한 목록이 보이는 모습 -->

## 스크립트가 요청하는 권한

설치 화면에 아래 권한이 나와요. 각각 이런 데 써요.

- `GM_getValue`, `GM_setValue`: 북마크, 자막 설정, 좌우 밸런스, 스플릿 뷰 배치를 저장해요.
- `GM_xmlhttpRequest`: 실시간 자막 엔진 파일을 GitHub 릴리스에서 받아요.
- `unsafeWindow`: 치지직 페이지 쪽 객체에 접근해요.

외부 접속(`@connect`)은 `github.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com` 세 곳뿐이에요.

## 업데이트

업데이트 주소와 다운로드 주소가 둘 다 위 설치 주소와 같아요. 그래서 Violentmonkey의 업데이트 확인을 돌리면 새 버전을 같은 주소에서 받아 와요.

## 설치 뒤에 생기는 것들

- 화질 메뉴의 1080p 우회는 Windows에서만 켜져요. [화질](/chan-necessity/features/deny-grid/)을 봐 주세요.
- 채팅창의 통나무 파워 받기 버튼을 자동으로 눌러요. [통나무 파워](/chan-necessity/features/auto-claim-logs/)를 봐 주세요.
- 플레이어에 북마크 버튼이 생겨요. [북마크](/chan-necessity/features/bookmarks/)를 봐 주세요.
- 플레이어 버튼 줄과 설정 메뉴에 「실시간 자막」이 생겨요. [실시간 자막](/chan-necessity/features/subtitles/)을 봐 주세요.
- 플레이어 설정 메뉴에 「좌우 밸런스」 슬라이더가 생겨요. [좌우 밸런스](/chan-necessity/features/split-view/#좌우-밸런스)를 봐 주세요.
- 플레이어와 사이드바에 「스플릿 뷰」 진입점이 생겨요. [스플릿 뷰](/chan-necessity/features/split-view/)를 봐 주세요.
- Firefox에서 `/clip-editor`(클립 편집기)를 열 수 있어요. 그 페이지에서만 브라우저를 Chrome 140으로 알려서, 편집기 팝업이 Firefox를 거부하지 않게 해요. 클립 인코딩은 원래대로 서버에서 해요.

## 문제가 생기면

자주 묻는 것은 [FAQ](/chan-necessity/faq/)에 모았어요. 거기 없는 문제는 [GitHub 이슈](https://github.com/RanolP/chan-necessity/issues)에 남겨 주세요. 스크립트 로그는 브라우저 콘솔에 `chan-necessity·<기능>` 분류가 붙은 줄로 남아요(예: `chan-necessity·captions`). 디버그 로그까지 보려면 Violentmonkey의 스크립트 Values 탭에서 `dev.logLevel`을 `"debug"`로 넣어 주세요.

스크립트는 Apache-2.0 라이선스예요.
