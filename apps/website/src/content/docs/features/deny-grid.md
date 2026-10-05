---
title: 화질
description: Windows에서 480p 항목을 1080p로 바꿔 받고, 마지막에 고른 화질을 기억해요.
---

치지직은 일부 환경에서 1080p 화질을 막아 둬요. chan-necessity는 화질 메뉴의 480p 항목을 1080p로 바꿔서 그 제한을 우회해요. 이 부분은 [refracta/FUCK-CHZZK-GRID-CHROME](https://github.com/refracta/FUCK-CHZZK-GRID-CHROME)(MIT)을 바탕으로 했어요.

## Windows에서만 켜져요

우회는 Windows에서만 동작해요. 다른 운영체제에서는 아무것도 바꾸지 않고, 콘솔에 아래 줄만 남겨요.

```
chan-necessity·deny-grid non-Windows platform detected; bypass disabled
```

## 어떻게 동작하나요

- 플레이어가 480p 재생목록(`.m3u8`) 주소를 요청하면, 그 주소를 1080p 재생목록으로 바꿔서 받아요. `fetch`와 XHR 양쪽을 다 감싸요.
- 화질 메뉴의 480p 항목 이름이 「1080p」로 바뀌고, 옆에 「with ChzzkBest」 배지가 붙어요.
- 메뉴 쪽 변경은 라이브 페이지(`/live/<채널 ID>`)에서만 해요.

<!-- TODO: screenshot — 플레이어 화질 메뉴에 「1080p」 항목과 「with ChzzkBest」 배지가 보이는 모습 -->

## 마지막 화질 기억하기

화질을 고르면 그 이름을 브라우저 `localStorage`의 `quality-text` 키에 저장해요. 다음에 플레이어가 새로 뜨면 그 화질을 한 번 다시 골라 줘요.

- 저장된 값이 없으면 기본값은 360p예요.
- 다시 고를 때는 설정 패널을 잠깐 열고, 항목을 누른 뒤 닫아요. 화면에서 설정 패널이 한 번 깜빡일 수 있어요.
- 직접 설정 패널을 열어 둔 동안에는 기다렸다가 닫힌 뒤에 골라요.
- 플레이어 하나당 한 번만 다시 골라요. 그 뒤에는 직접 고른 화질을 건드리지 않아요.
- 0.1.1 이전 버전이 저장한 「FUCK GRID」 이름도 같은 「1080p with ChzzkBest」 항목으로 읽어요.

## 관련 문서

- [설치](/chan-necessity/install/)
- [FAQ](/chan-necessity/faq/)
