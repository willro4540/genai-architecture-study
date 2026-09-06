# genai-architecture-study

비공개(폐쇄형) 상용 생성모델 — GPT-6 Astra(OpenAI), Seedance 2.0(ByteDance), Vidu Q3(ShengShu) — 의 오픈소스나 코드가 있는지 확인하다가, 전부 API 전용 비공개 모델이라는 걸 확인하고 대신 **실제 코드가 공개된 대표적인 오픈소스 대안들을 클론해서 직접 뜯어본 기록**.

## 왜 이 저장소인가

- GPT-6 Astra / Seedance 2.0 / Vidu Q3 세 모델 모두 가중치·코드 비공개, API로만 서비스됨 (`docs/00-closed-source-context.md`)
- 대신 같은 카테고리(프론티어 LLM / 오디오-비디오 생성)의 **진짜 오픈소스** 모델을 골라 GitHub에서 직접 클론하고, 코드를 줄 단위로 읽으며 아키텍처를 분석함
- 단순 요약이 아니라 실제 소스 파일 인용 + 동역학적(dynamical-systems) 관점의 심층 분석 위주

## 분석 대상

| 모델 | 카테고리 | 라이선스 | 문서 |
|---|---|---|---|
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) (Alibaba) | 텍스트/이미지/음성→비디오, 캐릭터 애니메이션 | Apache 2.0 | [docs/01-wan2.2.md](docs/01-wan2.2.md) |
| [DeepSeek-V4](https://huggingface.co/collections/deepseek-ai/deepseek-v4) (mHC) | LLM (residual 재설계) | MIT | [docs/02-deepseek-v4-mhc.md](docs/02-deepseek-v4-mhc.md) |
| [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) (KDA) | LLM (attention 대체) | Kimi K3 License | [docs/03-kimi-k3-kda.md](docs/03-kimi-k3-kda.md) |
| [LTX-2.5](https://github.com/Lightricks/LTX-2) (Lightricks) | native 오디오-비디오 동시생성 | 오픈 웨이트 | [docs/04-ltx-2.5.md](docs/04-ltx-2.5.md) |

## 총정리 — 네 모델의 "동역학적 재설계 지점"

표준 트랜스포머/디퓨전의 디폴트 동역학(고정 residual add, 고정 quadratic attention, 애드온식 조건화, 단일 벡터장)이 갖는 구조적 한계를, 각 모델이 서로 다른 축에서 재설계함:

| 모델 | 무엇을 재설계했나 | 축 |
|---|---|---|
| DeepSeek-V4 (mHC) | 레이어 간 residual 흐름 — H개 병렬 스트림 + 이중확률(doubly-stochastic) 제약 | 깊이(depth) |
| Kimi K3 (KDA) | 토큰 간 정보 전달(attention 자체) — 유한상태 delta-rule 선형 RNN | 시퀀스(sequence) |
| LTX-2.5 | 모달리티 간 결합 — 비디오/오디오 두 동역학계의 게이트된 양방향 커플링 | 모달리티(modality) |
| Wan2.2 | 시간(diffusion timestep)에 따른 벡터장 스위칭 + 조건 신호별 차등 주입 | 시간(diffusion time) |

자세한 근거와 인용 코드는 각 문서 참고.

## 참고

- 실제 클론한 원본 저장소는 각자의 라이선스(Apache 2.0 / MIT 등)를 따르며, 이 저장소엔 원본 코드를 재배포하지 않고 **분석 문서만** 포함함.
