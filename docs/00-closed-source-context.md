# 왜 오픈소스 대안으로 갔는가 — 폐쇄형 모델 3종 확인 기록

## GPT-6 Astra (OpenAI)
- 2026-09-03 공개, ChatGPT/API/Azure/AWS Bedrock으로만 서비스
- 코드·가중치 전혀 공개 안 됨. OpenAI는 GPT-2 이후 원래 폐쇄형 정책
- Preparedness Framework 기준 사이버보안 Critical 등급 최초 모델

## Seedance 2.0 (ByteDance)
- fal.ai API로 서비스되는 비디오 생성 모델(최대 15초 720p, 오디오 동시생성)
- 가중치·코드 미공개, HuggingFace 공식 배포 없음
- GitHub의 관련 저장소(`amrrs/seedance-2.0-api`, `SamurAIGPT/seedance-2-generator` 등)는 전부 **API 래퍼/SaaS 보일러플레이트**일 뿐, 모델 자체 코드 아님

## Vidu Q3 (ShengShu Technology)
- Q3 Pro/Turbo/Mix 라인업, 실시간·스토리 기반 비디오 생성
- ShengShu 공식 GitHub(`shengshu-ai`)의 `vidu-skills`(CLI), `vidu-mcp`(MCP 래퍼), `Vidu-S1`(설치법도 없이 웹 링크만) — 전부 `VIDU_TOKEN` API 키로 호출하는 클라이언트일 뿐, 모델 코드/가중치 없음

## 결론
세 모델 다 **뜯어볼 오픈소스 코드/가중치가 존재하지 않음**. 대신 같은 카테고리에서 실제로 코드·가중치가 공개된 대표적인 대안(Wan2.2, DeepSeek-V4, Kimi K3, LTX-2.5)을 찾아 직접 분석함.
