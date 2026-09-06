# Kimi K3 — KDA (Kimi Delta Attention) 심층분석

- 공식 저장소: https://github.com/MoonshotAI/Kimi-K3 (Kimi K3 License) — 실제로 확인해보니 config/processor 코드만 있고 모델링 코드는 없음(가중치만 96분할 safetensors, 총 1.56TB)
- 논문: "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692)
- 코드 근거: 커뮤니티 교육용 재구현 `hwilner/kimi-delta-attention` (논문 수식과 대조 확인) + 공식 `configuration_kimi_k3.py`의 `kda_layers`/`full_attn_layers`/`attn_res_block_size` 필드로 하이브리드 구조 확인
- 공식 고성능 커널: `MoonshotAI/FlashKDA`, `fla-org/flash-linear-attention`(`chunk_kda`)

## 수식과 코드

```
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
o_t = S_t^T q_t
```

```python
class KDACore(nn.Module):
    def forward(self, queries, keys, values, alpha, beta, initial_state=None):
        # state: (B, head_dim, value_dim) — 시퀀스 길이와 무관하게 고정 크기
        for t in range(seq_len):
            q_t, k_t, v_t = queries[:, t, :], keys[:, t, :], values[:, t, :]
            alpha_t, beta_t = alpha[:, t, :], beta[:, t, :]

            # Step 1: 채널별 감쇠 — Diag(α_t) S_{t-1}
            state = alpha_t.unsqueeze(-1) * state

            # Step 2: "지우기" — β_t k_t k_t^T Diag(α_t) S_{t-1}
            # 현재 state에서 k_t 방향 성분을 제거 (delta rule의 오차 수정 부분)
            k_state = torch.bmm(k_t.unsqueeze(-1), torch.bmm(k_t.unsqueeze(1), state))
            state = state - beta_t.unsqueeze(-1) * k_state

            # Step 3: "쓰기" — β_t k_t v_t^T (새 연관 기억 추가)
            kv_outer = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(1))
            state = state + beta_t.unsqueeze(-1) * kv_outer

            # 출력 = 현재 state에 쿼리를 곱한 것
            o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
```

## 동역학적 해석

일반 attention은 과거 토큰 전체(KV 캐시, 길이에 비례해 계속 커짐)를 매번 다시 훑지만, KDA는 **`(head_dim × value_dim)` 크기로 고정된 상태 행렬 `S_t` 하나만 들고 다니는 유한상태 선형 RNN**. 시퀀스가 아무리 길어져도 상태 크기는 불변 — 1M 토큰 컨텍스트를 감당하는 이유(전체 KV 저장 안 함).

**"델타(Delta)" 이름의 유래**: 1960년대 Widrow-Hoff 델타 학습 규칙(온라인 최소자승 오차 수정)을 시퀀스 위에서 매 스텝 실행하는 것과 동형. "새 (k,v) 쌍을 배울 때, 기존 메모리에서 k 방향으로 이미 저장된 내용을 먼저 지우고(Step 2) 그 자리에 새 값을 쓴다(Step 3)"는 게 오차만큼 수정하는 delta rule 그 자체. 메모리 행렬 `S_t`를 연상 기억(associative memory)으로 보고, 그 위에서 온라인 경사하강과 유사한 갱신을 매 토큰마다 도는 것 — Hopfield network/fast-weight 프로그래머 계열 아이디어의 현대판.

**Kimi 고유 개선점**: 기존 Gated DeltaNet은 스칼라 감쇠 `α_t`(모든 채널이 동일하게 잊음) 하나만 썼는데, KDA는 `Diag(α_t)`(채널별로 다른 잊음 속도)로 확장 — 채널마다 다른 시간축(multi-timescale)의 기억을 학습 가능.

**하이브리드 배치**: 공식 config의 `kda_layers`/`full_attn_layers` 필드가 확인해주듯, 대부분의 레이어는 KDA(값싼 선형 동역학)로 쓰고, 일부 레이어만 기존 quadratic full attention을 남겨 정확한 전역 검색이 필요한 경우를 커버하는 하이브리드 구조(Jamba/Zamba/Griffin 등 최근 장문맥 하이브리드 아키텍처들과 같은 패턴).

`attn_res_block_size`(Attention Residuals, AttnRes)는 config 필드로 존재 확인은 됐으나, 공개된 실제 구현 코드를 찾지 못해 정확한 메커니즘은 추가 확인 필요(추정: attention 계산에 특화된 블록 단위 residual 변형 — DeepSeek mHC와 유사한 발상일 가능성).

## 다른 두 모델과의 축 비교

Kimi K3의 KDA는 **토큰 간(시퀀스축) 정보 전달** 자체를 재설계한 것으로, DeepSeek-V4의 mHC(**레이어 간, 깊이축** 재설계)와 서로 다른 축을 겨냥한다 — 둘 다 "표준 attention/residual"의 구조적 한계를 다른 위치에서 공략.
