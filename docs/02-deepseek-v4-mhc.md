# DeepSeek-V4 — mHC (Manifold-Constrained Hyper-Connections) 심층분석

- 코드 출처: `huggingface/transformers` main 브랜치 `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py` (MIT)
- 전용 추론 저장소는 없고, 가중치는 HuggingFace `deepseek-ai` 조직에 공개 (Pro/Flash/Flash-DSpark, 총 685B: 671B 본체 + 14B MTP 모듈)

## 아키텍처 개요 (문서 기준)

V3의 MLA를 대체해 세 종류 하이브리드 어텐션(슬라이딩 윈도우 풀 어텐션 / 압축 희소 어텐션 CSA / 고압축 어텐션 HCA)을 레이어별 혼용. MoE 라우팅은 초반 레이어 `hash_moe`(정적 토큰-전문가 매핑, 부트스트래핑용) → 이후 표준 top-k `moe`로 전환. 활성화 함수는 MoE 게이트 `Sqrt(Softplus(·))`, FFN은 clamped SwiGLU. 표준 residual 대신 **mHC**를 채택.

## mHC 실제 코드

```python
class DeepseekV4HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections (mHC) (Xie et al., 2026)
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))   # pre / post / comb 각각의 학습 스케일

    def forward(self, hidden_streams: torch.Tensor):
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps          # 스트림 collapse 가중치
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)                # 범위 (0,2): 레이어 출력 게인
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)          # Sinkhorn-Knopp:
        for _ in range(self.hc_sinkhorn_iters - 1):                          # 행/열 정규화를 번갈아 반복
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)      # → 이중확률행렬(doubly-stochastic)로 근사 수렴
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed
```

디코더 레이어에서의 사용:
```python
class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states, input_ids=None, **kwargs):
        # hidden_states: [B, S, hc_mult, hidden]  ← H개 병렬 residual 스트림
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        hidden_states = post.unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return post.unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), hidden_states
        )
```

## 동역학적 해석

표준 residual `x_{l+1} = x_l + F(x_l)`은 상태가 1개(H=1), 전이행렬이 항등행렬로 고정된 선형계다. mHC는 이걸 **H개(`hc_mult`) 병렬 스트림을 갖는 학습된 선형 상태공간모델**로 일반화:

```
x_{l+1} = A_l · x_l  +  B_l ⊙ u_l
```
- `x_l` = `hidden_streams` (H개의 D차원 상태)
- `A_l` = `comb^T` — 스트림 간 전이행렬, 매 레이어·매 토큰마다 학습
- `u_l` = 단일 스트림만 계산되는 attn/mlp 출력, H개 스트림에 브로드캐스트
- `B_l` = `post` — 레이어별 출력 게인(범위 (0,2))
- `collapsed`(=`pre` 가중합)는 H개 상태를 하나로 읽어내는(readout) C행렬

**왜 Sinkhorn-Knopp인가**: 이중확률행렬(doubly-stochastic)은 Perron-Frobenius에 의해 스펙트럴 반경이 정확히 1이라, `comb^T @ x`가 절대 `x`의 크기를 증폭시키지 않음(비확장적). 깊이가 아무리 깊어져도(40층이든 400층이든) 신호 전파가 구조적으로 안정 — Wan2.2의 zero-init(초기값을 눌러 안정화)과 달리, **전이행렬 자체의 스펙트럴 특성을 제약해서** 안정성을 보장하는 방식.

**Birkhoff–von Neumann 정리로 보면**: 이중확률행렬의 극점은 순열행렬이고, 임의의 이중확률행렬은 순열행렬들의 볼록조합. 즉 `comb^T`는 "H개의 병렬 residual 스트림 내용을 토큰마다·레이어마다 학습된 비율로 소프트하게 재배치(순열)"하는 것과 동일 — 정보를 잃지도 폭발시키지도 않으면서 depth 방향으로 유연하게 재배치하는 채널.

**`post` 게이트(범위 (0,2))**: stochastic depth의 학습된·결정론적·연속적 버전. 값이 0에 가까우면 이번 레이어를 사실상 스킵.

## Wan2.2와의 대조

| | 스위칭 축 | 무엇을 제약하는가 | 안정화 원리 |
|---|---|---|---|
| Wan2.2 두 전문가 | 시간(timestep) | 어느 파라미터(모델)를 쓸지 통째로 교체 | 겹치는 노이즈 구간 학습으로 근사 연속 |
| DeepSeek-V4 mHC | 깊이(layer), 토큰마다 | H개 병렬 residual 스트림의 혼합행렬 | 이중확률 제약으로 스펙트럴 반경 1 고정 |
