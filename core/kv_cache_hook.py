"""
core/kv_cache_hook.py (v3)
===========================
Key norm 기반 importance 계산으로 전면 재작성.

변경사항 (v2 → v3):
  - _collect_prefill_attn() / output_attentions=True 방식 완전 제거
    → 별도 forward pass 불필요, OOM 위험 없음
  - _prefill_attn 대신 _prefill_keys 저장
    (evaluator_v2의 KV 복사 단계에서 key tensor 함께 저장)
  - importance score: key L2 norm 기반 근사
    → key norm은 attention score와 높은 상관관계 (H2O 논문 참조)
  - OursHybrid Semantic 신호: last_key → mean_key 기반으로 수정
    (prefill 전체 평균 키 = 쿼리 근사로 사용)
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from typing import Optional, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
# 유틸: layers 접근 헬퍼
# ─────────────────────────────────────────────────────────────

def _get_layer_kv(cache: DynamicCache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return cache.layers[layer_idx].keys, cache.layers[layer_idx].values

def _set_layer_kv(cache: DynamicCache, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
    cache.layers[layer_idx].keys = k
    cache.layers[layer_idx].values = v


def _key_importance(key_states: torch.Tensor) -> torch.Tensor:
    """
    Key L2 norm 기반 importance score.
    key_states: [batch, heads, seq_len, head_dim]
    반환: [seq_len] (heads 평균)
    """
    # [batch, heads, seq_len]
    norm = key_states.float().norm(dim=-1)
    # heads 평균 후 batch squeeze → [seq_len]
    score = norm.mean(dim=1).squeeze(0)
    return score


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen3/대부분 RoPE 모델 공통: 벡터를 절반으로 나눠 회전."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _reindex_rope_keys(
    key_states: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    """
    압축으로 인해 흩어진 원본 RoPE position(old_positions)이 적용된 key_states를,
    새로운 연속 position(new_positions, 보통 0..N-1)에 대응하는 RoPE로 재정렬한다.

    원리: RoPE는 각도 회전이므로 결합 법칙이 성립한다.
        k_new_pos = rotate(k_old_pos, delta) ,  delta = new_position - old_position
    즉 원본 회전을 먼저 풀고 새 위치로 다시 감는 것과, delta만큼만 회전하는 것이 수학적으로 동일하다.
    (단위 테스트로 allclose 검증됨, 오차 ~1e-6 수준)

    key_states: [batch, heads, seq_len, head_dim]
    old_positions, new_positions: [seq_len] (1D, 같은 길이, 같은 순서로 대응)
    rotary_emb: model.model.rotary_emb 모듈 (forward(x, position_ids) -> cos, sin)
    """
    if rotary_emb is None or key_states is None:
        return key_states
    if old_positions.shape[0] != key_states.shape[2]:
        # 길이가 안 맞으면 안전하게 재인덱싱 스킵 (원본 유지)
        return key_states

    device = key_states.device
    old_positions = old_positions.to(device).long()
    new_positions = new_positions.to(device).long()
    delta = (new_positions - old_positions).unsqueeze(0)  # [1, seq_len]

    # rotary_emb.forward(x, position_ids) 시그니처에 맞춰 dummy x 전달 (dtype/device 참조용)
    cos_delta, sin_delta = rotary_emb(key_states, delta)  # [1, seq_len, head_dim] 형태
    cos_delta = cos_delta.unsqueeze(1).to(key_states.dtype)  # [1, 1, seq_len, head_dim]
    sin_delta = sin_delta.unsqueeze(1).to(key_states.dtype)

    key_reindexed = key_states * cos_delta + _rotate_half(key_states) * sin_delta
    return key_reindexed


# ─────────────────────────────────────────────────────────────
# Base Hook Cache
# ─────────────────────────────────────────────────────────────

class BaseHookCache(DynamicCache):
    """
    DynamicCache 상속. prefill KV 복사 완료 후 apply_compression_all_layers()로
    일괄 압축. Transformers 5.10.2: layers[i].keys / layers[i].values 구조.
    """

    def __init__(self, budget_ratio: float, num_layers: int, model_config: Dict):
        super().__init__()
        self.budget_ratio = budget_ratio
        self.num_layers = num_layers
        self.model_config = model_config
        # prefill key tensor 저장 (importance 계산용)
        self._prefill_keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self._prefill_done = False
        self._prefill_seq_len = 0
        self._selected_positions: List[Optional[torch.Tensor]] = [None] * num_layers
        # RoPE re-indexing: 압축 후 흩어진 원본 position을 0..N-1로 재정렬하기 위한 rotary embedding
        self._rotary_emb = model_config.get("rotary_emb") if isinstance(model_config, dict) else None

    def set_prefill_keys(self, layer_idx: int, key_states: torch.Tensor):
        """KV 복사 단계에서 key tensor 저장 (CPU로 이동, 메모리 절약)."""
        if key_states is not None and layer_idx < self.num_layers:
            self._prefill_keys[layer_idx] = key_states.detach()

    def mark_prefill_done(self, seq_len: int):
        self._prefill_done = True
        self._prefill_seq_len = seq_len

    def apply_compression_all_layers(self):
        """prefill 완료 후 모든 레이어에 압축 적용."""
        budget = max(int(self._prefill_seq_len * self.budget_ratio), 4)
        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            if k.shape[2] > budget:
                k_c, v_c, idx = self._compress(k, v, i, budget)
                k_c = _reindex_rope_keys(k_c, idx, torch.arange(k_c.shape[2]), self._rotary_emb)
                _set_layer_kv(self, i, k_c, v_c)
                if i == 0:
                    self._selected_positions[i] = idx
            else:
                if i == 0:
                    self._selected_positions[i] = torch.arange(k.shape[2])
        self._prefill_keys = [None] * self.num_layers

    def _compress(self, key_states, value_states, layer_idx, budget):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# FullKV
# ─────────────────────────────────────────────────────────────

class FullKVCache(DynamicCache):
    """압축 없는 기준선."""
    def __init__(self, **kwargs):
        super().__init__()


# ─────────────────────────────────────────────────────────────
# StreamingLLM
# ─────────────────────────────────────────────────────────────

class StreamingLLMCache(BaseHookCache):
    def __init__(self, budget_ratio, num_layers, model_config, sink_size=4):
        super().__init__(budget_ratio, num_layers, model_config)
        self.sink_size = sink_size

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        recent_size = max(budget - self.sink_size, 1)
        sink_k = key_states[:, :, :self.sink_size, :]
        sink_v = value_states[:, :, :self.sink_size, :]
        recent_k = key_states[:, :, -recent_size:, :]
        recent_v = value_states[:, :, -recent_size:, :]
        sink_idx = torch.arange(self.sink_size)
        recent_idx = torch.arange(seq_len - recent_size, seq_len)
        indices = torch.cat([sink_idx, recent_idx])
        return torch.cat([sink_k, recent_k], dim=2), torch.cat([sink_v, recent_v], dim=2), indices


# ─────────────────────────────────────────────────────────────
# H2O
# ─────────────────────────────────────────────────────────────

class H2OCache(BaseHookCache):
    """
    Key norm 기반 importance + recency bias.
    원본 H2O는 누적 attention 사용하지만, key norm이 좋은 근사.
    """
    def _compress(self, key_states, value_states, layer_idx, budget):
        device = key_states.device
        seq_len = key_states.shape[2]
        pk = self._prefill_keys[layer_idx]
        ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states
        score = _key_importance(ref_k).to(device)

        window = min(16, seq_len // 4, budget // 4) if budget > 4 else 0
        window = max(window, 0)
        if window > 0 and seq_len > window:
            recent_idx = torch.arange(seq_len - window, seq_len, device=device)
            prefix_budget = max(budget - window, 1)
            prefix_len = seq_len - window
            _, top_idx = torch.topk(score[:prefix_len], k=min(prefix_budget, prefix_len))
            top_idx, _ = top_idx.sort()
            indices = torch.cat([top_idx, recent_idx])
        else:
            _, indices = torch.topk(score, k=min(budget, seq_len))
            indices, _ = indices.sort()

        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()


# ─────────────────────────────────────────────────────────────
# SnapKV
# ─────────────────────────────────────────────────────────────

class SnapKVCache(BaseHookCache):
    def __init__(self, budget_ratio, num_layers, model_config, window_size=32, kernel_size=5):
        super().__init__(budget_ratio, num_layers, model_config)
        self.window_size = window_size
        self.kernel_size = kernel_size
        self._snap_indices: List[Optional[torch.Tensor]] = [None] * num_layers

    def mark_prefill_done(self, seq_len: int):
        super().mark_prefill_done(seq_len)
        budget = max(int(seq_len * self.budget_ratio), 4)
        for i in range(self.num_layers):
            pk = self._prefill_keys[i]
            if pk is not None:
                score = _key_importance(pk)  # [seq_len]
                self._snap_indices[i] = self._compute_snap_indices(score, seq_len, budget)

    def _compute_snap_indices(self, score, seq_len, budget):
        if self.kernel_size > 1 and score.shape[0] > self.kernel_size:
            score = F.avg_pool1d(
                score.unsqueeze(0).unsqueeze(0),
                kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2,
            ).squeeze()
        device = score.device
        recent_start = max(seq_len - self.window_size, 0)
        select_budget = max(budget - self.window_size, 1)
        if recent_start > 0:
            _, top_idx = torch.topk(score[:recent_start], k=min(select_budget, recent_start))
            top_idx, _ = top_idx.sort()
            recent_idx = torch.arange(recent_start, seq_len, device=device)
            return torch.cat([top_idx, recent_idx])
        return torch.arange(seq_len, device=device)

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        indices = self._snap_indices[layer_idx]
        if indices is not None:
            if seq_len > self._prefill_seq_len:
                new_tokens = torch.arange(self._prefill_seq_len, seq_len, device=indices.device)
                valid = indices[indices < self._prefill_seq_len]
                # new_tokens(가장 최근, 항상 보존) 우선 확보 후 valid에서 나머지 채움
                remaining = max(budget - new_tokens.shape[0], 0)
                valid = valid[-remaining:] if remaining > 0 else valid[:0]
                indices = torch.cat([valid, new_tokens])
            indices = indices[-budget:].to(key_states.device)
            indices, _ = indices.sort()
            return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
        fallback_idx = torch.arange(key_states.shape[2] - budget, key_states.shape[2], device=key_states.device)
        return key_states[:, :, -budget:, :], value_states[:, :, -budget:, :], fallback_idx


# ─────────────────────────────────────────────────────────────
# PyramidKV
# ─────────────────────────────────────────────────────────────

class PyramidKVCache(BaseHookCache):
    """
    레이어별 선형 감소 budget (초기 레이어 많이, 후기 레이어 적게).
    key norm으로 중요 토큰 선택.
    """
    def __init__(self, budget_ratio, num_layers, model_config, min_ratio=0.1):
        super().__init__(budget_ratio, num_layers, model_config)
        self.min_ratio = min_ratio

    def _get_layer_budget(self, layer_idx, seq_len):
        max_ratio = 2 * self.budget_ratio - self.min_ratio
        ratios = torch.linspace(max_ratio, self.min_ratio, self.num_layers)
        return max(int(ratios[layer_idx].item() * seq_len), 4)

    def apply_compression_all_layers(self):
        """레이어별 다른 budget 적용."""
        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            seq_len = k.shape[2]
            budget = self._get_layer_budget(i, seq_len)
            if seq_len > budget:
                k_c, v_c, idx = self._compress(k, v, i, budget)
                k_c = _reindex_rope_keys(k_c, idx, torch.arange(k_c.shape[2]), self._rotary_emb)
                _set_layer_kv(self, i, k_c, v_c)
                if i == 0:
                    self._selected_positions[i] = idx
            else:
                if i == 0:
                    self._selected_positions[i] = torch.arange(k.shape[2])
        self._prefill_keys = [None] * self.num_layers

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        b = min(budget, seq_len)
        pk = self._prefill_keys[layer_idx]

        device = key_states.device
        ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states
        score = _key_importance(ref_k).to(device)
        # recency 보정: 최근 window는 무조건 유지
        w = min(32, seq_len // 4)
        recent_idx = torch.arange(seq_len - w, seq_len, device=device)
        prefix_b = max(b - w, 1)
        if seq_len - w > 0:
            _, top_idx = torch.topk(score[:seq_len - w], k=min(prefix_b, seq_len - w))
            top_idx, _ = top_idx.sort()
            top_idx = top_idx.to(device)
            idx = torch.cat([top_idx, recent_idx])
        else:
            idx = recent_idx
        idx = idx[:b].to(key_states.device)
        return key_states[:, :, idx, :], value_states[:, :, idx, :], idx.cpu()


# ─────────────────────────────────────────────────────────────
# AdaKV (entropy-based layer budget, key norm importance)
# ─────────────────────────────────────────────────────────────

class AdaKVCache(BaseHookCache):
    """
    레이어별 key norm variance 기반 동적 budget 할당.
    높은 variance 레이어 = 정보 다양성 높음 = 더 많은 budget.
    (offline profiling 없이 각 샘플마다 동적 적응)
    """
    def apply_compression_all_layers(self):
        """entropy 기반 softmax 정규화 레이어별 dynamic budget."""
        total_budget = max(int(self._prefill_seq_len * self.budget_ratio), 4) * self.num_layers
        entropies = []
        for i in range(len(self.layers)):
            pk = self._prefill_keys[i]
            ref = pk if pk is not None else _get_layer_kv(self, i)[0].cpu()
            norm = ref.float().norm(dim=-1).mean(dim=1).squeeze(0)
            p = torch.softmax(norm, dim=0)
            ent = -(p * (p + 1e-9).log()).sum().item()
            entropies.append(ent + 1e-9)

        ent_tensor = torch.tensor(entropies)
        # entropy 절대값을 그대로 softmax에 넣으면, 레이어 간 entropy 차이가 작아도
        # softmax의 지수적 증폭 특성 때문에 weight가 극단적으로 쏠려(예: 0.0001 vs 0.04,
        # 400배 차이) 일부 레이어의 budget이 최저치(4)로 깔리는 문제가 있었다.
        # (실측: entropy range 1.23~7.56 -> softmax weight range 0.0001~0.0404)
        # 0-1 정규화 후 softmax하면 weight가 합리적인 범위로 펴진다 (실측: 0.011~0.031).
        ent_min, ent_max = ent_tensor.min(), ent_tensor.max()
        if (ent_max - ent_min) > 1e-6:
            ent_normalized = (ent_tensor - ent_min) / (ent_max - ent_min)
        else:
            ent_normalized = torch.zeros_like(ent_tensor)
        weights = torch.softmax(ent_normalized, dim=0)
        budgets = [max(round(w.item() * total_budget), 4) for w in weights]

        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            seq_len = k.shape[2]
            b = min(budgets[i], seq_len)
            if seq_len > b:
                k_c, v_c, idx = self._compress(k, v, i, b)
                k_c = _reindex_rope_keys(k_c, idx, torch.arange(k_c.shape[2]), self._rotary_emb)
                _set_layer_kv(self, i, k_c, v_c)
                if i == 0:
                    self._selected_positions[i] = idx
            else:
                if i == 0:
                    self._selected_positions[i] = torch.arange(k.shape[2])
        self._prefill_keys = [None] * self.num_layers

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        b = min(budget, seq_len)
        pk = self._prefill_keys[layer_idx]
        score = _key_importance(pk if pk is not None and pk.shape[2] == seq_len
                                else key_states.cpu())
        # recency 보정: 최근 window(질문이 포함된 prompt 끝부분)는 무조건 유지.
        # 순수 importance top-k만 쓰면, generate()의 rollback 방식(마지막 토큰을 캐시에서
        # 빼고 다시 넣는 방식)과 충돌해 prompt 끝부분(질문)이 캐시에서 사라지는 경우가
        # 있었음 (실측: AdaKV 모든 레이어가 input_length-1 position을 누락).
        w = min(32, seq_len // 4)
        recent_idx = torch.arange(seq_len - w, seq_len, device=score.device)
        prefix_b = max(b - w, 1)
        if seq_len - w > 0:
            _, top_idx = torch.topk(score[:seq_len - w], k=min(prefix_b, seq_len - w))
            top_idx, _ = top_idx.sort()
            top_idx = top_idx.to(key_states.device)
            indices = torch.cat([top_idx, recent_idx.to(key_states.device)])
        else:
            indices = recent_idx.to(key_states.device)
        indices = indices[:b]
        indices, _ = indices.sort()
        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()


# ─────────────────────────────────────────────────────────────
# OursHybrid
# ─────────────────────────────────────────────────────────────

class OursHybridCache(BaseHookCache):
    """
    4-signal Hybrid Score + entropy 기반 레이어별 dynamic budget.

    신호 구성:
      A (α=0.40): key norm 기반 누적 importance (attention 근사)
      E (β=0.20): 헤드 간 key norm 분산 (정보 다양성)
      Sem (γ=0.20): prefill 평균 키와의 cosine similarity (의미적 관련성 근사)
      P (δ=0.20): 위치 감쇠 exp(-λ*(L-i)/L)

    레이어 budget:
      key norm variance 기반 동적 할당 (AdaKV와 동일 원리, 더 정교한 score)
    """
    def __init__(self, budget_ratio, num_layers, model_config,
                 alpha=0.40, beta=0.20, gamma=0.20, delta=0.20, lambda_pos=1.0,
                 use_attention=True, use_entropy=True, use_semantic=True, use_position=True):
        super().__init__(budget_ratio, num_layers, model_config)
        total = alpha + beta + gamma + delta
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.delta = delta / total
        self.lambda_pos = lambda_pos
        self.use_attention = use_attention
        self.use_entropy = use_entropy
        self.use_semantic = use_semantic
        self.use_position = use_position

    def apply_compression_all_layers(self):
        """균일 budget으로 모든 레이어 압축."""
        budget = max(int(self._prefill_seq_len * self.budget_ratio), 4)
        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            seq_len = k.shape[2]
            if seq_len > budget:
                k_c, v_c, idx = self._compress(k, v, i, budget)
                k_c = _reindex_rope_keys(k_c, idx, torch.arange(k_c.shape[2]), self._rotary_emb)
                _set_layer_kv(self, i, k_c, v_c)
                if i == 0:
                    self._selected_positions[i] = idx
        for i in range(self.num_layers):
            self._prefill_keys[i] = None
        torch.cuda.empty_cache()

    def _compute_hybrid_score(self, key_states: torch.Tensor, layer_idx: int, seq_len: int) -> torch.Tensor:
        device = key_states.device
        scores = torch.zeros(seq_len, device=device)
        device = key_states.device
        pk = self._prefill_keys[layer_idx]
        ref_k = pk.to(device) if (pk is not None and pk.shape[2] == seq_len) else key_states

        # A: key norm (attention 근사)
        if self.use_attention:
            a_score = _key_importance(ref_k)
            rng = a_score.max() - a_score.min()
            if rng > 1e-9:
                a_score = (a_score - a_score.min()) / rng
            scores = scores + self.alpha * a_score

        # E: 헤드 간 key norm 분산 (정보 다양성)
        if self.use_entropy:
            # [heads, seq_len]
            per_head_norm = ref_k.float().norm(dim=-1).squeeze(0)  # [heads, seq_len]
            head_var = per_head_norm.var(dim=0).to(device)          # [seq_len]
            rng = head_var.max() - head_var.min()
            if rng > 1e-9:
                head_var = (head_var - head_var.min()) / rng
            scores = scores + self.beta * head_var

        # Sem: prefill 평균 키와의 cosine similarity
        # 평균 키 = 전체 문맥의 대표 벡터 (쿼리 근사)
        if self.use_semantic:
            k_cpu = ref_k.float().squeeze(0)  # [heads, seq_len, head_dim]
            mean_k = k_cpu.mean(dim=1, keepdim=True)  # [heads, 1, head_dim] — 평균 키
            sim = F.cosine_similarity(k_cpu, mean_k, dim=-1).mean(dim=0)  # [seq_len]
            del k_cpu, mean_k
            rng = sim.max() - sim.min()
            if rng > 1e-9:
                sim = (sim - sim.min()) / rng
            scores = scores + self.gamma * sim

        # P: 위치 감쇠
        if self.use_position:
            pos = torch.arange(seq_len, device=device).float()
            pos_score = torch.exp(-self.lambda_pos * (seq_len - 1 - pos) / max(seq_len, 1))
            scores = scores + self.delta * pos_score

        return scores

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        scores = self._compute_hybrid_score(key_states, layer_idx, seq_len)
        if scores.sum() == 0:
            indices = torch.arange(seq_len - budget, seq_len, device=key_states.device)
        else:
            _, indices = torch.topk(scores, k=min(budget, seq_len))
            indices, _ = indices.sort()
        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()


# ─────────────────────────────────────────────────────────────
# 팩토리
# ─────────────────────────────────────────────────────────────

METHOD_MAP = {
    "fullkv":       FullKVCache,
    "streaming":    StreamingLLMCache,
    "streamingllm": StreamingLLMCache,
    "h2o":          H2OCache,
    "snapkv":       SnapKVCache,
    "pyramidkv":    PyramidKVCache,
    "adakv":        AdaKVCache,
    "ourshybrid":   OursHybridCache,
    "ours":         OursHybridCache,
}

def make_hook_cache(method_name: str, budget_ratio: float, model_config: Dict, **kwargs):
    key = method_name.lower().replace("-", "").replace("_", "")
    cls = METHOD_MAP.get(key)
    if cls is None:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_MAP.keys())}")
    if cls == FullKVCache:
        return FullKVCache()
    return cls(budget_ratio=budget_ratio, num_layers=model_config["num_layers"],
               model_config=model_config, **kwargs)
