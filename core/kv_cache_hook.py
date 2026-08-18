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


def _select_with_sink(score: torch.Tensor, seq_len: int, budget: int,
                       recent_w: int, sink_size: int, device) -> torch.Tensor:
    """
    budget 내에서 sink_size(시퀀스 최초 위치)와 recent_w(최근 위치)를 우선 보존하고,
    나머지 예산은 score 기반 top-k로 선택한다.
    StreamingLLMCache의 sink 회계 방식(budget 안에서 차감)과 동일한 관례를 따른다 —
    즉 sink4를 추가해도 전체 KV 캐시 크기(budget)는 baseline과 동일하게 유지된다.
    """
    sink_size = max(min(sink_size, budget, seq_len), 0)
    recent_w = max(min(recent_w, max(budget - sink_size, 0), seq_len), 0)
    mid_budget = max(budget - sink_size - recent_w, 0)

    sink_idx = torch.arange(sink_size, device=device)
    if recent_w > 0:
        recent_idx = torch.arange(seq_len - recent_w, seq_len, device=device)
    else:
        recent_idx = torch.empty(0, dtype=torch.long, device=device)

    mid_start, mid_end = sink_size, seq_len - recent_w
    if mid_budget > 0 and mid_end > mid_start:
        mid_score = score[mid_start:mid_end]
        k = min(mid_budget, mid_end - mid_start)
        _, top_idx = torch.topk(mid_score, k=k)
        top_idx = top_idx + mid_start
        top_idx, _ = top_idx.sort()
    else:
        top_idx = torch.empty(0, dtype=torch.long, device=device)

    indices = torch.cat([sink_idx, top_idx, recent_idx])
    indices = torch.unique(indices)
    indices, _ = indices.sort()
    return indices


# ─────────────────────────────────────────────────────────────
# Base Hook Cache
# ─────────────────────────────────────────────────────────────

class BaseHookCache(DynamicCache):
    """
    DynamicCache 상속. prefill KV 복사 완료 후 apply_compression_all_layers()로
    일괄 압축. Transformers 5.10.2: layers[i].keys / layers[i].values 구조.
    """

    def __init__(self, budget_ratio: float, num_layers: int, model_config: Dict, sink_size: int = 0,
                 invert_norm: bool = False):
        super().__init__()
        self.budget_ratio = budget_ratio
        self.num_layers = num_layers
        self.model_config = model_config
        self.sink_size = sink_size  # 0이면 기존 baseline과 완전히 동일한 동작
        self.invert_norm = invert_norm  # True면 low-key-norm 토큰 우선 (Devoto et al. 방향)
        # prefill key tensor 저장 (importance 계산용)
        self._prefill_keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self._prefill_done = False
        self._prefill_seq_len = 0
        self._selected_positions: List[Optional[torch.Tensor]] = [None] * num_layers

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
        if self.invert_norm:
            score = -score

        window = min(16, seq_len // 4, budget // 4) if budget > 4 else 0
        window = max(window, 0)
        indices = _select_with_sink(score, seq_len, budget, window, self.sink_size, device)

        return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()


# ─────────────────────────────────────────────────────────────
# SnapKV
# ─────────────────────────────────────────────────────────────

class SnapKVCache(BaseHookCache):
    def __init__(self, budget_ratio, num_layers, model_config, window_size=32, kernel_size=5, sink_size=0,
                 invert_norm=False):
        super().__init__(budget_ratio, num_layers, model_config, sink_size=sink_size,
                         invert_norm=invert_norm)
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
                if self.invert_norm:
                    score = -score
                self._snap_indices[i] = self._compute_snap_indices(score, seq_len, budget)

    def _compute_snap_indices(self, score, seq_len, budget):
        if self.kernel_size > 1 and score.shape[0] > self.kernel_size:
            score = F.avg_pool1d(
                score.unsqueeze(0).unsqueeze(0),
                kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2,
            ).squeeze()
        device = score.device
        return _select_with_sink(score, seq_len, budget, self.window_size, self.sink_size, device)

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        indices = self._snap_indices[layer_idx]
        if indices is not None:
            if seq_len > self._prefill_seq_len:
                new_tokens = torch.arange(self._prefill_seq_len, seq_len, device=indices.device)
                valid = indices[indices < self._prefill_seq_len]
                indices = torch.cat([valid, new_tokens])
            indices = indices[:budget].to(key_states.device)
            return key_states[:, :, indices, :], value_states[:, :, indices, :], indices.cpu()
        fallback_idx = torch.arange(key_states.shape[2] - budget, key_states.shape[2])
        return key_states[:, :, -budget:, :], value_states[:, :, -budget:, :], fallback_idx


# ─────────────────────────────────────────────────────────────
# PyramidKV
# ─────────────────────────────────────────────────────────────

class PyramidKVCache(BaseHookCache):
    """
    레이어별 선형 감소 budget (초기 레이어 많이, 후기 레이어 적게).
    key norm으로 중요 토큰 선택.
    """
    def __init__(self, budget_ratio, num_layers, model_config, min_ratio=0.1, sink_size=0,
                 invert_norm=False):
        super().__init__(budget_ratio, num_layers, model_config, sink_size=sink_size,
                         invert_norm=invert_norm)
        self.min_ratio = min_ratio

    def _get_layer_budget(self, layer_idx, seq_len):
        max_ratio = 2 * self.budget_ratio - self.min_ratio
        ratios = torch.linspace(max_ratio, self.min_ratio, self.num_layers)
        return max(int(ratios[layer_idx].item() * seq_len), 4)

    def apply_compression_all_layers(self):
        """레이어별 다른 budget 적용. 모든 레이어 동일 길이(min_budget) 보장."""
        budgets = []
        for i in range(len(self.layers)):
            k, _ = _get_layer_kv(self, i)
            budgets.append(self._get_layer_budget(i, k.shape[2]))
        avg_budget = max(round(sum(budgets) / len(budgets)), 4)

        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            seq_len = k.shape[2]
            b = min(avg_budget, seq_len)
            if seq_len > b:
                k_c, v_c, idx = self._compress(k, v, i, b)
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
        if self.invert_norm:
            score = -score
        w = min(32, seq_len // 4)
        idx = _select_with_sink(score, seq_len, b, w, self.sink_size, device)

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
        ent_min, ent_max = ent_tensor.min(), ent_tensor.max()
        if (ent_max - ent_min) > 1e-9:
            ent_normalized = (ent_tensor - ent_min) / (ent_max - ent_min)
        else:
            ent_normalized = torch.ones_like(ent_tensor)
        weights = torch.softmax(ent_normalized, dim=0)
        budgets = [max(round(w.item() * total_budget), 4) for w in weights]

        # 모든 레이어가 동일한 길이를 가져야 generate()가 올바르게 동작
        # avg_budget으로 통일 (min_budget은 너무 작아서 budget 낭비 발생)
        avg_budget = max(round(sum(budgets) / len(budgets)), 4)
        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            seq_len = k.shape[2]
            b = min(avg_budget, seq_len)
            if seq_len > b:
                k_c, v_c, idx = self._compress(k, v, i, b)
                _set_layer_kv(self, i, k_c, v_c)
                if i == 0:
                    self._selected_positions[i] = idx
            else:
                if i == 0:
                    self._selected_positions[i] = torch.arange(k.shape[2])
        self._prefill_keys = [None] * self.num_layers

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        device = key_states.device
        pk = self._prefill_keys[layer_idx]
        score = _key_importance(pk.to(device) if pk is not None and pk.shape[2] == seq_len
                                else key_states)
        if self.invert_norm:
            score = -score
        w = min(32, seq_len // 4, budget // 2)
        w = max(w, 0)
        indices = _select_with_sink(score, seq_len, budget, w, self.sink_size, device)

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
                 use_attention=True, use_entropy=True, use_semantic=False, use_position=True,
                 sink_size=0, invert_norm=False):
        super().__init__(budget_ratio, num_layers, model_config, sink_size=sink_size,
                         invert_norm=invert_norm)
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
            if self.invert_norm:
                a_score = -a_score
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
        device = key_states.device
        scores = self._compute_hybrid_score(key_states, layer_idx, seq_len)
        w = min(32, seq_len // 4, budget // 2)
        w = max(w, 0)
        indices = _select_with_sink(scores, seq_len, budget, w, self.sink_size, device)

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
