"""
core/kv_cache_hook.py (v2)
===========================
Transformers 5.10.2의 새 DynamicCache 구조에 맞춘 Hook Cache 구현.

변경사항:
  - key_cache/value_cache → layers[i].keys / layers[i].values
  - update() 반환값이 누적된 전체 KV
  - DynamicCache 상속 후 update() 오버라이드 방식 유지
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from typing import Optional, Dict, List, Tuple, Any


# ─────────────────────────────────────────────────────────────
# 유틸: layers 접근 헬퍼
# ─────────────────────────────────────────────────────────────

def _get_layer_kv(cache: DynamicCache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return cache.layers[layer_idx].keys, cache.layers[layer_idx].values

def _set_layer_kv(cache: DynamicCache, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
    cache.layers[layer_idx].keys = k
    cache.layers[layer_idx].values = v


# ─────────────────────────────────────────────────────────────
# Base Hook Cache
# ─────────────────────────────────────────────────────────────

class BaseHookCache(DynamicCache):
    """
    DynamicCache를 상속하여 update() 시 실시간 KV 압축 적용.
    Transformers 5.10.2: layers[i].keys / layers[i].values 구조 사용.
    """

    def __init__(self, budget_ratio: float, num_layers: int, model_config: Dict):
        super().__init__()
        self.budget_ratio = budget_ratio
        self.num_layers = num_layers
        self.model_config = model_config
        self._prefill_attn: List[Optional[torch.Tensor]] = [None] * num_layers
        self._prefill_done = False
        self._prefill_seq_len = 0

    def set_prefill_attn(self, layer_idx: int, attn: Optional[torch.Tensor]):
        if attn is not None and layer_idx < self.num_layers:
            if attn.dim() == 4:
                self._prefill_attn[layer_idx] = attn.mean(dim=2).detach().cpu()
            else:
                self._prefill_attn[layer_idx] = attn.detach().cpu()

    def mark_prefill_done(self, seq_len: int):
        self._prefill_done = True
        self._prefill_seq_len = seq_len

    def apply_compression_all_layers(self):
        """prefill 완료 후 모든 레이어에 압축 적용."""
        budget = max(int(self._prefill_seq_len * self.budget_ratio), 4)
        for i in range(len(self.layers)):
            k, v = _get_layer_kv(self, i)
            if k.shape[2] > budget:
                k_c, v_c = self._compress(k, v, i, budget)
                _set_layer_kv(self, i, k_c, v_c)

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
        recent_size = max(budget - self.sink_size, 1)
        sink_k = key_states[:, :, :self.sink_size, :]
        sink_v = value_states[:, :, :self.sink_size, :]
        recent_k = key_states[:, :, -recent_size:, :]
        recent_v = value_states[:, :, -recent_size:, :]
        return torch.cat([sink_k, recent_k], dim=2), torch.cat([sink_v, recent_v], dim=2)


# ─────────────────────────────────────────────────────────────
# H2O
# ─────────────────────────────────────────────────────────────

class H2OCache(BaseHookCache):
    def __init__(self, budget_ratio, num_layers, model_config):
        super().__init__(budget_ratio, num_layers, model_config)
        self._accum_scores: List[Optional[torch.Tensor]] = [None] * num_layers

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        scores = self._accum_scores[layer_idx]
        prefill_attn = self._prefill_attn[layer_idx]

        if scores is not None and scores.shape[-1] >= seq_len:
            s = scores[..., :seq_len].mean(dim=0)
        elif prefill_attn is not None and prefill_attn.shape[-1] >= seq_len:
            s = prefill_attn.mean(dim=1).squeeze(0)[:seq_len]
        else:
            return key_states[:, :, -budget:, :], value_states[:, :, -budget:, :]

        if prefill_attn is not None and prefill_attn.shape[-1] == seq_len:
            pa = prefill_attn.mean(dim=1).squeeze(0)
            s = s.to(pa.device) + pa

        _, indices = torch.topk(s.cpu(), k=min(budget, seq_len))
        indices, _ = indices.sort()
        indices = indices.to(key_states.device)
        return key_states[:, :, indices, :], value_states[:, :, indices, :]


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
            attn = self._prefill_attn[i]
            if attn is not None:
                self._snap_indices[i] = self._compute_snap_indices(attn, seq_len, budget)

    def _compute_snap_indices(self, attn, seq_len, budget):
        score = attn.mean(dim=1).squeeze(0)
        if self.kernel_size > 1 and score.shape[-1] > self.kernel_size:
            score = F.avg_pool1d(
                score.unsqueeze(0).unsqueeze(0),
                kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2,
            ).squeeze()
        recent_start = max(seq_len - self.window_size, 0)
        select_budget = max(budget - self.window_size, 1)
        if recent_start > 0:
            _, top_idx = torch.topk(score[:recent_start].cpu(), k=min(select_budget, recent_start))
            top_idx, _ = top_idx.sort()
            recent_idx = torch.arange(recent_start, seq_len)
            return torch.cat([top_idx, recent_idx])
        return torch.arange(seq_len)

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        indices = self._snap_indices[layer_idx]
        if indices is not None:
            if seq_len > self._prefill_seq_len:
                new_tokens = torch.arange(self._prefill_seq_len, seq_len)
                valid = indices[indices < self._prefill_seq_len]
                indices = torch.cat([valid, new_tokens])
            indices = indices[:budget].to(key_states.device)
            return key_states[:, :, indices, :], value_states[:, :, indices, :]
        return key_states[:, :, -budget:, :], value_states[:, :, -budget:, :]


# ─────────────────────────────────────────────────────────────
# PyramidKV
# ─────────────────────────────────────────────────────────────

class PyramidKVCache(BaseHookCache):
    def __init__(self, budget_ratio, num_layers, model_config, min_ratio=0.1):
        super().__init__(budget_ratio, num_layers, model_config)
        self.min_ratio = min_ratio

    def _get_layer_budget(self, layer_idx, seq_len):
        max_ratio = 2 * self.budget_ratio - self.min_ratio
        ratios = torch.linspace(max_ratio, self.min_ratio, self.num_layers)
        return max(int(ratios[layer_idx].item() * seq_len), 4)

    def _compress(self, key_states, value_states, layer_idx, budget):
        seq_len = key_states.shape[2]
        b = min(budget, seq_len)
        pa = self._prefill_attn[layer_idx]
        if pa is not None and pa.shape[-1] >= seq_len:
            s = pa.mean(dim=1).squeeze(0)[:seq_len]
            w = min(32, seq_len // 4)
            recent_idx = torch.arange(seq_len - w, seq_len)
            prefix_b = max(b - w, 1)
            if seq_len - w > 0:
                _, top_idx = torch.topk(s[:seq_len-w].cpu(), k=min(prefix_b, seq_len-w))
                top_idx, _ = top_idx.sort()
                idx = torch.cat([top_idx, recent_idx])
            else:
                idx = recent_idx
            idx = idx[:b].to(key_states.device)
            return key_states[:, :, idx, :], value_states[:, :, idx, :]
        return key_states[:, :, -b:, :], value_states[:, :, -b:, :]


# ─────────────────────────────────────────────────────────────
# AdaKV (entropy-based, no offline profiling)
# ─────────────────────────────────────────────────────────────

class AdaKVCache(BaseHookCache):
    """AdaKV: prefill entropy 기반 레이어별 budget (균일로 fallback)."""
    def __init__(self, budget_ratio, num_layers, model_config, smoothing=0.1):
        super().__init__(budget_ratio, num_layers, model_config)
        self.smoothing = smoothing

    def _compress(self, key_states, value_states, layer_idx, budget):
        # prefill attn entropy로 중요 토큰 선택
        attn = self._prefill_attn[layer_idx]
        seq_len = key_states.shape[2]
        if attn is not None and attn.shape[-1] >= seq_len:
            score = attn.mean(dim=1).squeeze(0)[:seq_len]
            _, indices = torch.topk(score.cpu(), k=min(budget, seq_len))
            indices, _ = indices.sort()
            indices = indices.to(key_states.device)
            return key_states[:, :, indices, :], value_states[:, :, indices, :]
        return key_states[:, :, -budget:, :], value_states[:, :, -budget:, :]


# ─────────────────────────────────────────────────────────────
# OursHybrid
# ─────────────────────────────────────────────────────────────

class OursHybridCache(BaseHookCache):
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
        self._accum_attn: List[Optional[torch.Tensor]] = [None] * num_layers

    def _compute_hybrid_score(self, key_states, layer_idx, seq_len):
        device = key_states.device
        scores = torch.zeros(seq_len, device=device)

        # A: 누적 attention
        if self.use_attention:
            src = self._accum_attn[layer_idx]
            pa = self._prefill_attn[layer_idx]
            a_score = torch.zeros(seq_len, device=device)
            if src is not None and src.shape[-1] >= seq_len:
                a_score = src[..., :seq_len].mean(dim=0).to(device)
            if pa is not None and pa.shape[-1] == seq_len:
                a_score = a_score + pa.mean(dim=1).squeeze(0).to(device)
            rng = a_score.max() - a_score.min()
            if rng > 1e-9:
                a_score = (a_score - a_score.min()) / rng
            scores = scores + self.alpha * a_score

        # E: 헤드간 분산
        if self.use_entropy and self._prefill_attn[layer_idx] is not None:
            attn = self._prefill_attn[layer_idx].to(device)
            if attn.shape[-1] == seq_len:
                head_var = attn.var(dim=1).squeeze(0)
                rng = head_var.max() - head_var.min()
                if rng > 1e-9:
                    head_var = (head_var - head_var.min()) / rng
                scores = scores + self.beta * head_var

        # Sem: cosine similarity with last key
        if self.use_semantic:
            last_k = key_states[:, :, -1:, :]
            all_k = key_states[:, :, :seq_len, :]
            sim = F.cosine_similarity(all_k, last_k, dim=-1).mean(dim=1).squeeze(0)
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
        return key_states[:, :, indices, :], value_states[:, :, indices, :]


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
