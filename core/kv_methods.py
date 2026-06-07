"""
core/kv_methods.py
===================
모든 KV 캐시 압축 방법 구현.

구현 목록:
1. FullKV         - 압축 없음 (상한 기준선)
2. StreamingLLM   - Attention Sink + 슬라이딩 윈도우
3. H2O            - Heavy Hitter Oracle (누적 어텐션 기반)
4. SnapKV         - Observation Window 기반 prefill 압축
5. PyramidKV      - 정적 피라미드형 레이어별 할당
6. AdaKV          - 오프라인 프로파일링 entropy 기반 적응 할당
7. OursHybrid     - 제안 방법: Hybrid Score + 동적 entropy 예산 할당

모든 메서드는 KVCacheMethod 인터페이스를 따름.
GQA(Grouped Query Attention) 완전 지원.
"""

import math
from core.kv_base import to_legacy_cache
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from core.kv_base import (
    KVCacheMethod,
    select_topk_tokens,
    apply_token_selection,
    get_kv_cache_size_mb,
)


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def compute_attn_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    어텐션 가중치에서 엔트로피 계산.
    
    Args:
        attn_weights: (batch, heads, q_len, kv_len) 또는 (heads, q_len, kv_len)
    
    Returns:
        평균 엔트로피 (스칼라 텐서)
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # batch 차원 제거
    # attn_weights: (heads, q_len, kv_len)
    # 수치 안정성을 위해 clamp
    attn_weights = attn_weights.clamp(min=1e-9)
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1).mean()
    return entropy


def expand_kv_heads(tensor: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    GQA: KV 헤드를 쿼리 헤드 수에 맞게 확장.
    
    Args:
        tensor: (batch, num_kv_heads, seq_len, head_dim)
        num_q_heads: 쿼리 헤드 수
        num_kv_heads: KV 헤드 수
    
    Returns:
        (batch, num_q_heads, seq_len, head_dim)
    """
    if num_q_heads == num_kv_heads:
        return tensor
    
    repeat_factor = num_q_heads // num_kv_heads
    # (batch, num_kv_heads, 1, seq_len, head_dim) -> repeat -> squeeze
    return tensor.repeat_interleave(repeat_factor, dim=1)


def get_accumulated_attn_scores(
    attn_weights_list: List[Optional[torch.Tensor]],
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """
    레이어별 어텐션 가중치에서 토큰별 누적 중요도 점수 추출.
    
    Returns:
        (seq_len,) 텐서 또는 None
    """
    attn = attn_weights_list[layer_idx]
    if attn is None:
        return None
    
    # attn shape 처리: (batch, heads, q_len, kv_len) or (heads, q_len, kv_len)
    if attn.dim() == 4:
        attn = attn[0]  # batch 제거
    # attn: (heads, q_len, kv_len)
    
    # 모든 쿼리 위치, 모든 헤드에 걸쳐 합산
    # scores[i] = 해당 키 토큰이 얼마나 어텐션을 받았는지
    scores = attn.sum(dim=(0, 1))  # (kv_len,)
    
    # 정규화
    if scores.sum() > 0:
        scores = scores / scores.sum()
    
    return scores


# ─────────────────────────────────────────────
# 1. FullKV (압축 없음)
# ─────────────────────────────────────────────

class FullKV(KVCacheMethod):
    """압축 없이 모든 토큰 유지. 정확도 상한 기준선."""
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        return past_key_values
    
    def get_name(self):
        return "FullKV"


# ─────────────────────────────────────────────
# 2. StreamingLLM
# ─────────────────────────────────────────────

class StreamingLLM(KVCacheMethod):
    """
    Attention Sink (앞 k개) + 최근 윈도우 토큰 유지.
    
    References: Xiao et al., "Efficient Streaming LLMs with Attention Sinks" (2023)
    """
    
    def __init__(self, model_config: Dict, sink_size: int = 4):
        super().__init__(model_config)
        self.sink_size = sink_size
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        compressed = []
        
        for layer_kv in past_key_values:
            if layer_kv is None:
                compressed.append(None)
                continue
            
            k, v = layer_kv
            seq_len = k.shape[2]
            budget = max(int(seq_len * budget_ratio), self.sink_size + 1)
            
            if seq_len <= budget:
                compressed.append((k, v))
                continue
            
            # Sink 토큰 + 최근 윈도우 토큰
            window_size = budget - self.sink_size
            sink_k = k[:, :, :self.sink_size, :]
            sink_v = v[:, :, :self.sink_size, :]
            recent_k = k[:, :, -window_size:, :]
            recent_v = v[:, :, -window_size:, :]
            
            k_new = torch.cat([sink_k, recent_k], dim=2)
            v_new = torch.cat([sink_v, recent_v], dim=2)
            compressed.append((k_new, v_new))
        
        return tuple(compressed)
    
    def get_name(self):
        return "StreamingLLM"


# ─────────────────────────────────────────────
# 3. H2O (Heavy Hitter Oracle)
# ─────────────────────────────────────────────

class H2O(KVCacheMethod):
    """
    누적 어텐션 점수 기반 동적 KV 캐시 제거.
    Heavy Hitter 토큰(높은 누적 어텐션) + 최근 토큰 유지.
    
    References: Zhang et al., "H2O: Heavy-Hitter Oracle for LLM KV Cache" (2023)
    """
    
    def __init__(self, model_config: Dict, recent_ratio: float = 0.1):
        super().__init__(model_config)
        self.recent_ratio = recent_ratio  # 최근 토큰 보호 비율
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        layer_indices = []
        
        for layer_idx, layer_kv in enumerate(past_key_values):
            if layer_kv is None:
                layer_indices.append(None)
                continue
            
            k, v = layer_kv
            seq_len = k.shape[2]
            budget = max(int(seq_len * budget_ratio), 4)
            
            if seq_len <= budget:
                layer_indices.append(None)
                continue
            
            # 어텐션 점수 가져오기
            attn_scores = get_accumulated_attn_scores(attn_weights_list, layer_idx)
            
            if attn_scores is None or attn_scores.shape[0] != seq_len:
                # 폴백: 균일 점수 (최근 토큰 약간 가중)
                attn_scores = torch.ones(seq_len)
                recent_len = max(1, int(seq_len * self.recent_ratio))
                attn_scores[-recent_len:] += 0.5
            
            # 최근 토큰 보호: 최근 recent_ratio만큼은 항상 유지
            recent_count = max(1, int(budget * self.recent_ratio))
            hh_count = budget - recent_count
            
            # Heavy Hitter: 최근 토큰 제외하고 상위 hh_count 선택
            if hh_count > 0 and seq_len > recent_count:
                historical_scores = attn_scores[:-recent_count]
                _, hh_indices = torch.topk(
                    historical_scores, k=min(hh_count, len(historical_scores)),
                    largest=True, sorted=False
                )
                hh_indices, _ = torch.sort(hh_indices)
                recent_indices = torch.arange(seq_len - recent_count, seq_len)
                indices = torch.cat([hh_indices, recent_indices])
            else:
                indices = torch.arange(seq_len - budget, seq_len)
            
            layer_indices.append(indices)
        
        return apply_token_selection(past_key_values, layer_indices)
    
    def get_name(self):
        return "H2O"


# ─────────────────────────────────────────────
# 4. SnapKV
# ─────────────────────────────────────────────

class SnapKV(KVCacheMethod):
    """
    Prefill 단계의 관찰 윈도우(마지막 w개 토큰)의 어텐션 패턴으로
    중요 KV 토큰 선택.
    
    References: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation" (2024)
    """
    
    def __init__(self, model_config: Dict, window_size: int = 32, kernel_size: int = 5):
        super().__init__(model_config)
        self.window_size = window_size
        self.kernel_size = kernel_size
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        layer_indices = []
        
        for layer_idx, layer_kv in enumerate(past_key_values):
            if layer_kv is None:
                layer_indices.append(None)
                continue
            
            k, v = layer_kv
            seq_len = k.shape[2]
            budget = max(int(seq_len * budget_ratio), 4)
            
            if seq_len <= budget:
                layer_indices.append(None)
                continue
            
            attn = attn_weights_list[layer_idx]
            
            if attn is not None:
                if attn.dim() == 4:
                    attn = attn[0]  # (heads, q_len, kv_len)
                
                # 관찰 윈도우: 마지막 window_size개 쿼리 토큰의 어텐션 사용
                obs_len = min(self.window_size, attn.shape[1])
                obs_attn = attn[:, -obs_len:, :]  # (heads, obs_len, kv_len)
                
                # 헤드 평균 + 쿼리 평균
                scores = obs_attn.mean(dim=0).mean(dim=0)  # (kv_len,)
                
                # 풀링으로 지역성 보정
                if self.kernel_size > 1 and seq_len >= self.kernel_size:
                    scores_2d = scores.unsqueeze(0).unsqueeze(0)  # (1, 1, kv_len)
                    scores_2d = F.avg_pool1d(
                        scores_2d,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    )
                    scores = scores_2d.squeeze()
                    if scores.shape[0] > seq_len:
                        scores = scores[:seq_len]
            else:
                # 폴백: 균일 + 최근성
                scores = torch.arange(seq_len, dtype=torch.float)
            
            indices = select_topk_tokens(scores, budget)
            layer_indices.append(indices)
        
        return apply_token_selection(past_key_values, layer_indices)
    
    def get_name(self):
        return "SnapKV"


# ─────────────────────────────────────────────
# 5. PyramidKV
# ─────────────────────────────────────────────

class PyramidKV(KVCacheMethod):
    """
    정적 피라미드형 레이어별 예산 할당.
    초기(얕은) 레이어 → 큰 예산, 후기(깊은) 레이어 → 작은 예산.
    
    References: Cai et al., "PyramidKV: Dynamic KV Cache Compression" (2024)
    """
    
    def __init__(self, model_config: Dict, min_ratio: float = 0.1):
        super().__init__(model_config)
        self.min_ratio = min_ratio  # 가장 깊은 레이어의 최소 예산 비율
    
    def _get_layer_budgets(self, seq_len: int, budget_ratio: float) -> List[int]:
        """
        피라미드형 레이어별 예산 계산.
        전체 평균 = budget_ratio × seq_len.
        """
        num_layers = self.num_layers
        total_budget = int(seq_len * budget_ratio)
        
        # 선형 피라미드: layer 0 = max_ratio, layer L-1 = min_ratio
        max_ratio = 2 * budget_ratio - self.min_ratio
        max_ratio = min(max_ratio, 1.0)
        
        budgets = []
        for l in range(num_layers):
            # 얕은 레이어 = 큰 비율, 깊은 레이어 = 작은 비율
            ratio = max_ratio - (max_ratio - self.min_ratio) * (l / max(num_layers - 1, 1))
            layer_budget = max(int(seq_len * ratio), 4)
            budgets.append(layer_budget)
        
        return budgets
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        layer_indices = []
        
        # 시퀀스 길이 결정
        seq_len = None
        for layer_kv in past_key_values:
            if layer_kv is not None:
                seq_len = layer_kv[0].shape[2]
                break
        
        if seq_len is None:
            return past_key_values
        
        layer_budgets = self._get_layer_budgets(seq_len, budget_ratio)
        
        for layer_idx, layer_kv in enumerate(past_key_values):
            if layer_kv is None:
                layer_indices.append(None)
                continue
            
            k, v = layer_kv
            seq_len_l = k.shape[2]
            budget = layer_budgets[layer_idx]
            
            if seq_len_l <= budget:
                layer_indices.append(None)
                continue
            
            # 어텐션 점수 사용 (없으면 최근성 기반)
            attn_scores = get_accumulated_attn_scores(attn_weights_list, layer_idx)
            if attn_scores is None or attn_scores.shape[0] != seq_len_l:
                attn_scores = torch.arange(seq_len_l, dtype=torch.float)
            
            indices = select_topk_tokens(attn_scores, budget)
            layer_indices.append(indices)
        
        return apply_token_selection(past_key_values, layer_indices)
    
    def get_name(self):
        return "PyramidKV"


# ─────────────────────────────────────────────
# 6. AdaKV
# ─────────────────────────────────────────────

class AdaKV(KVCacheMethod):
    """
    오프라인 프로파일링된 entropy 기반 적응형 레이어별 예산 할당.
    가장 강력한 선행 기준선.
    
    구현:
    - 실제 오프라인 프로파일링 대신, 현재 입력의 어텐션 entropy를 사용
      (단, 레이어별 가중치를 누적/평활화하여 "프로파일링 효과" 시뮬레이션)
    - entropy가 높은 레이어 = 더 많은 정보 → 큰 예산
    
    References: Yuan et al., "Ada-KV: Optimizing KV Cache Eviction" (2024)
    """
    
    def __init__(self, model_config: Dict, smoothing: float = 0.1):
        super().__init__(model_config)
        self.smoothing = smoothing
        # 오프라인 프로파일링 가중치 (없으면 현재 입력에서 계산)
        self._profile_weights: Optional[torch.Tensor] = None
    
    def _compute_layer_entropy(
        self,
        attn_weights_list: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """레이어별 평균 attention entropy 계산."""
        entropies = []
        for layer_idx in range(self.num_layers):
            attn = attn_weights_list[layer_idx]
            if attn is not None:
                entropy = compute_attn_entropy(attn).item()
            else:
                entropy = 1.0  # 폴백: 균일
            entropies.append(entropy)
        return torch.tensor(entropies, dtype=torch.float)
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        # 레이어별 entropy 계산
        entropies = self._compute_layer_entropy(attn_weights_list)
        

        # entropies NaN 처리
        entropies = torch.nan_to_num(entropies, nan=1.0/self.num_layers)
        # 프로파일 가중치와 평활화 혼합
        if self._profile_weights is not None:
            weights = (1 - self.smoothing) * self._profile_weights + self.smoothing * entropies
        else:
            weights = entropies
        
        # 가중치 정규화 + NaN 처리
        weights = weights / (weights.sum() + 1e-9)
        # NaN/Inf 안전 처리: 균일 분포로 대체
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones(self.num_layers) / self.num_layers
        
        # 시퀀스 길이
        seq_len = None
        for layer_kv in past_key_values:
            if layer_kv is not None:
                seq_len = layer_kv[0].shape[2]
                break
        
        if seq_len is None:
            return past_key_values
        
        total_budget = int(seq_len * budget_ratio)
        layer_budgets = [
            max(int(weights[l].item() * total_budget * self.num_layers), 4)
            for l in range(self.num_layers)
        ]
        
        layer_indices = []
        for layer_idx, layer_kv in enumerate(past_key_values):
            if layer_kv is None:
                layer_indices.append(None)
                continue
            
            k, v = layer_kv
            seq_len_l = k.shape[2]
            budget = min(layer_budgets[layer_idx], seq_len_l)
            
            if seq_len_l <= budget:
                layer_indices.append(None)
                continue
            
            attn_scores = get_accumulated_attn_scores(attn_weights_list, layer_idx)
            if attn_scores is None or attn_scores.shape[0] != seq_len_l:
                attn_scores = torch.arange(seq_len_l, dtype=torch.float)
            
            indices = select_topk_tokens(attn_scores, budget)
            layer_indices.append(indices)
        
        return apply_token_selection(past_key_values, layer_indices)
    
    def get_name(self):
        return "AdaKV"


# ─────────────────────────────────────────────
# 7. OursHybrid (제안 방법)
# ─────────────────────────────────────────────

class OursHybrid(KVCacheMethod):
    """
    제안 방법: Adaptive Layer-wise Hybrid KV Cache Management.
    
    Layer 2: 동적 entropy 기반 레이어별 예산 할당
      Bl = round(rl × Btotal), rl = Hl / ΣHl'
    
    Layer 1: Hybrid Importance Score
      S(ti) = α·A(ti) + β·E(ti) + γ·Sem(ti) + δ·P(ti)
      기본값: α=0.40, β=0.20, γ=0.20, δ=0.20
    """
    
    def __init__(
        self,
        model_config: Dict,
        alpha: float = 0.40,   # 누적 어텐션 가중치
        beta: float = 0.20,    # 엔트로피 기반 분산 가중치
        gamma: float = 0.20,   # 의미적 유사성 가중치
        delta: float = 0.20,   # 위치 감쇠 가중치
        lambda_pos: float = 1.0,  # 위치 감쇠율
        # ablation 제어
        use_attention: bool = True,
        use_entropy: bool = True,
        use_semantic: bool = True,
        use_position: bool = True,
        # 레이어 할당 전략
        allocation_strategy: str = "entropy",  # "entropy" | "uniform" | "pyramid" | "random"
    ):
        super().__init__(model_config)
        
        # 가중치 정규화
        total = alpha + beta + gamma + delta
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.delta = delta / total
        self.lambda_pos = lambda_pos
        
        # Ablation 플래그
        self.use_attention = use_attention
        self.use_entropy = use_entropy
        self.use_semantic = use_semantic
        self.use_position = use_position
        
        # 레이어 할당 전략
        self.allocation_strategy = allocation_strategy
    
    # ── Layer 2: 동적 레이어별 예산 할당 ──────────
    
    def _compute_dynamic_budgets(
        self,
        attn_weights_list: List[Optional[torch.Tensor]],
        total_budget: int,
        seq_len: int,
    ) -> List[int]:
        """
        실시간 attention entropy 기반 레이어별 예산 계산.
        오프라인 프로파일링 불필요.
        """
        if self.allocation_strategy == "uniform":
            base = total_budget // self.num_layers
            budgets = [base] * self.num_layers
            remainder = total_budget - base * self.num_layers
            for i in range(remainder):
                budgets[i] += 1
            return budgets
        
        elif self.allocation_strategy == "pyramid":
            # 정적 피라미드 (얕은 레이어 → 큰 예산)
            weights = torch.linspace(1.0, 0.1, self.num_layers)
            weights = weights / weights.sum()
            return [max(int(w.item() * total_budget), 4) for w in weights]
        
        elif self.allocation_strategy == "random":
            import random
            weights = [random.random() for _ in range(self.num_layers)]
            total_w = sum(weights)
            return [max(int(w / total_w * total_budget), 4) for w in weights]
        
        else:  # "entropy" (기본값, 제안 방법)
            entropies = []
            for layer_idx in range(self.num_layers):
                attn = attn_weights_list[layer_idx]
                if attn is not None:
                    h = compute_attn_entropy(attn).item()
                    h = max(h, 1e-6)  # 수치 안정성
                else:
                    h = 1.0
                entropies.append(h)
            
            entropy_tensor = torch.tensor(entropies, dtype=torch.float)
            # NaN/Inf 방어
            entropy_tensor = torch.nan_to_num(entropy_tensor, nan=1.0, posinf=1.0, neginf=0.0)
            total = entropy_tensor.sum()
            if total < 1e-9:
                entropy_tensor = torch.ones_like(entropy_tensor)
                total = entropy_tensor.sum()
            weights = entropy_tensor / total
            
            budgets = []
            for l in range(self.num_layers):
                w = weights[l].item()
                if w != w:  # NaN 체크
                    w = 1.0 / self.num_layers
                b = max(round(w * total_budget * self.num_layers), 4)
                b = min(b, seq_len)
                budgets.append(b)
            
            return budgets
    
    # ── Layer 1: Hybrid Importance Score ──────────
    
    def _compute_hybrid_score(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attn: Optional[torch.Tensor],
        layer_idx: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        4개 신호 합산으로 토큰 중요도 점수 계산.
        
        Args:
            k: (batch, num_kv_heads, seq_len, head_dim)
            v: (batch, num_kv_heads, seq_len, head_dim)
            attn: 어텐션 가중치 (None 가능)
            layer_idx: 레이어 인덱스
            seq_len: 시퀀스 길이
        
        Returns:
            (seq_len,) 중요도 점수
        """
        device = k.device
        scores = torch.zeros(seq_len, device=device)
        
        # ── 신호 1: 누적 어텐션 (A) ──
        if self.use_attention and attn is not None:
            if attn.dim() == 4:
                attn_2d = attn[0]
            else:
                attn_2d = attn
            # (heads, q_len, kv_len) → 모든 위치에서 받은 누적 어텐션
            attn_2d = attn_2d.to(device)
            A = attn_2d.sum(dim=(0, 1))  # (kv_len,)
            if A.shape[0] == seq_len:
                if A.sum() > 0:
                    A = A / A.sum()
                scores += self.alpha * A
        
        # ── 신호 2: 엔트로피 기반 분산 (E) ──
        if self.use_entropy and attn is not None:
            if attn.dim() == 4:
                attn_2d = attn[0]
            else:
                attn_2d = attn
            attn_2d = attn_2d.to(device)
            
            # 각 토큰(kv)에 대한 헤드 간 분산 측정
            # attn_2d: (heads, q_len, kv_len)
            # 각 kv 토큰에 대해, 헤드들이 얼마나 일치해서 어텐션하는지 (일치 = 정보 집중)
            token_attn = attn_2d.mean(dim=1)  # (heads, kv_len)
            
            # 헤드 간 엔트로피 (낮으면 집중, 높으면 분산)
            token_attn_norm = token_attn / (token_attn.sum(dim=0, keepdim=True) + 1e-9)
            h = -(token_attn_norm * (token_attn_norm + 1e-9).log()).sum(dim=0)  # (kv_len,)
            
            if h.shape[0] == seq_len:
                n_heads = attn_2d.shape[0]
                max_entropy = math.log(n_heads) if n_heads > 1 else 1.0
                # 분산(entropy)이 높을수록 중요 (다양한 헤드가 주목)
                E = h / (max_entropy + 1e-9)
                if E.sum() > 0:
                    E = E / E.sum()
                scores += self.beta * E
        
        # ── 신호 3: 의미적 유사성 (Sem) ──
        if self.use_semantic:
            # 마지막 쿼리 표현 (k의 마지막 토큰)과 나머지 토큰들의 cosine 유사성
            # k: (batch, num_kv_heads, seq_len, head_dim)
            k_float = k[0].float()  # (num_kv_heads, seq_len, head_dim)
            
            # 쿼리 표현: 마지막 토큰 (현재 생성 위치)
            query_rep = k_float[:, -1, :]  # (num_kv_heads, head_dim)
            key_reps = k_float[:, :-1, :]  # (num_kv_heads, seq_len-1, head_dim)
            
            if key_reps.shape[1] > 0:
                # cosine 유사성: (num_kv_heads, seq_len-1)
                query_norm = F.normalize(query_rep, dim=-1)  # (heads, dim)
                key_norm = F.normalize(key_reps, dim=-1)      # (heads, seq-1, dim)
                sim = torch.einsum("hd,hsd->hs", query_norm, key_norm)  # (heads, seq-1)
                
                # 헤드 간 최대값 (어느 헤드라도 유사하면 중요)
                sim_max = sim.max(dim=0).values  # (seq_len-1,)
                sim_max = (sim_max + 1) / 2  # [-1,1] → [0,1]
                
                # 마지막 토큰(자기 자신)은 1.0
                Sem = torch.cat([sim_max, torch.ones(1, device=device)])
                if Sem.sum() > 0:
                    Sem = Sem / Sem.sum()
                scores += self.gamma * Sem
        
        # ── 신호 4: 위치 감쇠 (P) ──
        if self.use_position:
            positions = torch.arange(seq_len, dtype=torch.float, device=device)
            # P(ti) = exp(-λ · (L-i)/L)
            # 최근 토큰(i → L) = 높은 점수
            P = torch.exp(-self.lambda_pos * (seq_len - 1 - positions) / seq_len)
            if P.sum() > 0:
                P = P / P.sum()
            scores += self.delta * P
        
        return scores
    
    # ── 통합 compress ──────────────────────────────
    
    def compress(self, past_key_values, attn_weights_list, budget_ratio, **kwargs):
        past_key_values = to_legacy_cache(past_key_values)
        # 시퀀스 길이
        seq_len = None
        for layer_kv in past_key_values:
            if layer_kv is not None:
                seq_len = layer_kv[0].shape[2]
                break
        
        if seq_len is None:
            return past_key_values
        
        total_budget = max(int(seq_len * budget_ratio), 4)
        
        # Layer 2: 동적 레이어별 예산 할당
        layer_budgets = self._compute_dynamic_budgets(
            attn_weights_list, total_budget, seq_len
        )
        
        # Layer 1: 레이어별 Hybrid Score 계산 및 토큰 선택
        layer_indices = []
        
        for layer_idx, layer_kv in enumerate(past_key_values):
            if layer_kv is None:
                layer_indices.append(None)
                continue
            
            k, v = layer_kv
            seq_len_l = k.shape[2]
            budget = min(layer_budgets[layer_idx], seq_len_l)
            
            if seq_len_l <= budget:
                layer_indices.append(None)
                continue
            
            attn = attn_weights_list[layer_idx]
            
            # Hybrid Score 계산
            scores = self._compute_hybrid_score(k, v, attn, layer_idx, seq_len_l)
            
            # 점수가 모두 0인 경우 폴백
            if scores.sum() == 0:
                scores = torch.arange(seq_len_l, dtype=torch.float, device=k.device)
            
            indices = select_topk_tokens(scores.cpu(), budget)
            layer_indices.append(indices)
        
        return apply_token_selection(past_key_values, layer_indices)
    
    def get_name(self):
        return "OursHybrid"


# ─────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────

def create_kv_method(method_name: str, model_config: Dict, **kwargs) -> KVCacheMethod:
    """
    메서드 이름으로 KV 캐시 압축 객체 생성.
    
    Args:
        method_name: "fullkv" | "streaming" | "h2o" | "snapkv" | "pyramidkv" | "adakv" | "ours"
        model_config: 모델 설정 딕셔너리
        **kwargs: 각 메서드별 추가 파라미터
    
    Returns:
        KVCacheMethod 인스턴스
    """
    name_lower = method_name.lower()
    
    if name_lower in ("fullkv", "full_kv", "full"):
        return FullKV(model_config)
    elif name_lower in ("streaming", "streaminglLM", "streamingllm"):
        return StreamingLLM(model_config, **kwargs)
    elif name_lower == "h2o":
        return H2O(model_config, **kwargs)
    elif name_lower == "snapkv":
        return SnapKV(model_config, **kwargs)
    elif name_lower in ("pyramidkv", "pyramid"):
        return PyramidKV(model_config, **kwargs)
    elif name_lower in ("adakv", "ada_kv", "ada"):
        return AdaKV(model_config, **kwargs)
    elif name_lower in ("ours", "hybrid", "ourshybrid"):
        return OursHybrid(model_config, **kwargs)
    else:
        raise ValueError(f"Unknown KV method: {method_name}")
