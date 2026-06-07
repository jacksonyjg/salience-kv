"""
core/kv_base.py
================
KV 캐시 압축 메서드의 기반 클래스 및 어텐션 후크 인프라.

모든 KV 압축 방법의 공통 인터페이스:
- prefill 후 KV 캐시 접근
- 레이어별 캐시 압축
- generate()와의 통합

Transformers 4.38.0 기준 past_key_values 구조:
  tuple of (key_tensor, value_tensor) per layer
  key shape: (batch, num_kv_heads, seq_len, head_dim)
"""

import time
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class KVCacheMethod(ABC):
    """KV 캐시 압축 방법의 추상 기반 클래스."""
    
    def __init__(self, model_config: Dict):
        """
        Args:
            model_config: load_model_and_tokenizer에서 반환된 config 딕셔너리
        """
        self.cfg = model_config
        self.num_layers = model_config["num_layers"]
        self.num_heads = model_config["num_heads"]
        self.num_kv_heads = model_config["num_kv_heads"]
        self.head_dim = model_config.get("head_dim", 128)
        
        # 어텐션 가중치 저장 (후크용)
        self._attn_weights: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._hooks = []
    
    @abstractmethod
    def compress(
        self,
        past_key_values: Tuple,
        attn_weights_list: List[Optional[torch.Tensor]],
        budget_ratio: float,
        **kwargs,
    ) -> Tuple:
        """
        KV 캐시 압축 수행.
        
        Args:
            past_key_values: prefill 후 KV 캐시 tuple
            attn_weights_list: 레이어별 어텐션 가중치 리스트
            budget_ratio: 유지할 토큰 비율 (0~1)
        
        Returns:
            압축된 past_key_values (동일 구조)
        """
        pass
    
    def get_name(self) -> str:
        return self.__class__.__name__


def register_attention_hooks(model, num_layers: int) -> Tuple[List, List]:
    """
    모델의 각 어텐션 레이어에 출력 후크를 등록하여 어텐션 가중치 캡처.
    
    Transformers 4.38.0에서 output_attentions=True 사용 시
    메모리 오버헤드가 크므로 후크 방식 사용.
    
    Returns:
        (hooks, attn_weights_list)
    """
    attn_weights_list = [None] * num_layers
    hooks = []
    
    # 모델 아키텍처별 어텐션 모듈 찾기
    attn_modules = _find_attention_modules(model)
    
    for layer_idx, attn_module in enumerate(attn_modules):
        if layer_idx >= num_layers:
            break
        
        def make_hook(idx):
            def hook_fn(module, input, output):
                # output은 (hidden_states,) 또는 (hidden_states, attn_weights, ...)
                # output_attentions=True일 때 attn_weights가 포함됨
                if isinstance(output, tuple) and len(output) > 1:
                    attn_w = output[1]
                    if attn_w is not None and isinstance(attn_w, torch.Tensor):
                        attn_weights_list[idx] = attn_w.detach().cpu()
            return hook_fn
        
        hook = attn_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    return hooks, attn_weights_list


def remove_hooks(hooks: List):
    """등록된 후크 모두 제거."""
    for hook in hooks:
        hook.remove()


def _find_attention_modules(model) -> List:
    """모델에서 어텐션 모듈 리스트 추출."""
    attn_modules = []
    
    # Qwen3 / Qwen2 구조
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                attn_modules.append(layer.self_attn)
    # Phi-3 구조
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                attn_modules.append(layer.self_attn)
    # Gemma-2 구조
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                attn_modules.append(layer.self_attn)
    
    if not attn_modules:
        logger.warning("Could not find attention modules via standard path")
    
    return attn_modules


def get_kv_cache_size_mb(past_key_values, dtype: torch.dtype = torch.float16) -> float:
    """KV 캐시 전체 메모리 크기 계산 (MB)."""
    total_elements = 0
    # DynamicCache 호환 처리 (Transformers 5.x)
    if hasattr(past_key_values, 'layers'):
        for layer in past_key_values.layers:
            if hasattr(layer, 'keys') and layer.keys is not None:
                total_elements += layer.keys.numel() + layer.values.numel()
    elif hasattr(past_key_values, 'get_seq_length'):
        pass  # 크기 계산 불가 시 0 반환
    else:
        for layer_kv in past_key_values:
            if layer_kv is None:
                continue
            k, v = layer_kv
            total_elements += k.numel() + v.numel()

    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    return total_elements * bytes_per_element / (1024 ** 2)

def to_legacy_cache(past_key_values):
    """DynamicCache → tuple of (k, v) per layer 변환."""
    # Transformers 5.x DynamicCache (DynamicLayer 구조)
    if hasattr(past_key_values, 'layers'):
        return tuple(
            (layer.keys, layer.values)
            for layer in past_key_values.layers
        )
    # key_cache/value_cache 구조 (구버전 DynamicCache)
    if hasattr(past_key_values, 'key_cache'):
        return tuple(
            (k, v)
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
        )
    # 이미 튜플 형태
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    return past_key_values

def tuple_to_dynamic_cache(past_key_values):
    """tuple of (k, v) → DynamicCache 변환 (Transformers 5.x generate() 호환)."""
    from transformers import DynamicCache
    if isinstance(past_key_values, (tuple, list)):
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(k, v, layer_idx)
        return cache
    return past_key_values

def select_topk_tokens(
    scores: torch.Tensor,
    budget: int,
    min_budget: int = 4,
) -> torch.Tensor:
    """
    점수 기반 상위 K개 토큰 인덱스 선택.
    
    Args:
        scores: (seq_len,) 중요도 점수
        budget: 유지할 토큰 수
        min_budget: 최소 유지 토큰 수
    
    Returns:
        정렬된 인덱스 (seq_len 순서 유지)
    """
    seq_len = scores.shape[0]
    budget = max(min(budget, seq_len), min_budget)
    
    _, topk_indices = torch.topk(scores, k=budget, largest=True, sorted=False)
    topk_indices, _ = torch.sort(topk_indices)  # 원래 순서 유지
    return topk_indices


def apply_token_selection(
    past_key_values: Tuple,
    layer_indices: List[torch.Tensor],
) -> Tuple:
    """
    레이어별 토큰 선택 인덱스를 KV 캐시에 적용.
    
    Args:
        past_key_values: 원본 KV 캐시
        layer_indices: 레이어별 선택할 토큰 인덱스 리스트
    
    Returns:
        압축된 KV 캐시
    """
    compressed = []
    for layer_idx, layer_kv in enumerate(past_key_values):
        if layer_kv is None:
            compressed.append(None)
            continue
        
        k, v = layer_kv
        indices = layer_indices[layer_idx]
        
        if indices is None:
            compressed.append((k, v))
            continue
        
        # k, v shape: (batch, num_kv_heads, seq_len, head_dim)
        indices = indices.to(k.device)
        k_compressed = k[:, :, indices, :]
        v_compressed = v[:, :, indices, :]
        compressed.append((k_compressed, v_compressed))
    
    return tuple(compressed)
