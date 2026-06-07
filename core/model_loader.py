"""
core/model_loader.py
====================
모델 로드 및 프롬프트 생성 모듈.

핵심 이슈 해결:
- Qwen3-4B: chat_template 사용, thinking_mode=False, enable_thinking=False
- Phi-3-mini: <|user|> / <|assistant|> 템플릿
- Gemma-2-2B: <start_of_turn> 템플릿
- GQA(Grouped Query Attention) 지원 — num_key_value_heads != num_attention_heads
- token embedding 오류: pad_token 명시적 설정
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 모델별 설정
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B",
        "num_layers": 32,
        "num_heads": 8,          # num_attention_heads (GQA query heads)
        "num_kv_heads": 8,       # num_key_value_heads (GQA kv heads) — Qwen3-4B는 8로 동일
        "head_dim": 128,
        "dtype": torch.float16,
        "max_length": 32768,
        "prompt_format": "qwen3_chat",
        "attn_impl": "eager",    # flash_attention_2 or eager
    },
    "phi-3-mini": {
        "hf_name": "microsoft/Phi-3-mini-128k-instruct",
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 96,
        "dtype": torch.float16,
        "max_length": 131072,
        "prompt_format": "phi3_chat",
        "attn_impl": "eager",
    },
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b-it",
        "num_layers": 26,
        "num_heads": 8,
        "num_kv_heads": 4,       # GQA: 4 kv heads
        "head_dim": 256,
        "dtype": torch.bfloat16,  # Gemma-2는 bfloat16 권장
        "max_length": 8192,
        "prompt_format": "gemma2_chat",
        "attn_impl": "eager",
    },
}


# ─────────────────────────────────────────────
# 프롬프트 생성 (모델별)
# ─────────────────────────────────────────────

def make_prompt(model_key: str, tokenizer, context: str, question: str, task_type: str = "qa") -> str:
    """
    모델별 올바른 프롬프트 포맷 생성.
    
    Args:
        model_key: MODEL_CONFIGS 키 (e.g., "qwen3-4b")
        tokenizer: HuggingFace tokenizer
        context: 문서/컨텍스트 텍스트
        question: 질문 또는 요약 지시
        task_type: "qa" | "summarization"
    
    Returns:
        모델에 입력할 완성된 프롬프트 문자열
    """
    fmt = MODEL_CONFIGS[model_key]["prompt_format"]
    
    if task_type == "qa":
        user_content = (
            f"Read the following text carefully and answer the question.\n\n"
            f"Text:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely based on the text."
        )
    else:  # summarization
        user_content = (
            f"Please summarize the following document concisely.\n\n"
            f"Document:\n{context}\n\n"
            f"Summary:"
        )

    if fmt == "qwen3_chat":
        # Qwen3: apply_chat_template 사용 (enable_thinking=False 필수)
        # thinking mode가 켜지면 <think> 토큰이 추가되어 성능 측정 왜곡 발생
        messages = [{"role": "user", "content": user_content}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Transformers 버전 호환: thinking 태그 강제 제거
        return prompt

    elif fmt == "phi3_chat":
        # Phi-3: <|user|>\n ... \n<|end|>\n<|assistant|>\n
        prompt = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n"
        return prompt

    elif fmt == "gemma2_chat":
        # Gemma-2: <start_of_turn>user\n ... <end_of_turn>\n<start_of_turn>model\n
        prompt = (
            f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return prompt

    else:
        raise ValueError(f"Unknown prompt format: {fmt}")


def tokenize_prompt(
    prompt: str,
    tokenizer,
    model_key: str,
    max_input_length: int = 31000,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    프롬프트를 토크나이즈하고 모델 입력 딕셔너리 반환.
    
    주요 처리:
    - truncation: 최대 길이 초과 시 컨텍스트 중간 부분 제거 (head + tail 보존)
    - attention_mask 명시적 생성
    - token_type_ids 제거 (일부 모델에서 오류 발생)
    """
    # 토크나이즈 (truncation 없이 먼저 길이 확인)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,  # chat_template에서 이미 처리됨
        truncation=False,
    )
    
    input_ids = encoded["input_ids"]
    seq_len = input_ids.shape[1]
    
    # 최대 길이 초과 시 truncation
    if seq_len > max_input_length:
        logger.warning(f"Input length {seq_len} > max {max_input_length}, truncating...")
        # head 500 + tail (max-500) 보존 전략
        head_len = 500
        tail_len = max_input_length - head_len
        input_ids = torch.cat([
            input_ids[:, :head_len],
            input_ids[:, -tail_len:]
        ], dim=1)
    
    attention_mask = torch.ones_like(input_ids)
    
    result = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    return result


def load_model_and_tokenizer(
    model_key: str,
    device: str = "cuda",
    use_flash_attn: bool = False,
) -> Tuple:
    """
    모델과 토크나이저 로드.
    
    Args:
        model_key: MODEL_CONFIGS 키
        device: "cuda" | "cpu"
        use_flash_attn: FlashAttention2 사용 여부 (호환 GPU 필요)
    
    Returns:
        (model, tokenizer, config_dict)
    """
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["hf_name"]
    dtype = cfg["dtype"]
    
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # ★ pad_token 설정 (없으면 token embedding 오류 발생)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # padding_side = left (decoder-only 모델 배치 처리 시 필요)
    tokenizer.padding_side = "left"
    
    logger.info(f"Loading model: {model_name} in {dtype}")
    
    attn_impl = cfg["attn_impl"]
    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("FlashAttention2 enabled")
        except ImportError:
            logger.warning("FlashAttention not installed, falling back to eager")
            attn_impl = "eager"
    
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
    }
    
    # Gemma-2: sliding_window attention 관련 경고 억제
    if model_key == "gemma-2-2b":
        model_kwargs["attn_implementation"] = "eager"

    # Qwen3: SDPA는 attn_weights=None 반환 → hook으로 attention 수집 불가
    # eager로 강제해야 Hybrid Score의 Attention 신호(α=0.40) 작동
    if model_key == "qwen3-4b":
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    
    # 실제 모델 config에서 head 정보 업데이트 (설정값과 다를 수 있음)
    actual_num_heads = model.config.num_attention_heads
    actual_kv_heads = getattr(model.config, "num_key_value_heads", actual_num_heads)
    actual_layers = model.config.num_hidden_layers
    
    # CONFIG를 실제 값으로 업데이트
    cfg = dict(cfg)
    cfg["num_heads"] = actual_num_heads
    cfg["num_kv_heads"] = actual_kv_heads
    cfg["num_layers"] = actual_layers
    cfg["model_key"] = model_key
    
    logger.info(
        f"Model loaded: layers={actual_layers}, "
        f"num_heads={actual_num_heads}, num_kv_heads={actual_kv_heads}"
    )
    
    return model, tokenizer, cfg


def get_model_info(model_key: str) -> Dict:
    """모델 설정 정보 반환 (모델 로드 없이)."""
    return dict(MODEL_CONFIGS[model_key])
