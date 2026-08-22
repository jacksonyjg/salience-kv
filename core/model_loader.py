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
        # [2026-08-22 정식 반영] LongBench 공식 dataset2prompt.json의 QMSum 형식 채택.
        # question이 있으면(QMSum) 쿼리 기반 프롬프트, question=""이면(gov_report) 기존 동작
        # 완전히 그대로 유지 - "우리 이전 구현" 기준 byte-identical(LongBench 공식 gov_report
        # 템플릿과 동일하다는 뜻 아님 - 공식 gov_report 템플릿은 "You are given a report by a
        # government agency..."로 시작해서 현재 구현과 다름, 2026-08-22 GPT 21차 검토 지적).
        # 이 파일이 QMSum corrected 템플릿의 단일 출처(freeze).
        if question:
            user_content = (
                "You are given a meeting transcript and a query containing a question or "
                "instruction. Answer the query in one or more sentences.\n\n"
                f"Transcript:\n{context}\n\n"
                "Now, answer the query based on the above meeting transcript in one or more "
                "sentences.\n\n"
                f"Query: {question}\n"
                "Answer:"
            )
        else:
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
        # thinking 억제: 빈 think 블록으로 강제 종료
        if "<think>" in prompt:
            import re
            prompt = re.sub(r'<think>.*?</think>', '', prompt, flags=re.DOTALL).strip()
        # 프롬프트 끝에 빈 think 블록 추가 (개행 유무 무관하게 처리)
        if prompt.endswith('<|im_start|>assistant'):
            prompt = prompt + '\n<think>\n\n</think>\n\n'
        elif prompt.endswith('<|im_start|>assistant\n'):
            prompt = prompt + '<think>\n\n</think>\n\n'
        return prompt

    elif fmt == "phi3_chat":
        # Phi-3: <|user|>\n ... \n<|end|>\n<|assistant|>\n
        prompt = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n"
        return prompt

    elif fmt == "gemma2_chat":
        # Gemma-2: <bos> 필수 (없으면 생성이 완전히 붕괴됨, 2026-07-04 확인)
        bos = tokenizer.bos_token or "<bos>"
        prompt = (
            f"{bos}<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return prompt

    else:
        raise ValueError(f"Unknown prompt format: {fmt}")


def tokenize_prompt(
    prompt: str,
    tokenizer,
    model_key: str,
    max_input_length: int = 8192,
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
        # Phi-3-mini는 Transformers 5.x에 내장 아키텍처로 편입되어 있어 trust_remote_code 불필요
        # (저장소의 구버전 원격 코드는 DynamicCache.from_legacy_cache() 등 제거된 API에 의존해 비호환)
        "trust_remote_code": model_key != "phi-3-mini",
        "attn_implementation": attn_impl,
    }
    
    # Gemma-2: sliding_window attention 관련 경고 억제
    if model_key == "gemma-2-2b":
        model_kwargs["attn_implementation"] = "eager"

    # Qwen3: importance 계산은 key norm 기반(attn_weights 불필요, v3 기준)
    # sdpa로 전환하여 eager의 O(L^2) 메모리 문제 해결
    if model_key == "qwen3-4b":
        model_kwargs["attn_implementation"] = "sdpa"
    # Phi-3-mini: trust_remote_code=False(내장 아키텍처)로 전환 후 sdpa 재시도
    if model_key == "phi-3-mini":
        model_kwargs["attn_implementation"] = "sdpa"

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
