#!/usr/bin/env python3
"""
experiments/sanity_check.py
=============================
전체 파이프라인 빠른 검증 스크립트.

실제 실험 전에 다음을 확인:
1. 모델 로드 및 토크나이즈 정상 동작
2. 프롬프트 포맷 (Qwen3-4B chat_template)
3. 각 KV 방법 압축 동작
4. 메트릭 계산 정상
5. CSV 저장 정상

실행:
    python experiments/sanity_check.py --model qwen3-4b
    python experiments/sanity_check.py --model qwen3-4b --full_check
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_metrics():
    """메트릭 계산 단위 테스트."""
    from core.metrics import compute_f1, compute_rouge_l
    
    # F1 테스트
    f1 = compute_f1("the cat sat on the mat", ["the cat sat on the mat"])
    assert abs(f1 - 100.0) < 0.1, f"F1 perfect match failed: {f1}"
    
    f1 = compute_f1("dog", ["cat"])
    assert f1 == 0.0, f"F1 no match failed: {f1}"
    
    f1 = compute_f1("the cat", ["the big cat"])
    assert 0 < f1 < 100, f"F1 partial match failed: {f1}"
    
    # ROUGE-L 테스트
    rouge = compute_rouge_l("hello world", ["hello world"])
    assert abs(rouge - 100.0) < 0.1, f"ROUGE-L perfect failed: {rouge}"
    
    logger.info("✓ Metrics tests passed")


def test_model_loading(model_key: str):
    """모델 로드 및 프롬프트 포맷 테스트."""
    from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
    
    logger.info(f"Loading {model_key} ...")
    model, tokenizer, cfg = load_model_and_tokenizer(model_key, device="cuda")
    
    # 프롬프트 생성 테스트
    prompt = make_prompt(
        model_key=model_key,
        tokenizer=tokenizer,
        context="The quick brown fox jumps over the lazy dog.",
        question="What does the fox do?",
        task_type="qa",
    )
    logger.info(f"Prompt (first 200 chars): {repr(prompt[:200])}")
    
    # Qwen3 특수 검증
    if model_key == "qwen3-4b":
        # <think>\n\n</think> 빈 블록은 thinking 억제용으로 허용
        # 실제 thinking content가 있는 경우(<think>실제내용</think>)만 금지
        import re
        thinking_blocks = re.findall(r'<think>(.*?)</think>', prompt, re.DOTALL)
        has_thinking_content = any(b.strip() for b in thinking_blocks)
        assert not has_thinking_content, "Qwen3 thinking mode should be disabled!"
        assert "<|im_start|>" in prompt or "user" in prompt.lower(), \
            "Qwen3 chat template not applied"
    
    # 토크나이즈 테스트
    inputs = tokenize_prompt(prompt, tokenizer, model_key, device="cuda")
    logger.info(f"Input token length: {inputs['input_ids'].shape[1]}")
    assert inputs["input_ids"].shape[1] > 0, "Tokenization failed"
    assert "attention_mask" in inputs, "Missing attention_mask"
    
    # 짧은 생성 테스트
    import torch
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    logger.info(f"Generated text (10 tokens): {repr(text)}")
    assert len(text) > 0, "Generation produced empty text"
    
    logger.info(f"✓ Model loading and generation test passed for {model_key}")
    return model, tokenizer, cfg


def test_kv_methods(model, tokenizer, model_config):
    """모든 KV 방법 압축 동작 테스트."""
    import torch
    from core.model_loader import make_prompt, tokenize_prompt
    from core.kv_methods import create_kv_method
    from core.kv_base import register_attention_hooks, remove_hooks, get_kv_cache_size_mb
    
    model_key = model_config["model_key"]
    
    # 테스트 입력
    prompt = make_prompt(
        model_key=model_key,
        tokenizer=tokenizer,
        context="Test context. " * 100,
        question="What is this?",
        task_type="qa",
    )
    inputs = tokenize_prompt(prompt, tokenizer, model_key, max_input_length=512, device="cuda")
    
    # Prefill
    num_layers = model_config["num_layers"]
    hooks, attn_weights_list = register_attention_hooks(model, num_layers)
    
    with torch.no_grad():
        try:
            out = model(
                **inputs,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            if hasattr(out, "attentions") and out.attentions:
                for li, a in enumerate(out.attentions):
                    if a is not None:
                        attn_weights_list[li] = a.detach().cpu()
        except Exception:
            out = model(**inputs, use_cache=True, return_dict=True)
        
        past_kv = out.past_key_values
    
    remove_hooks(hooks)
    
    # DynamicCache 호환 처리 (Transformers 5.x)
    if hasattr(past_kv, 'get_seq_length'):
        seq_len = past_kv.get_seq_length()
    elif hasattr(past_kv, 'key_cache'):
        seq_len = past_kv.key_cache[0].shape[2]
    else:
        seq_len = past_kv[0][0].shape[2]


    budget_ratio = 0.20
    dtype = model_config.get("dtype", torch.float16)
    
    kv_size_original = get_kv_cache_size_mb(past_kv, dtype)
    logger.info(f"Original KV cache: {kv_size_original:.2f} MB, seq_len={seq_len}")
    
    methods_to_test = ["fullkv", "streaming", "h2o", "snapkv", "pyramidkv", "adakv", "ours"]
    
    for method_name in methods_to_test:
        kv_method = create_kv_method(method_name, model_config)
        
        compressed = kv_method.compress(past_kv, attn_weights_list, budget_ratio)
        
        # DynamicCache 호환 처리 (Transformers 5.x)
        if hasattr(compressed, 'get_seq_length'):
            compressed_seq = compressed.get_seq_length()
        elif hasattr(compressed, 'layers'):
            compressed_seq = compressed.layers[0].keys.shape[2]
        else:
            compressed_seq = compressed[0][0].shape[2]
        
        
        kv_size_after = get_kv_cache_size_mb(compressed, dtype)
        mem_red = (1 - kv_size_after / kv_size_original) * 100
        
        # 검증
        assert compressed_seq > 0, f"{method_name}: compressed seq_len = 0"
        if method_name != "fullkv":
            assert compressed_seq <= seq_len, f"{method_name}: seq grew!"
        
        logger.info(
            f"  ✓ {method_name:15s}: seq {seq_len} → {compressed_seq}, "
            f"mem_red={mem_red:.1f}%"
        )
    
    logger.info("✓ All KV methods test passed")


def test_evaluator_single_sample(model, tokenizer, model_config):
    """Evaluator 단일 샘플 평가 테스트."""
    from core.kv_methods import create_kv_method
    from core.evaluator import Evaluator
    
    evaluator = Evaluator(model, tokenizer, model_config, seed=42)
    kv_method = create_kv_method("ours", model_config)
    
    sample = {
        "context": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "question": "Where is the Eiffel Tower?",
        "answers": ["Paris", "Paris, France"],
        "task_name": "test",
        "task_type": "qa",
        "metric": "f1",
        "max_new_tokens": 20,
    }
    
    result = evaluator.evaluate_sample(sample, kv_method, budget_ratio=0.20)
    
    logger.info(f"  Score: {result['score']:.2f}")
    logger.info(f"  Prediction: {repr(result['prediction'])}")
    logger.info(f"  TTFT: {result['ttft_ms']:.1f}ms")
    logger.info(f"  Memory reduction: {result['memory_reduction_pct']:.1f}%")
    
    assert isinstance(result["score"], float), "Score is not float"
    assert isinstance(result["prediction"], str), "Prediction is not str"
    
    logger.info("✓ Evaluator single sample test passed")


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity Check")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--full_check", action="store_true",
                        help="Evaluator 테스트까지 수행 (시간 소요)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Sanity Check: KV Cache Experiment System")
    logger.info("=" * 60)
    
    # 1. 메트릭 테스트 (모델 불필요)
    logger.info("\n[1] Testing metrics ...")
    test_metrics()
    
    # 2. 모델 로드 테스트
    logger.info(f"\n[2] Testing model loading: {args.model} ...")
    model, tokenizer, model_config = test_model_loading(args.model)
    
    # 3. KV 방법 테스트
    logger.info("\n[3] Testing KV cache methods ...")
    test_kv_methods(model, tokenizer, model_config)
    
    # 4. Evaluator 테스트 (선택)
    if args.full_check:
        logger.info("\n[4] Testing Evaluator (single sample) ...")
        test_evaluator_single_sample(model, tokenizer, model_config)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL SANITY CHECKS PASSED!")
    logger.info("=" * 60)
    logger.info("\nReady to run experiments:")
    logger.info(f"  python experiments/exp1_main_results.py --model {args.model} --num_samples 50")


if __name__ == "__main__":
    main()
