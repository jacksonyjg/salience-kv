#!/usr/bin/env python3
"""
experiments/exp6_overhead.py
==============================
실험 6: 계산 오버헤드 분석

목적: 압축 방법이 추가하는 계산 비용 측정
- prefill 단계 추가 시간 (ms)
- 메모리 오버헤드 (MB)
- FlashAttention 활성화/비활성화 비교

실행:
    python experiments/exp6_overhead.py --model qwen3-4b
"""

import sys
import os
import argparse
import time
import logging
import gc

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.kv_methods import create_kv_method
from core.kv_base import register_attention_hooks, remove_hooks, get_kv_cache_size_mb
from core.results_manager import (
    save_results_csv, save_results_json, print_result_table, get_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"results/exp6_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

# 오버헤드 측정용 합성 입력 길이
TEST_LENGTHS = [4096, 8192, 16384]
METHODS_TO_TEST = ["fullkv", "adakv", "ours"]
NUM_WARMUP = 3
NUM_TRIALS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Exp6: Computational Overhead")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def measure_overhead(
    model,
    tokenizer,
    model_config,
    method_name,
    budget_ratio,
    seq_length,
    num_warmup=NUM_WARMUP,
    num_trials=NUM_TRIALS,
):
    """
    특정 시퀀스 길이에서 압축 방법의 오버헤드 측정.
    
    Returns:
        {
            "prefill_ms": float,
            "compress_ms": float,
            "total_ttft_ms": float,
            "peak_memory_mb": float,
            "kv_size_before_mb": float,
            "kv_size_after_mb": float,
        }
    """
    model_key = model_config["model_key"]
    device = next(model.parameters()).device
    
    kv_method = create_kv_method(method_name, model_config)
    
    # 합성 입력 생성 (반복 텍스트로 목표 길이 달성)
    dummy_text = "This is a test sentence for overhead measurement. " * (seq_length // 10 + 1)
    prompt = make_prompt(
        model_key=model_key,
        tokenizer=tokenizer,
        context=dummy_text,
        question="What is the main topic?",
        task_type="qa",
    )
    inputs = tokenize_prompt(prompt, tokenizer, model_key, max_input_length=seq_length, device=str(device))
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    actual_len = input_ids.shape[1]
    
    prefill_times = []
    compress_times = []
    peak_memories = []
    kv_before_sizes = []
    kv_after_sizes = []
    
    dtype = model_config.get("dtype", torch.float16)
    num_layers = model_config["num_layers"]
    
    for trial in range(num_warmup + num_trials):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        hooks, attn_weights_list = register_attention_hooks(model, num_layers)
        
        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
                if hasattr(out, "attentions") and out.attentions:
                    for li, attn in enumerate(out.attentions):
                        if attn is not None:
                            attn_weights_list[li] = attn.detach().cpu()
            except Exception:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = out.past_key_values
        t1 = time.perf_counter()
        
        remove_hooks(hooks)
        
        kv_before = get_kv_cache_size_mb(past_kv, dtype)
        
        # 압축
        t2 = time.perf_counter()
        compressed_kv = kv_method.compress(past_kv, attn_weights_list, budget_ratio)
        t3 = time.perf_counter()
        
        kv_after = get_kv_cache_size_mb(compressed_kv, dtype)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        if trial >= num_warmup:
            prefill_times.append((t1 - t0) * 1000)
            compress_times.append((t3 - t2) * 1000)
            kv_before_sizes.append(kv_before)
            kv_after_sizes.append(kv_after)
            peak_memories.append(peak_mem)
        
        del past_kv, compressed_kv, out
        gc.collect()
        torch.cuda.empty_cache()
    
    return {
        "seq_length": actual_len,
        "method": method_name,
        "prefill_ms": np.mean(prefill_times),
        "prefill_std_ms": np.std(prefill_times),
        "compress_ms": np.mean(compress_times),
        "compress_std_ms": np.std(compress_times),
        "total_ms": np.mean(prefill_times) + np.mean(compress_times),
        "peak_memory_mb": np.mean(peak_memories),
        "kv_size_before_mb": np.mean(kv_before_sizes),
        "kv_size_after_mb": np.mean(kv_after_sizes),
        "memory_reduction_pct": (
            (1 - np.mean(kv_after_sizes) / np.mean(kv_before_sizes)) * 100
            if np.mean(kv_before_sizes) > 0 else 0.0
        ),
    }


def main():
    args = parse_args()
    
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    
    seq_lengths = args.seq_lengths if args.seq_lengths else TEST_LENGTHS
    
    logger.info("Experiment 6: Computational Overhead Analysis")
    logger.info(f"  Model: {args.model} | Budget: {args.budget:.0%}")
    logger.info(f"  Seq lengths: {seq_lengths}")
    
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, device="cuda"
    )
    
    all_results = []
    table_rows = []
    
    for seq_len in seq_lengths:
        for method_name in METHODS_TO_TEST:
            logger.info(f"\n  Method={method_name}, seq_len={seq_len}")
            
            try:
                result = measure_overhead(
                    model=model,
                    tokenizer=tokenizer,
                    model_config=model_config,
                    method_name=method_name,
                    budget_ratio=args.budget,
                    seq_length=seq_len,
                )
                
                all_results.append(result)
                
                row = {
                    "Method": method_name.upper(),
                    "Seq_Len": result["seq_length"],
                    "Prefill_ms": round(result["prefill_ms"], 1),
                    "Compress_ms": round(result["compress_ms"], 1),
                    "Total_ms": round(result["total_ms"], 1),
                    "Peak_Mem_MB": round(result["peak_memory_mb"], 1),
                    "KV_Before_MB": round(result["kv_size_before_mb"], 1),
                    "KV_After_MB": round(result["kv_size_after_mb"], 1),
                    "Mem_Reduction_%": round(result["memory_reduction_pct"], 1),
                }
                table_rows.append(row)
                
                logger.info(
                    f"    prefill={result['prefill_ms']:.1f}ms, "
                    f"compress={result['compress_ms']:.1f}ms, "
                    f"mem_red={result['memory_reduction_pct']:.1f}%"
                )
            
            except Exception as e:
                logger.error(f"  Failed: {e}")
    
    print_result_table(table_rows, title=f"Exp6 Computational Overhead | {args.model}")
    save_results_csv(table_rows, f"exp6_overhead_{args.model}_{timestamp}.csv")
    save_results_json(
        {"experiment": "exp6", "model": args.model, "results": all_results},
        f"exp6_overhead_{args.model}_{timestamp}.json",
    )
    
    # 추가: 코드 줄 수 검증
    logger.info("\n--- Code Line Count Verification ---")
    core_files = [
        "core/kv_methods.py",
        "core/kv_base.py",
    ]
    for f in core_files:
        if os.path.exists(f):
            with open(f) as fh:
                lines = fh.readlines()
                non_empty = [l for l in lines if l.strip() and not l.strip().startswith("#")]
                logger.info(f"  {f}: {len(lines)} total lines, {len(non_empty)} code lines")
    
    logger.info("Experiment 6 completed!")


if __name__ == "__main__":
    main()
