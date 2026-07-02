#!/usr/bin/env python3
"""
experiments/exp6_overhead.py
Exp6: 계산 오버헤드 분석 (evaluator_v2 기반)
- TTFT, throughput, memory reduction 측정
- 시퀀스 길이별 분석
"""
import sys, os, argparse, logging, time, gc
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.kv_cache_hook import make_hook_cache, BaseHookCache
from core.results_manager import save_results_csv, save_results_json, get_timestamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"results/exp6_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

TEST_LENGTHS = [2048, 4096, 8192]
METHODS = ["fullkv", "adakv", "ours"]
NUM_TRIALS = 3

def parse_args():
    parser = argparse.ArgumentParser(description="Exp6: Computational Overhead")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "phi-3-mini", "gemma-2-2b"])
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=None)
    return parser.parse_args()

def measure_single(model, tokenizer, model_config, method_name, budget, seq_length):
    device = next(model.parameters()).device
    dummy_text = "This is a test sentence for overhead measurement. " * (seq_length // 10 + 1)
    prompt = make_prompt(model_config["model_key"], tokenizer, dummy_text, "What is the main topic?", "qa")
    inputs = tokenize_prompt(prompt, tokenizer, model_config["model_key"], max_input_length=seq_length, device=str(device))
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    actual_len = input_ids.shape[1]

    prefill_times, compress_times, mem_reds = [], [], []

    for trial in range(NUM_TRIALS):
        gc.collect()
        torch.cuda.empty_cache()

        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            prefill_kv = out.past_key_values
        prefill_times.append((time.perf_counter() - t0) * 1000)

        kv_before_mb = sum(
            l.keys.numel() * 2 / (1024**2)
            for l in prefill_kv.layers if hasattr(l, 'keys') and l.keys is not None
        ) if hasattr(prefill_kv, 'layers') else 0.0

        # Compress
        t1 = time.perf_counter()
        if method_name != "fullkv":
            hook_cache = make_hook_cache(method_name, budget, model_config)
            for i, layer in enumerate(prefill_kv.layers):
                if i >= len(hook_cache.layers):
                    hook_cache.update(layer.keys, layer.values, i)
                else:
                    hook_cache.layers[i].keys = layer.keys.clone()
                    hook_cache.layers[i].values = layer.values.clone()
                hook_cache.set_prefill_keys(i, layer.keys)
            hook_cache.mark_prefill_done(actual_len)
            hook_cache.apply_compression_all_layers()
            kv_after_mb = sum(
                l.keys.numel() * 2 / (1024**2)
                for l in hook_cache.layers if hasattr(l, 'keys') and l.keys is not None
            )
            mem_red = (1 - kv_after_mb / kv_before_mb) * 100 if kv_before_mb > 0 else 0.0
        else:
            mem_red = 0.0
        compress_times.append((time.perf_counter() - t1) * 1000)
        mem_reds.append(mem_red)

        del prefill_kv
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "method": method_name,
        "seq_length": actual_len,
        "prefill_ms": round(sum(prefill_times)/len(prefill_times), 1),
        "compress_ms": round(sum(compress_times)/len(compress_times), 1),
        "total_ms": round((sum(prefill_times)+sum(compress_times))/len(prefill_times), 1),
        "mem_reduction_%": round(sum(mem_reds)/len(mem_reds), 1),
    }

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    timestamp = get_timestamp()
    seq_lengths = args.seq_lengths if args.seq_lengths else TEST_LENGTHS

    model, tokenizer, model_config = load_model_and_tokenizer(args.model, device="cuda")

    table_rows = []
    for seq_len in seq_lengths:
        for method_name in METHODS:
            logger.info(f"  {method_name} @ seq_len={seq_len}")
            try:
                result = measure_single(model, tokenizer, model_config, method_name, args.budget, seq_len)
                table_rows.append(result)
                logger.info(f"    prefill={result['prefill_ms']}ms, compress={result['compress_ms']}ms, mem_red={result['mem_reduction_%']}%")
            except Exception as e:
                logger.error(f"  Failed: {e}", exc_info=True)

    print("\n" + "="*75)
    print(f"  Exp6 Overhead | {args.model} | Budget={args.budget:.0%}")
    print("="*75)
    print(f"{'Method':<10} | {'SeqLen':>7} | {'Prefill':>9} | {'Compress':>10} | {'Total':>8} | {'Mem_Red%':>8}")
    print("-"*75)
    for r in table_rows:
        print(f"{r['method']:<10} | {r['seq_length']:>7} | {r['prefill_ms']:>7}ms | {r['compress_ms']:>8}ms | {r['total_ms']:>6}ms | {r['mem_reduction_%']:>7}%")
    print("="*75)

    save_results_csv(table_rows, f"exp6_overhead_{args.model}_{timestamp}.csv")
    save_results_json({"model": args.model, "results": table_rows}, f"exp6_overhead_{args.model}_{timestamp}.json")
    logger.info("Experiment 6 completed!")

if __name__ == "__main__":
    main()
