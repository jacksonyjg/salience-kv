"""
experiments/exp_table7_efficiency.py
========================================
TABLE VII: Efficiency (Prefill / Compression / TTFT / Decode Throughput / KV Memory)

GPT 검토 반영 설계(2026-08-22) — 기존 exp6_overhead.py(generation 없음, invert_norm/sink_size 누락,
CUDA synchronize 없음)와 exp_table9_efficiency.py(CUDA synchronize 없음) 둘 다 안 쓰고 신규 작성.

- 모델: Qwen3-4B, budget=20%
- 시퀀스 길이: 2048 / 4096 / 8192 (실제 gov_report 문서를 토큰 단위로 정확히 잘라 고정)
- 방법: FullKV, AdaKV-adapted(invert_norm=True), SalienceKV-Sink4(invert_norm=True, sink_size=4)
- 모든 GPU 타이밍 앞뒤로 torch.cuda.synchronize()
- warm-up 1회 + 측정 5회 평균
- 고정 길이 decode(100토큰)로 throughput 측정
- KV 캐시 크기: K+V 전부, 각 텐서의 element_size() 기반 정확한 byte 수 사용
- 저장: results/v3_verified/table7_efficiency_*.json/csv

논문 컬럼명 권고(GPT 검토, 2026-08-22):
  | Method | Seq. Len. | Prefill | Compression | Prefill+Comp.(TTFT) | Decode tok/s | KV Footprint | KV Reduction |
- TTFT는 "prompt 제출부터 압축 후 첫 토큰까지"로 조작적 정의(= prefill + compression latency).
  엄밀히는 첫 토큰 logits 자체는 prefill 직후 이미 존재하므로, 논문 Methods에 이 정의를 명시할 것.
- 메모리 컬럼은 "GPU Memory"가 아니라 "KV Cache Footprint/Reduction"으로 명명
  (모델 weight/activation 등 포함한 것으로 오해받지 않도록).

⚠️ 실행 전 GPT 검토 대기 중 — 이 코드는 아직 실행되지 않았습니다.
"""
import sys, os, argparse, logging, time, gc, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.kv_cache_hook import make_hook_cache, BaseHookCache
from core.results_manager import save_results_csv, save_results_json, get_timestamp

logger = logging.getLogger(__name__)

TARGET_SEQ_LENS = [2048, 4096, 8192]
BUDGET_RATIO = 0.20
DECODE_STEPS = 100
N_WARMUP = 1
N_TRIALS = 5

METHODS = [
    ("FullKV", "fullkv", {}),
    ("AdaKV-adapted", "adakv", {"invert_norm": True}),
    ("SalienceKV-Sink4", "ours", {"invert_norm": True, "sink_size": 4}),
]


def get_fixed_length_input(tokenizer, model_key, target_len, device):
    """실제 gov_report 문서를 이어붙여서 target_len 토큰으로 정확히 통제(효율성 측정 목적).
    [2026-08-22 GPT 지적 반영] tokenize_prompt()의 head+tail truncation을 사용해
    Qwen 챗 suffix(<|im_end|><|im_start|>assistant...)를 보존 - 단순 ids[:target_len]로
    앞에서만 자르면 이 suffix가 날아가 "요약 생성"이 아니라 "문서 continuation"을 측정하게 됨."""
    samples = load_longbench_task("gov_report", num_samples=5, seed=42)
    combined_context = " ".join(s["context"] for s in samples)
    prompt = make_prompt(model_key=model_key, tokenizer=tokenizer,
                          context=combined_context, question="", task_type="summarization")
    inputs = tokenize_prompt(prompt, tokenizer, model_key, max_input_length=target_len, device=str(device))
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    assert input_ids.shape[1] == target_len, \
        f"seq_len 불일치: {input_ids.shape[1]} != {target_len}"
    return input_ids, attention_mask


def kv_size_mb(cache):
    """[2026-08-22 GPT 지적 반영] model.dtype 기반 고정 bytes_per 대신,
    각 keys/values 텐서 자신의 element_size()를 사용 - 더 정확함."""
    total = 0
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
    return total / (1024 * 1024)


def run_one_trial(model, tokenizer, model_cfg, method_name, method_kwargs,
                   input_ids, attention_mask, device, budget_ratio):
    """1회 시행: prefill -> compress -> 고정길이 decode. 전부 synchronize로 감쌈."""
    seq_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=True, return_dict=True, output_attentions=False)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    prefill_kv = prefill_out.past_key_values
    last_logits = prefill_out.logits[:, -1, :].detach().clone()
    del prefill_out  # [2026-08-22 GPT 지적] FullKV reference를 계속 잡고 있지 않도록 즉시 해제
    kv_before_mb = kv_size_mb(prefill_kv)

    hook_cache = make_hook_cache(method_name, budget_ratio, model_cfg, **method_kwargs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if isinstance(hook_cache, BaseHookCache):
        if hasattr(prefill_kv, "layers"):
            for i, layer in enumerate(prefill_kv.layers):
                if i >= len(hook_cache.layers):
                    hook_cache.update(layer.keys, layer.values, i)
                else:
                    hook_cache.layers[i].keys = layer.keys.clone()
                    hook_cache.layers[i].values = layer.values.clone()
                hook_cache.set_prefill_keys(i, layer.keys)
        hook_cache.mark_prefill_done(seq_len)
        hook_cache.apply_compression_all_layers()
        compressed_cache = hook_cache
    else:
        compressed_cache = prefill_kv
    torch.cuda.synchronize()
    compress_ms = (time.perf_counter() - t0) * 1000

    kv_after_mb = kv_size_mb(compressed_cache)

    # [2026-08-22 GPT 지적] compression 완료 후 원본 prefill_kv reference도 정리
    if compressed_cache is not prefill_kv:
        del prefill_kv

    # 고정 길이 decode (throughput 측정)
    cache_len = compressed_cache.layers[0].keys.shape[2] if hasattr(compressed_cache, "layers") else seq_len
    next_token = last_logits.argmax(dim=-1, keepdim=True)
    cur_pos = seq_len

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(DECODE_STEPS):
            cache_position = torch.tensor([cur_pos], device=device)
            attn_mask_step = torch.ones(1, cache_len + step + 1, dtype=attention_mask.dtype, device=device)
            step_out = model(input_ids=next_token, past_key_values=compressed_cache,
                              cache_position=cache_position, attention_mask=attn_mask_step,
                              use_cache=True, return_dict=True)
            next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_pos += 1
    torch.cuda.synchronize()
    decode_elapsed_s = time.perf_counter() - t0
    decode_throughput = DECODE_STEPS / decode_elapsed_s if decode_elapsed_s > 0 else 0.0

    mem_reduction_pct = (1 - kv_after_mb / kv_before_mb) * 100 if kv_before_mb > 0 else 0.0

    del step_out, next_token, last_logits
    del compressed_cache, hook_cache
    if "prefill_kv" in locals():
        del prefill_kv
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "prefill_ms": prefill_ms, "compress_ms": compress_ms,
        "ttft_ms": prefill_ms + compress_ms,
        "decode_throughput": decode_throughput,
        "kv_before_mb": kv_before_mb, "kv_after_mb": kv_after_mb,
        "mem_reduction_pct": mem_reduction_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", type=int, nargs="+", default=TARGET_SEQ_LENS)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    n_trials = args.n_trials

    log_dir = "logs/v3_verified"
    results_dir = "results/v3_verified"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(f"{log_dir}/table7_efficiency_{get_timestamp()}.log")],
        force=True,
    )
    import core.results_manager as rm
    rm.RESULTS_DIR = results_dir

    logger.info(f"TABLE VII: Efficiency | seq_lens={args.seq_lens} | budget={BUDGET_RATIO:.0%} | "
                f"warmup={N_WARMUP} trials={n_trials} decode_steps={DECODE_STEPS}")

    model, tokenizer, model_cfg = load_model_and_tokenizer("qwen3-4b")
    device = next(model.parameters()).device

    all_results = []
    timestamp = get_timestamp()

    for seq_len in args.seq_lens:
        logger.info(f"\n{'='*60}\nSeqLen: {seq_len}\n{'='*60}")
        input_ids, attention_mask = get_fixed_length_input(tokenizer, "qwen3-4b", seq_len, device)

        for label, method_name, method_kwargs in METHODS:
            logger.info(f"Method: {label} | kwargs={method_kwargs}")

            # warm-up
            for _ in range(N_WARMUP):
                run_one_trial(model, tokenizer, model_cfg, method_name, method_kwargs,
                               input_ids, attention_mask, device, BUDGET_RATIO)

            trials = []
            for t in range(n_trials):
                r = run_one_trial(model, tokenizer, model_cfg, method_name, method_kwargs,
                                   input_ids, attention_mask, device, BUDGET_RATIO)
                trials.append(r)
                logger.info(f"  trial {t+1}/{n_trials}: prefill={r['prefill_ms']:.1f}ms "
                            f"compress={r['compress_ms']:.1f}ms decode_tp={r['decode_throughput']:.2f}tok/s")

            avg = {k: statistics.mean(x[k] for x in trials) for k in trials[0].keys()}
            std = {f"{k}_std": (statistics.stdev(x[k] for x in trials) if len(trials) > 1 else 0.0)
                   for k in trials[0].keys()}
            avg.update(std)
            avg["seq_len"] = seq_len
            avg["method"] = label
            avg["trials"] = trials  # JSON 전용 - CSV 저장 시에는 별도 요약 dict 사용
            all_results.append(avg)

            logger.info(f"  평균: prefill={avg['prefill_ms']:.1f}ms compress={avg['compress_ms']:.1f}ms "
                        f"ttft={avg['ttft_ms']:.1f}ms decode_tp={avg['decode_throughput']:.2f}tok/s "
                        f"mem_reduction={avg['mem_reduction_pct']:.1f}%")

            save_results_json({"results": all_results}, f"table7_efficiency_{timestamp}.json")

    csv_rows = [{k: v for k, v in r.items() if k != "trials"} for r in all_results]
    save_results_csv(csv_rows, f"table7_efficiency_{timestamp}.csv")
    logger.info(f"\n완료: {len(all_results)}개 설정")


if __name__ == "__main__":
    main()
