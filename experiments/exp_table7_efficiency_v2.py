"""
experiments/exp_table7_efficiency_v2.py
========================================
TABLE VII v2: Efficiency (30 independent prompts per sequence length, GPT 19차 검토 반영)

기존 v1의 문제(GPT 지적): 같은 프롬프트 하나를 N번 반복 측정 -> "timing repetition"이지
"30-sample 실험"이 아님. 다른 모든 TABLE(II~VI, IX)의 N=30/task와 개념이 안 맞음.

v2 설계:
- 3개 긴 컨텍스트 태스크(gov_report/qmsum/narrativeqa) x 10개 서로 다른 실제 문서 = 30개 독립 프롬프트
- 사전 확인 결과(2026-08-22) 15개 샘플 전부 3개 태스크 모두 8192토큰을 초과 -> 이어붙이기 불필요,
  개별 실제 문서를 tokenize_prompt()로 target_len에 truncate(챗 suffix 보존)
- 동일한 30개 문서를 세 controlled length(2048/4096/8192)에서 재사용(90 independent prompts 아님 -
  "30 documents x 3 controlled lengths"로 표현할 것, GPT 19차 검토)
- FullKV/AdaKV-adapted/SalienceKV-Sink4를 동일 30개 프롬프트에 paired 적용
- [2026-08-22 GPT 19차 검토] method-major 대신 prompt-major + method order rotation:
  프롬프트마다 3가지 순서([0,1,2]/[1,2,0]/[2,0,1])를 순환시켜 GPU 온도/클럭 등 시간 drift가
  특정 method에만 몰리는 걸 방지
- warm-up: 각 seq_len에서 방법별로 1회씩(One warm-up run per method per context length)
- [2026-08-22 GPT 19차 검토] N=30을 assert로 강제(조용히 skip 금지, seq_len 불일치 시 즉시 실패)
- torch.cuda.synchronize() 전부 유지, KV footprint는 element_size() 기반 유지
- "ttft_ms" 대신 "prefill_plus_comp_ms"로 명명(진짜 TTFT를 직접 측정한 게 아니므로, GPT 19차 검토)

⚠️ 실행 전 GPT 검토 대기 중 — 아직 실행되지 않았습니다.
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
TASKS = ["gov_report", "qmsum", "narrativeqa"]
N_PER_TASK = 10  # 3 tasks x 10 = 30 prompts per seq_len
SEED = 42

METHODS = [
    ("FullKV", "fullkv", {}),
    ("AdaKV-adapted", "adakv", {"invert_norm": True}),
    ("SalienceKV-Sink4", "ours", {"invert_norm": True, "sink_size": 4}),
]


def kv_size_mb(cache):
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


def run_one(model, model_cfg, method_name, method_kwargs, input_ids, attention_mask, device, budget_ratio):
    """1개 프롬프트에 대한 1회 측정: prefill -> compress -> 고정길이 decode."""
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
    del prefill_out
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
    if compressed_cache is not prefill_kv:
        del prefill_kv

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
        "prefill_plus_comp_ms": prefill_ms + compress_ms,
        "decode_throughput": decode_throughput,
        "kv_before_mb": kv_before_mb, "kv_after_mb": kv_after_mb,
        "mem_reduction_pct": mem_reduction_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", type=int, nargs="+", default=TARGET_SEQ_LENS)
    parser.add_argument("--n_per_task", type=int, default=N_PER_TASK)
    args = parser.parse_args()

    log_dir = "logs/v3_verified"
    results_dir = "results/v3_verified"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(f"{log_dir}/table7_v2_efficiency_{get_timestamp()}.log")],
        force=True,
    )
    import core.results_manager as rm
    rm.RESULTS_DIR = results_dir

    expected_n = len(TASKS) * args.n_per_task
    logger.info(f"TABLE VII v2: Efficiency ({expected_n} independent prompts) | seq_lens={args.seq_lens} | "
                f"tasks={TASKS} | n_per_task={args.n_per_task} | budget={BUDGET_RATIO:.0%}")

    model, tokenizer, model_cfg = load_model_and_tokenizer("qwen3-4b")
    device = next(model.parameters()).device

    # [2026-08-22 수정] gov_report 15개 샘플 중 최소 6512토큰(=8192 미달)이 실제로 assert 실패를
    # 유발함 - "15개면 충분"이라는 이전 확인이 틀렸음(N=10 draw는 정렬 안 된 순서라 짧은 문서가
    # 포함될 수 있음). 더 큰 pool(POOL_SIZE)에서 max(seq_lens) 이상인 문서만 필터링해서
    # 안정된 n_per_task개를 확보 - 이 고정 세트를 세 길이 전부에서 재사용(paired scaling 설계 유지).
    POOL_SIZE = 30
    max_seq_len = max(args.seq_lens)
    raw_samples = []  # [(task, idx, sample_dict), ...]
    for task in TASKS:
        pool = load_longbench_task(task, num_samples=POOL_SIZE, seed=SEED)
        valid = []
        for i, s in enumerate(pool):
            prompt = make_prompt(model_key="qwen3-4b", tokenizer=tokenizer, context=s["context"],
                                  question=s["question"], task_type=s["task_type"])
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) >= max_seq_len:
                valid.append((task, i, s))
            if len(valid) >= args.n_per_task:
                break
        assert len(valid) >= args.n_per_task, \
            f"{task}: pool({POOL_SIZE}) 중 {max_seq_len}토큰 이상 문서가 {len(valid)}개뿐, " \
            f"{args.n_per_task}개 필요 - POOL_SIZE를 늘려야 함"
        raw_samples.extend(valid)
    logger.info(f"확보된 프롬프트 pool: {len(raw_samples)}개 (기대값 {len(TASKS)*args.n_per_task}개, "
                f"각 문서는 최대 목표 길이({max_seq_len}) 이상으로 사전 검증됨)")

    all_results = []
    timestamp = get_timestamp()

    for seq_len in args.seq_lens:
        logger.info(f"\n{'='*60}\nSeqLen: {seq_len}\n{'='*60}")

        # 30개 프롬프트를 이 seq_len에 맞춰 토큰화
        tokenized_prompts = []
        for task, idx, s in raw_samples:
            prompt = make_prompt(model_key="qwen3-4b", tokenizer=tokenizer, context=s["context"],
                                  question=s["question"], task_type=s["task_type"])
            inputs = tokenize_prompt(prompt, tokenizer, "qwen3-4b", max_input_length=seq_len, device=str(device))
            # [2026-08-22 GPT 19차 검토] 조용히 skip하지 않고 강제 assert - N=30 보장
            assert inputs["input_ids"].shape[1] == seq_len, \
                f"{task}[{idx}] seq_len 불일치: {inputs['input_ids'].shape[1]} != {seq_len}"
            tokenized_prompts.append((task, idx, inputs["input_ids"], inputs["attention_mask"]))

        expected_n = len(TASKS) * args.n_per_task
        assert len(tokenized_prompts) == expected_n, \
            f"seq_len={seq_len}: usable prompts {len(tokenized_prompts)} != {expected_n}"
        logger.info(f"  토큰화 완료: {len(tokenized_prompts)}/{expected_n}개 프롬프트 확보(assert 통과)")

        # [2026-08-22 GPT 19차 검토] warm-up: 방법별로 1회씩
        for label, method_name, method_kwargs in METHODS:
            wu_task, wu_idx, wu_ids, wu_mask = tokenized_prompts[0]
            run_one(model, model_cfg, method_name, method_kwargs, wu_ids, wu_mask, device, BUDGET_RATIO)
        logger.info(f"  warm-up 완료: 방법 {len(METHODS)}개 × 1회")

        # [2026-08-22 GPT 19차 검토, 가장 중요한 수정] method-major -> prompt-major + method order rotation
        # 3가지 순서를 순환시켜 GPU 온도/클럭 등 시간 drift가 특정 method에만 몰리지 않게 함
        method_orders = [
            [0, 1, 2], [1, 2, 0], [2, 0, 1],
        ]
        per_method_results = {label: [] for label, _, _ in METHODS}

        for prompt_i, (task, idx, input_ids, attention_mask) in enumerate(tokenized_prompts):
            order = method_orders[prompt_i % len(method_orders)]
            for method_i in order:
                label, method_name, method_kwargs = METHODS[method_i]
                r = run_one(model, model_cfg, method_name, method_kwargs,
                            input_ids, attention_mask, device, BUDGET_RATIO)
                r["task"] = task
                r["sample_idx"] = idx
                per_method_results[label].append(r)
            if (prompt_i + 1) % 10 == 0:
                logger.info(f"  진행: {prompt_i+1}/{len(tokenized_prompts)}개 프롬프트 완료")

        for label, method_name, method_kwargs in METHODS:
            per_prompt_results = per_method_results[label]
            # [2026-08-22 GPT 19차 검토] "ttft_ms" -> "prefill_plus_comp_ms"로 명칭 변경
            # (진짜 TTFT를 직접 측정한 게 아니라 prefill+compression 합산값이므로)
            keys = ["prefill_ms", "compress_ms", "prefill_plus_comp_ms", "decode_throughput",
                    "kv_before_mb", "kv_after_mb", "mem_reduction_pct"]
            avg = {k: statistics.mean(x[k] for x in per_prompt_results) for k in keys}
            std = {f"{k}_std": (statistics.stdev(x[k] for x in per_prompt_results)
                                 if len(per_prompt_results) > 1 else 0.0) for k in keys}
            avg.update(std)
            avg["seq_len"] = seq_len
            avg["method"] = label
            avg["n_prompts"] = len(per_prompt_results)
            avg["per_prompt"] = per_prompt_results
            all_results.append(avg)

            logger.info(f"  평균(n={len(per_prompt_results)}): prefill={avg['prefill_ms']:.1f}±{avg['prefill_ms_std']:.1f}ms "
                        f"compress={avg['compress_ms']:.1f}±{avg['compress_ms_std']:.1f}ms "
                        f"decode_tp={avg['decode_throughput']:.2f}±{avg['decode_throughput_std']:.2f}tok/s "
                        f"mem_reduction={avg['mem_reduction_pct']:.1f}%")

            save_results_json({"results": all_results}, f"table7_v2_efficiency_{timestamp}.json")

    csv_rows = [{k: v for k, v in r.items() if k != "per_prompt"} for r in all_results]
    save_results_csv(csv_rows, f"table7_v2_efficiency_{timestamp}.csv")
    logger.info(f"\n완료: {len(all_results)}개 설정 (각 최대 {len(TASKS)*args.n_per_task}개 프롬프트 기반)")


if __name__ == "__main__":
    main()
