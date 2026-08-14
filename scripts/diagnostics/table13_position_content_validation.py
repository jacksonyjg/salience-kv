import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import logging
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
import core.evaluator_v2 as ev2_module
from core.evaluator_v2 import EvaluatorV2
from core.results_manager import save_results_csv, save_results_json, get_timestamp
import core.kv_cache_hook as hook_module

os.makedirs("logs/v2_verified", exist_ok=True)
os.makedirs("results/v2_verified", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/v2_verified/table13_{get_timestamp()}.log"),
    ],
)
logger = logging.getLogger(__name__)

MODEL_KEY = "qwen3-4b"
TASKS = ["qmsum", "gov_report"]
NUM_SAMPLES = 10
BUDGET_RATIO = 0.20
SINK_SIZE = 4
SEED = 42

_ORIGINAL_SELECT_WITH_SINK = hook_module._select_with_sink
_ORIGINAL_TOKENIZE_PROMPT = ev2_module.tokenize_prompt


def make_patched_select_with_sink(slot: str):
    def patched(score, seq_len, budget, recent_w, sink_size, device):
        if slot == "none":
            sink_size = 0

        sink_size = max(min(sink_size, budget, seq_len), 0)
        recent_w = max(min(recent_w, max(budget - sink_size, 0), seq_len), 0)
        mid_budget = max(budget - sink_size - recent_w, 0)

        if sink_size == 0:
            sink_idx = torch.empty(0, dtype=torch.long, device=device)
        elif slot == "front":
            sink_idx = torch.arange(sink_size, device=device)
        elif slot == "middle":
            mid_point = seq_len // 2
            start = max(mid_point - sink_size // 2, 0)
            sink_idx = torch.arange(start, start + sink_size, device=device)
        elif slot == "end":
            end_point = max(seq_len - recent_w - sink_size, 0)
            sink_idx = torch.arange(end_point, end_point + sink_size, device=device)
        else:
            raise ValueError(f"unknown slot: {slot}")

        if recent_w > 0:
            recent_idx = torch.arange(seq_len - recent_w, seq_len, device=device)
        else:
            recent_idx = torch.empty(0, dtype=torch.long, device=device)

        excluded = set(sink_idx.tolist()) | set(recent_idx.tolist())
        all_idx = torch.arange(seq_len, device=device)
        candidate_mask = torch.tensor([i not in excluded for i in range(seq_len)], device=device)
        candidate_idx = all_idx[candidate_mask]

        if mid_budget > 0 and len(candidate_idx) > 0:
            candidate_score = score[candidate_idx]
            k = min(mid_budget, len(candidate_idx))
            _, top_local = torch.topk(candidate_score, k=k)
            mid_idx = candidate_idx[top_local]
        else:
            mid_idx = torch.empty(0, dtype=torch.long, device=device)

        indices = torch.cat([sink_idx, mid_idx, recent_idx])
        indices = torch.unique(indices, sorted=True)
        return indices

    return patched


def make_placeholder_tokenize_prompt(sink_size: int, placeholder_token_id: int):
    def patched(prompt, tokenizer, model_key, max_input_length, device):
        inputs = _ORIGINAL_TOKENIZE_PROMPT(prompt, tokenizer, model_key, max_input_length, device)
        input_ids = inputs["input_ids"].clone()
        n = min(sink_size, input_ids.shape[1])
        input_ids[:, :n] = placeholder_token_id
        inputs["input_ids"] = input_ids
        return inputs
    return patched


def run_condition(evaluator, condition_name, mode, tasks, num_samples, budget, seed, tokenizer):
    logger.info(f"\n{'='*60}")
    logger.info(f"Condition: {condition_name} (mode={mode})")
    logger.info(f"{'='*60}")

    kind, slot = mode
    if kind == "position":
        hook_module._select_with_sink = make_patched_select_with_sink(slot)
        ev2_module.tokenize_prompt = _ORIGINAL_TOKENIZE_PROMPT
        sink_size_kwarg = SINK_SIZE if slot != "none" else 0
    elif kind == "content":
        hook_module._select_with_sink = _ORIGINAL_SELECT_WITH_SINK
        placeholder_id = tokenizer.pad_token_id
        ev2_module.tokenize_prompt = make_placeholder_tokenize_prompt(SINK_SIZE, placeholder_id)
        sink_size_kwarg = SINK_SIZE
    else:
        raise ValueError(mode)

    task_scores = {}
    for task_name in tasks:
        samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
        scores = []
        for i, sample in enumerate(samples):
            r = evaluator.evaluate_sample(
                sample, "ours", budget,
                measure_efficiency=False,
                method_kwargs={"sink_size": sink_size_kwarg},
            )
            scores.append(r["score"])
            if (i + 1) % 5 == 0:
                logger.info(f"  [{condition_name}] {task_name} [{i+1}/{len(samples)}] "
                            f"avg={sum(scores)/len(scores):.2f}")
        avg = sum(scores) / len(scores) if scores else 0.0
        task_scores[task_name] = round(avg, 2)
        logger.info(f"  → {task_name}: {avg:.2f}")

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0

    hook_module._select_with_sink = _ORIGINAL_SELECT_WITH_SINK
    ev2_module.tokenize_prompt = _ORIGINAL_TOKENIZE_PROMPT

    return {
        "condition": condition_name,
        "mode": f"{kind}:{slot}",
        "task_scores": task_scores,
        "avg_score": round(avg_score, 2),
    }


def main():
    logger.info("TABLE XIII: Sink Position/Content Controlled Validation (경량 확인용)")
    logger.info(f"  Model: {MODEL_KEY} | Tasks: {TASKS} | Samples: {NUM_SAMPLES}/task | Budget: {BUDGET_RATIO:.0%}")

    model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
    evaluator = EvaluatorV2(model, tokenizer, model_config)
    logger.info(f"  placeholder_token_id (pad_token): {tokenizer.pad_token_id}")

    conditions = [
        ("front_real", ("position", "front")),
        ("middle_real", ("position", "middle")),
        ("end_real", ("position", "end")),
        ("front_placeholder", ("content", None)),
        ("none", ("position", "none")),
    ]

    all_results = []
    timestamp = get_timestamp()
    for name, mode in conditions:
        result = run_condition(evaluator, name, mode, TASKS, NUM_SAMPLES, BUDGET_RATIO, SEED, tokenizer)
        all_results.append(result)

        json_data = {
            "experiment": "table13_position_content_validation",
            "model": MODEL_KEY,
            "tasks": TASKS,
            "num_samples": NUM_SAMPLES,
            "budget_ratio": BUDGET_RATIO,
            "sink_size": SINK_SIZE,
            "results": all_results,
        }
        save_results_json(json_data, f"table13_position_content_{timestamp}.json")
        logger.info(f"[중간 저장 완료] {len(all_results)}/{len(conditions)} 조건")

    save_results_csv(all_results, f"table13_position_content_{timestamp}.csv")
    logger.info(f"\n완료: {len(all_results)}개 조건")
    for r in all_results:
        logger.info(f"  {r['condition']:<20} avg_score={r['avg_score']}")


if __name__ == "__main__":
    main()
