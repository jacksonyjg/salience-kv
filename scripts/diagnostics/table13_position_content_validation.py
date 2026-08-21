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
from core.collapse_metrics import is_collapsed, word_repetition_ratio, char_repetition_ratio
import core.kv_cache_hook as hook_module

import argparse

logger = logging.getLogger(__name__)

MODEL_KEY = "qwen3-4b"
TASKS = ["qmsum", "gov_report"]
NUM_SAMPLES = 10
BUDGET_RATIO = 0.20
SINK_SIZE = 4
SEED = 42

_ORIGINAL_SELECT_WITH_SINK = hook_module._select_with_sink
_ORIGINAL_TOKENIZE_PROMPT = ev2_module.tokenize_prompt


def make_patched_select_with_sink(slot: str, offset: int = 0):
    def patched(score, seq_len, budget, recent_w, sink_size, device):
        if slot == "none":
            sink_size = 0

        sink_size = max(min(sink_size, budget, seq_len), 0)
        recent_w = max(min(recent_w, max(budget - sink_size, 0), seq_len), 0)
        mid_budget = max(budget - sink_size - recent_w, 0)

        if sink_size == 0:
            sink_idx = torch.empty(0, dtype=torch.long, device=device)
        elif slot == "front":
            # offset=0이면 기존과 완전히 동일(하위호환). offset>0이면 "문서 내용 시작 지점"
            # 기준으로 sink 위치를 잡음 (절대 위치 0이 아니라 고정 지시어 템플릿 이후).
            start = min(offset, max(seq_len - sink_size, 0))
            sink_idx = torch.arange(start, start + sink_size, device=device)
        elif slot == "middle":
            mid_point = seq_len // 2
            start = max(mid_point - sink_size // 2, 0)
            sink_idx = torch.arange(start, start + sink_size, device=device)
        elif slot == "end":
            end_point = max(seq_len - recent_w - sink_size, 0)
            sink_idx = torch.arange(end_point, end_point + sink_size, device=device)
        elif slot == "random":
            # recent window를 제외한 범위에서 무작위 위치 sink_size개 선택.
            # 재현성: main()에서 torch.manual_seed(SEED)를 한 번 고정하므로,
            # 스크립트를 동일하게 재실행하면 호출 순서가 같아 동일 결과가 나옴.
            candidate_range = max(seq_len - recent_w, 0)
            if candidate_range > 0:
                k = min(sink_size, candidate_range)
                perm = torch.randperm(candidate_range, device=device)[:k]
                sink_idx = torch.sort(perm).values
            else:
                sink_idx = torch.empty(0, dtype=torch.long, device=device)
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


def make_placeholder_tokenize_prompt(sink_size: int, placeholder_token_id: int, offset: int = 0):
    def patched(prompt, tokenizer, model_key, max_input_length, device):
        inputs = _ORIGINAL_TOKENIZE_PROMPT(prompt, tokenizer, model_key, max_input_length, device)
        input_ids = inputs["input_ids"].clone()
        seq_len = input_ids.shape[1]
        start = min(offset, max(seq_len - sink_size, 0))
        n = min(sink_size, seq_len - start)
        input_ids[:, start:start + n] = placeholder_token_id
        inputs["input_ids"] = input_ids
        return inputs
    return patched


def compute_content_start_offset(tokenizer, model_key, task_type="summarization"):
    """고정 지시어 템플릿(예: 'Please summarize the following document concisely.\\n\\nDocument:\\n')이
    몇 토큰인지 계산 — 실제 문서 내용이 시작하는 위치. make_prompt의 내부 문자열을 하드코딩하지 않고,
    실제 context와 마커 context 두 프롬프트를 각각 토큰화해서 공통 접두부 길이를 재는 방식으로
    강건하게 계산(make_prompt 구현이 바뀌어도 자동으로 맞음)."""
    from core.model_loader import make_prompt
    real_context = "This is a placeholder real-looking document body used only to diverge from the marker."
    marker_context = "\uE000\uE000\uE000\uE000 MARKERMARKERMARKER \uE000\uE000\uE000\uE000"
    dummy_question = ""  # summarization 템플릿은 question 미사용
    p1 = make_prompt(model_key, tokenizer, real_context, dummy_question, task_type)
    p2 = make_prompt(model_key, tokenizer, marker_context, dummy_question, task_type)
    ids1 = tokenizer(p1, add_special_tokens=False)["input_ids"]
    ids2 = tokenizer(p2, add_special_tokens=False)["input_ids"]
    offset = 0
    for a, b in zip(ids1, ids2):
        if a != b:
            break
        offset += 1
    return offset


def run_condition(evaluator, condition_name, mode, tasks, num_samples, budget, seed, tokenizer, invert_norm,
                   sink_size_override=None, content_offset=0):
    logger.info(f"\n{'='*60}")
    logger.info(f"Condition: {condition_name} (mode={mode}, content_offset={content_offset})")
    logger.info(f"{'='*60}")

    kind, slot = mode
    effective_sink_size = sink_size_override if sink_size_override is not None else SINK_SIZE
    if kind == "position":
        hook_module._select_with_sink = make_patched_select_with_sink(slot, offset=content_offset)
        ev2_module.tokenize_prompt = _ORIGINAL_TOKENIZE_PROMPT
        sink_size_kwarg = effective_sink_size if slot != "none" else 0
    elif kind == "content":
        hook_module._select_with_sink = _ORIGINAL_SELECT_WITH_SINK
        if slot == "newline":
            # StreamingLLM 원 논문(Xiao et al.)의 실제 실험과 동일 — "\n" 토큰으로 치환
            newline_ids = tokenizer.encode("\n", add_special_tokens=False)
            placeholder_id = newline_ids[0] if newline_ids else tokenizer.pad_token_id
        else:
            # 기존(pad_token, 특수 제어 토큰 <|endoftext|>) — chat template 구조 마커까지
            # 덮어쓰게 되어 "special-token poisoning" 가능성 있음, front_newline과 대조용으로 유지
            placeholder_id = tokenizer.pad_token_id
        ev2_module.tokenize_prompt = make_placeholder_tokenize_prompt(effective_sink_size, placeholder_id,
                                                                        offset=content_offset)
        sink_size_kwarg = effective_sink_size
    else:
        raise ValueError(mode)

    task_scores = {}
    all_sample_records = {}
    for task_name in tasks:
        samples = load_longbench_task(task_name, num_samples=num_samples, seed=seed)
        scores = []
        sample_records = []
        for i, sample in enumerate(samples):
            r = evaluator.evaluate_sample(
                sample, "ours", budget,
                measure_efficiency=False,
                method_kwargs={"sink_size": sink_size_kwarg, "invert_norm": invert_norm},
            )
            scores.append(r["score"])
            pred = r["prediction"]
            sample_records.append({
                "sample_idx": i,
                "score": r["score"],
                "prediction": pred,
                "word_rep": word_repetition_ratio(pred),
                "char_rep": char_repetition_ratio(pred),
                "collapsed": is_collapsed(pred),
            })
            if (i + 1) % 5 == 0:
                logger.info(f"  [{condition_name}] {task_name} [{i+1}/{len(samples)}] "
                            f"avg={sum(scores)/len(scores):.2f}")
        avg = sum(scores) / len(scores) if scores else 0.0
        task_scores[task_name] = round(avg, 2)
        all_sample_records[task_name] = sample_records
        logger.info(f"  → {task_name}: {avg:.2f}")

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0

    hook_module._select_with_sink = _ORIGINAL_SELECT_WITH_SINK
    ev2_module.tokenize_prompt = _ORIGINAL_TOKENIZE_PROMPT

    return {
        "condition": condition_name,
        "mode": f"{kind}:{slot}",
        "sink_size_used": sink_size_kwarg,
        "task_scores": task_scores,
        "avg_score": round(avg_score, 2),
        "sample_records": all_sample_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--invert_norm", action="store_true",
                        help="key-norm 선택 방향을 corrected(low-norm 우선, Devoto et al. 방향)로 전환.")
    args = parser.parse_args()

    log_dir = "logs/v3_verified" if args.invert_norm else "logs/v2_verified"
    results_dir = "results/v3_verified" if args.invert_norm else "results/v2_verified"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/table13_{get_timestamp()}.log"),
        ],
        force=True,
    )

    import core.results_manager as rm
    rm.RESULTS_DIR = results_dir

    logger.info("TABLE X: Sink Position/Content Controlled Validation")
    logger.info(f"  Model: {MODEL_KEY} | Tasks: {TASKS} | Samples: {args.num_samples}/task | Budget: {BUDGET_RATIO:.0%}")
    logger.info(f"  Key-norm direction: {'CORRECTED' if args.invert_norm else 'legacy'} | "
                f"Results dir: {results_dir} | Log dir: {log_dir}")

    import torch as _torch
    _torch.manual_seed(SEED)  # random 슬롯 조건의 재현성 확보 (동일 실행 순서 시 동일 결과)

    model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
    evaluator = EvaluatorV2(model, tokenizer, model_config)
    logger.info(f"  placeholder_token_id (pad_token): {tokenizer.pad_token_id}")

    content_offset = compute_content_start_offset(tokenizer, MODEL_KEY, task_type="summarization")
    logger.info(f"  문서 내용 시작 오프셋(고정 지시어 템플릿 길이): {content_offset} 토큰 "
                f"(qmsum/gov_report 둘 다 task_type=summarization, 동일 템플릿 공유)")

    # (이름, mode, sink_size_override) — override=None이면 기본 SINK_SIZE(4) 사용
    # (이름, mode, sink_size_override, use_content_offset) — override=None이면 기본 SINK_SIZE(4) 사용
    # use_content_offset=True면 절대 위치 0이 아니라 "문서 내용이 실제로 시작하는 지점" 기준으로
    # sink을 잡음 — 8/8 세션 원본 설계(real_content_sink vs meaningless_prefix_sink) 복원.
    conditions = [
        ("front_real", ("position", "front"), None, False),
        ("middle_real", ("position", "middle"), None, False),
        ("end_real", ("position", "end"), None, False),
        ("random_real", ("position", "random"), None, False),   # 신규: sink=4를 무작위 위치에 (front_real과 대조)
        ("front_placeholder", ("content", "pad"), None, False),   # slot 명시(하위호환 동일 동작)
        ("front_newline", ("content", "newline"), None, False),  # 신규: StreamingLLM과 동일 조건(GPT 제안)
        ("none", ("position", "none"), None, False),
        ("front_1", ("position", "front"), 1, False),            # 신규: 위치 0 단 하나만 고정
        ("random_1", ("position", "random"), 1, False),          # 신규: 무작위 위치 단 하나만 고정 (front_1과 대조)
        # 신규(2026-08-21): 문서 내용 시작 지점 기준 — 고정 지시어 템플릿은 안 건드림
        ("front_real_content", ("position", "front"), None, True),
        ("front_placeholder_content", ("content", "pad"), None, True),
        ("front_newline_content", ("content", "newline"), None, True),
    ]

    all_results = []
    timestamp = get_timestamp()
    for name, mode, sink_override, use_content_offset in conditions:
        offset = content_offset if use_content_offset else 0
        result = run_condition(evaluator, name, mode, TASKS, args.num_samples, BUDGET_RATIO, SEED,
                                tokenizer, args.invert_norm, sink_size_override=sink_override,
                                content_offset=offset)
        all_results.append(result)

        json_data = {
            "experiment": "table13_position_content_validation",
            "model": MODEL_KEY,
            "tasks": TASKS,
            "num_samples": args.num_samples,
            "budget_ratio": BUDGET_RATIO,
            "sink_size": SINK_SIZE,
            "invert_norm": args.invert_norm,
            "content_offset": content_offset,
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
