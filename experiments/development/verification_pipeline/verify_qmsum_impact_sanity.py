"""
QMSum corrected(LongBench 공식) prompt로 N=5 impact sanity check.
core/model_loader.py는 건드리지 않음 - evaluator_v2.EvaluatorV2.evaluate_sample 안에서
쓰는 make_prompt를 이 스크립트 실행 중에만 monkeypatch.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import core.model_loader as ml
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2
from core.collapse_metrics import is_collapsed, word_repetition_ratio
import core.evaluator_v2 as ev2_module

_orig_make_prompt = ml.make_prompt

def make_prompt_corrected_qmsum(model_key, tokenizer, context, question, task_type="qa"):
    """QMSum(question 존재)만 LongBench 공식 형식으로 교체, 나머지는 원본 그대로."""
    if task_type == "summarization" and question:
        user_content = (
            "You are given a meeting transcript and a query containing a question or instruction. "
            "Answer the query in one or more sentences.\n\n"
            f"Transcript:\n{context}\n\n"
            "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
            f"Query: {question}\n"
            "Answer:"
        )
        fmt = ml.MODEL_CONFIGS[model_key]["prompt_format"]
        if fmt == "qwen3_chat":
            messages = [{"role": "user", "content": user_content}]
            return tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        return user_content
    return _orig_make_prompt(model_key, tokenizer, context, question, task_type)


model, tokenizer, model_cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tokenizer, model_cfg)

samples = load_longbench_task("qmsum", num_samples=5, seed=42)

METHODS = [
    ("fullkv", "FullKV", {}),
    ("h2o", "H2O-adapted", {"invert_norm": True}),
    ("snapkv", "SnapKV-adapted", {"invert_norm": True}),
    ("ours", "SalienceKV-Sink4", {"invert_norm": True, "sink_size": 4}),
]

results = {}
for method_name, label, kwargs in METHODS:
    print(f"\n{'='*60}\n{label} (kwargs={kwargs})\n{'='*60}")

    # legacy(현재 그대로)
    ml.make_prompt = _orig_make_prompt
    ev2_module.make_prompt = _orig_make_prompt
    legacy_records = []
    for i, s in enumerate(samples):
        r = ev.evaluate_sample(s, method_name, 0.20, method_kwargs=kwargs)
        legacy_records.append({"idx": i, "score": r["score"], "collapsed": is_collapsed(r["prediction"]),
                                "prediction": r["prediction"]})

    # corrected(쿼리 포함)
    ml.make_prompt = make_prompt_corrected_qmsum
    ev2_module.make_prompt = make_prompt_corrected_qmsum
    corrected_records = []
    for i, s in enumerate(samples):
        r = ev.evaluate_sample(s, method_name, 0.20, method_kwargs=kwargs)
        corrected_records.append({"idx": i, "score": r["score"], "collapsed": is_collapsed(r["prediction"]),
                                   "prediction": r["prediction"]})

    ml.make_prompt = _orig_make_prompt
    ev2_module.make_prompt = _orig_make_prompt

    results[label] = {"legacy": legacy_records, "corrected": corrected_records}

    legacy_avg = sum(x["score"] for x in legacy_records) / len(legacy_records)
    corrected_avg = sum(x["score"] for x in corrected_records) / len(corrected_records)
    legacy_collapse = sum(1 for x in legacy_records if x["collapsed"])
    corrected_collapse = sum(1 for x in corrected_records if x["collapsed"])

    print(f"legacy:    avg_score={legacy_avg:.2f}  collapse={legacy_collapse}/{len(legacy_records)}")
    print(f"corrected: avg_score={corrected_avg:.2f}  collapse={corrected_collapse}/{len(corrected_records)}")
    for i in range(len(samples)):
        print(f"  [{i}] legacy: {legacy_records[i]['prediction'][:100]!r}")
        print(f"      corrected: {corrected_records[i]['prediction'][:100]!r}")

print(f"\n\n{'='*60}\n=== 전체 요약 ===\n{'='*60}")
for label in results:
    lr = results[label]["legacy"]
    cr = results[label]["corrected"]
    la = sum(x["score"] for x in lr) / len(lr)
    ca = sum(x["score"] for x in cr) / len(cr)
    lc = sum(1 for x in lr if x["collapsed"])
    cc = sum(1 for x in cr if x["collapsed"])
    print(f"{label:20s} legacy={la:6.2f}(collapse {lc}/5)  corrected={ca:6.2f}(collapse {cc}/5)  diff={ca-la:+.2f}")
