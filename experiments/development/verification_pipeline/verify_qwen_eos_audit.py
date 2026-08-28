"""
GPT 지적 반영: multi-EOS patch(evaluator_v2.py)가 기존 TABLE II Qwen collapse 샘플들에
실제 영향을 줬는지 targeted re-run으로 확인.
전면 재실행 대신, H2O/PyramidKV-adapted/AdaKV-adapted의 collapse=True였던 샘플 중
방법당 15개(총 45개)만 뽑아서 patched evaluator로 재실행 → old prediction과 직접 비교.
"""
import sys, json
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2

with open("/workspace/kv-cache-exp/audit_targets.json") as f:
    audit_list = json.load(f)

method_key_map = {
    "H2O": "h2o",
    "PyramidKV-adapted": "pyramidkv",
    "AdaKV-adapted": "adakv",
}

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev = EvaluatorV2(model, tok, cfg)

# task별로 샘플을 캐싱해서 반복 로드 방지
task_cache = {}

mismatch_count = 0
eos_151643_count = 0
results_log = []

for i, item in enumerate(audit_list):
    task = item["task"]
    idx = item["sample_idx"]
    method_label = item["method"]
    method_key = method_key_map[method_label]

    if task not in task_cache:
        task_cache[task] = load_longbench_task(task, num_samples=30, seed=42)
    sample = task_cache[task][idx]

    r = ev.evaluate_sample(sample, method_key, 0.20, method_kwargs={"invert_norm": True})

    if r.get("error"):
        print(f"[{i+1}/45] {method_label} {task} idx={idx}: 재실행 실패 - {r['error']}")
        continue

    new_pred = r["prediction"]
    old_pred = item["old_prediction"]
    match = (new_pred == old_pred)
    final_tok = r.get("final_token_id")
    hit_151643 = (final_tok == 151643)

    if not match:
        mismatch_count += 1
    if hit_151643:
        eos_151643_count += 1

    status = "✅동일" if match else "❌다름"
    eos_flag = " [151643 발생!]" if hit_151643 else ""
    print(f"[{i+1}/45] {method_label:20s} {task:12s} idx={idx:2d}  {status}  "
          f"final_token={final_tok}  terminated_by_eos={r.get('terminated_by_eos')}  "
          f"hit_max={r.get('hit_max_new_tokens')}{eos_flag}")

    results_log.append({
        "method": method_label, "task": task, "sample_idx": idx,
        "match": match, "final_token_id": final_tok,
        "terminated_by_eos": r.get("terminated_by_eos"),
        "hit_max_new_tokens": r.get("hit_max_new_tokens"),
        "old_prediction": old_pred, "new_prediction": new_pred,
    })

print(f"\n{'='*60}")
print(f"=== 최종 요약 (45개 중) ===")
print(f"prediction 불일치: {mismatch_count}개")
print(f"151643(<|endoftext|>)으로 종료: {eos_151643_count}개")
print(f"{'='*60}")

with open("/workspace/kv-cache-exp/results/v3_verified/qwen_eos_audit_table2.json", "w") as f:
    json.dump(results_log, f, indent=2, ensure_ascii=False)
print("저장 완료: results/v3_verified/qwen_eos_audit_table2.json")
