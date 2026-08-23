import json

files = [
    ("signal_ablation sink0", "results/v3_verified/exp7_signal_ablation_qwen3-4b_sink0_20260823_043358.json"),
    ("signal_ablation sink4", "results/v3_verified/exp7_signal_ablation_qwen3-4b_sink4_20260823_050944.json"),
    ("extra_signals sink0", "results/v3_verified/exp7_extra_signals_qwen3-4b_20260823_054206.json"),
    ("extra_signals sink4", "results/v3_verified/exp7_extra_signals_qwen3-4b_20260823_055407.json"),
    ("budget_sensitivity", "results/v3_verified/exp8_budget_sensitivity_qwen3-4b_20260823_060549.json"),
]

for label, path in files:
    d = json.load(open(path))
    print(f"=== {label} ===")
    print(f"설정 개수: {len(d['results'])}")
    for r in d["results"][:3]:
        # 실제 키 이름 확인(method/variant/signal 등 스크립트마다 다를 수 있음)
        method_key = r.get("method") or r.get("variant") or r.get("signal") or r.get("name") or "?"
        print(f"  키 목록: {list(r.keys())}")
        recs = r["sample_records"].get("qmsum", [])
        print(f"  {method_key}: qmsum 샘플 개수={len(recs)}")
    print()
