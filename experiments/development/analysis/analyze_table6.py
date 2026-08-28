"""
TABLE VI (Sink Intervention) 결과 분석
budget20 + budget80 JSON을 합쳐서 method x sink크기 요약표 생성.
읽기 전용 - 결과 파일만 읽음, 실험에 영향 없음.
"""
import json
import re
from pathlib import Path

RESULTS_DIR = Path("results/v2_verified")
FILES = {
    0.2: "exp6_sink_intervention_qwen3-4b_budget20_20260813_130827.json",
    0.8: "exp6_sink_intervention_qwen3-4b_budget80_20260813_200526.json",
}

def parse_method(name):
    m = re.match(r"(.+)_m(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return name, None

all_rows = []
for budget, fname in FILES.items():
    path = RESULTS_DIR / fname
    d = json.load(open(path))
    for r in d["results"]:
        method, sink = parse_method(r["method"])
        all_rows.append({
            "budget": budget, "method": method, "sink": sink,
            "avg_score": r["avg_score"],
            "avg_collapse_pct": r["avg_collapse_rate_pct"],
            "avg_collapse_frac": r["avg_collapse_frac"],
            "qmsum_score": r["task_scores"]["qmsum"],
            "gov_report_score": r["task_scores"]["gov_report"],
            "qmsum_collapse": r["task_collapse_rates"]["qmsum"],
            "gov_report_collapse": r["task_collapse_rates"]["gov_report"],
        })

methods = sorted(set(r["method"] for r in all_rows))
sinks = sorted(set(r["sink"] for r in all_rows if r["sink"] is not None))
budgets = sorted(FILES.keys())

for budget in budgets:
    print(f"\n{'='*90}")
    print(f"BUDGET = {int(budget*100)}%")
    print(f"{'='*90}")
    print(f"\n--- 평균 점수 (avg_score) ---")
    header = f"{'Method':<14}" + "".join(f"{'sink='+str(s):>10}" for s in sinks)
    print(header)
    for method in methods:
        row = f"{method:<14}"
        for s in sinks:
            match = [r for r in all_rows if r["budget"]==budget and r["method"]==method and r["sink"]==s]
            val = f"{match[0]['avg_score']:.2f}" if match else "N/A"
            row += f"{val:>10}"
        print(row)
    print(f"\n--- 붕괴율 (%) ---")
    print(header)
    for method in methods:
        row = f"{method:<14}"
        for s in sinks:
            match = [r for r in all_rows if r["budget"]==budget and r["method"]==method and r["sink"]==s]
            val = f"{match[0]['avg_collapse_pct']:.1f}" if match else "N/A"
            row += f"{val:>10}"
        print(row)

print(f"\n{'='*90}")
print("SINK=0 -> SINK=4 개선폭 (핵심 가설 검증)")
print(f"{'='*90}")
print(f"{'Budget':<8}{'Method':<14}{'sink0_score':>12}{'sink4_score':>12}{'Δscore':>10}{'sink0_collapse%':>16}{'sink4_collapse%':>16}")
for budget in budgets:
    for method in methods:
        r0 = [r for r in all_rows if r["budget"]==budget and r["method"]==method and r["sink"]==0]
        r4 = [r for r in all_rows if r["budget"]==budget and r["method"]==method and r["sink"]==4]
        if r0 and r4:
            r0, r4 = r0[0], r4[0]
            delta = r4["avg_score"] - r0["avg_score"]
            print(f"{int(budget*100):<8}{method:<14}{r0['avg_score']:>12.2f}{r4['avg_score']:>12.2f}{delta:>10.2f}{r0['avg_collapse_pct']:>16.1f}{r4['avg_collapse_pct']:>16.1f}")
