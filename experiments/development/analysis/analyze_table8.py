import json

RESULTS_FILE = "results/v2_verified/exp8_budget_sensitivity_qwen3-4b_20260814_075410.json"

d = json.load(open(RESULTS_FILE))
results = d["results"]

methods = sorted(set(r["method"] for r in results),
                  key=lambda m: [r["method"] for r in results].index(m))
budgets = sorted(set(r["budget_ratio"] for r in results))

print(f"{'Method':<25}" + "".join(f"{'B='+str(int(b*100))+'%':>10}" for b in budgets))
for method in methods:
    row = f"{method:<25}"
    for b in budgets:
        match = [r for r in results if r["method"] == method and r["budget_ratio"] == b]
        val = f"{match[0]['avg_score']:.2f}" if match else "N/A"
        row += f"{val:>10}"
    print(row)

print("\n--- 태스크별 (qmsum) ---")
print(f"{'Method':<25}" + "".join(f"{'B='+str(int(b*100))+'%':>10}" for b in budgets))
for method in methods:
    row = f"{method:<25}"
    for b in budgets:
        match = [r for r in results if r["method"] == method and r["budget_ratio"] == b]
        val = f"{match[0]['task_scores']['qmsum']:.2f}" if match else "N/A"
        row += f"{val:>10}"
    print(row)

print("\n--- 태스크별 (hotpotqa) ---")
print(f"{'Method':<25}" + "".join(f"{'B='+str(int(b*100))+'%':>10}" for b in budgets))
for method in methods:
    row = f"{method:<25}"
    for b in budgets:
        match = [r for r in results if r["method"] == method and r["budget_ratio"] == b]
        val = f"{match[0]['task_scores']['hotpotqa']:.2f}" if match else "N/A"
        row += f"{val:>10}"
    print(row)
