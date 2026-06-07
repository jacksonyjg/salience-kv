# core/__init__.py
# 지연 임포트: torch 없는 환경에서도 metrics/results_manager 사용 가능
from core.metrics import compute_score, aggregate_scores
from core.results_manager import save_results_csv, save_results_json, print_result_table
