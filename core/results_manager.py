"""
core/results_manager.py
========================
실험 결과 CSV 자동 저장 및 로드.
"""

import os
import csv
import json
import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results_csv(
    results: List[Dict],
    filename: str,
    results_dir: str = RESULTS_DIR,
) -> str:
    """결과 딕셔너리 리스트를 CSV로 저장."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    if not results:
        logger.warning("No results to save")
        return filepath
    
    fieldnames = list(results[0].keys())
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def save_results_json(
    results: Dict,
    filename: str,
    results_dir: str = RESULTS_DIR,
) -> str:
    """결과를 JSON으로 저장."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON saved to {filepath}")
    return filepath


def format_table_row(
    method: str,
    task_scores: Dict[str, float],
    task_order: List[str],
    avg_score: float,
    mem_reduction: float = 0.0,
    ttft_ms: float = 0.0,
    throughput: float = 0.0,
) -> Dict:
    """논문 Table 형식으로 행 딕셔너리 생성."""
    row = {"Method": method}
    for task in task_order:
        row[task] = round(task_scores.get(task, 0.0), 1)
    row["Avg"] = round(avg_score, 1)
    row["Mem_Reduction_%"] = round(mem_reduction, 1)
    row["TTFT_ms"] = round(ttft_ms, 1)
    row["Throughput_tps"] = round(throughput, 1)
    return row


def print_result_table(rows: List[Dict], title: str = "Results"):
    """결과를 터미널 테이블 형식으로 출력."""
    if not rows:
        return
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    # 헤더
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    
    header_str = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_str)
    print("-" * len(header_str))
    
    for row in rows:
        row_str = " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)
        print(row_str)
    
    print(f"{'='*80}\n")
