"""
core/metrics.py
================
평가 메트릭 계산: Token-level F1 (QA), ROUGE-L (요약).

LongBench 공식 평가 방식 재현.
"""

import re
import string
from collections import Counter
from typing import List, Union


# ─────────────────────────────────────────────
# 텍스트 정규화
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """소문자 변환, 구두점/관사/공백 정규화."""
    text = text.lower()
    # 구두점 제거
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 관사 제거
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 연속 공백 정리
    text = " ".join(text.split())
    return text


def get_tokens(text: str) -> List[str]:
    return normalize_text(text).split()


# ─────────────────────────────────────────────
# F1 Score (Token-level)
# ─────────────────────────────────────────────

def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    토큰 수준 F1 점수 계산.
    여러 정답 중 최대값 반환 (LongBench 공식 방식).
    
    Returns:
        F1 score (0~100)
    """
    best_f1 = 0.0
    pred_tokens = get_tokens(prediction)
    
    for gt in ground_truths:
        gt_tokens = get_tokens(gt)
        
        if not pred_tokens and not gt_tokens:
            f1 = 1.0
        elif not pred_tokens or not gt_tokens:
            f1 = 0.0
        else:
            pred_counter = Counter(pred_tokens)
            gt_counter = Counter(gt_tokens)
            
            # 공통 토큰 수
            common = sum((pred_counter & gt_counter).values())
            
            precision = common / len(pred_tokens)
            recall = common / len(gt_tokens)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        
        best_f1 = max(best_f1, f1)
    
    return best_f1 * 100.0  # 0~100 스케일


# ─────────────────────────────────────────────
# ROUGE-L Score
# ─────────────────────────────────────────────

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Longest Common Subsequence 길이."""
    m, n = len(x), len(y)
    # 메모리 효율: 2행만 유지
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
    """
    ROUGE-L F1 점수 계산.
    여러 정답 중 최대값 반환.
    
    Returns:
        ROUGE-L F1 score (0~100)
    """
    best_rouge_l = 0.0
    pred_tokens = get_tokens(prediction)
    
    for gt in ground_truths:
        gt_tokens = get_tokens(gt)
        
        if not pred_tokens and not gt_tokens:
            score = 1.0
        elif not pred_tokens or not gt_tokens:
            score = 0.0
        else:
            lcs_len = _lcs_length(pred_tokens, gt_tokens)
            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(gt_tokens)
            
            if precision + recall == 0:
                score = 0.0
            else:
                score = 2 * precision * recall / (precision + recall)
        
        best_rouge_l = max(best_rouge_l, score)
    
    return best_rouge_l * 100.0  # 0~100 스케일


# ─────────────────────────────────────────────
# 통합 메트릭 함수
# ─────────────────────────────────────────────

def compute_score(
    prediction: str,
    ground_truths: List[str],
    metric: str,
) -> float:
    """
    태스크 메트릭에 따라 점수 계산.
    
    Args:
        prediction: 모델 생성 텍스트
        ground_truths: 정답 리스트
        metric: "f1" | "rouge_l"
    
    Returns:
        점수 (0~100)
    """
    if metric == "f1":
        return compute_f1(prediction, ground_truths)
    elif metric == "rouge_l":
        return compute_rouge_l(prediction, ground_truths)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def aggregate_scores(scores: List[float]) -> float:
    """점수 리스트의 평균 반환."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
