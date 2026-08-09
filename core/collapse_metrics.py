"""
core/collapse_metrics.py
=========================
반복 붕괴(repetition collapse) 판정 지표.

0808 세션에서 확정된 정의를 재구현:
  붕괴 판정 = word_rep > 0.3 OR char_rep > 0.7

반복률 공식은 seq-rep-n (Welleck et al., neural text degeneration 계열 지표):
  rep-n(text) = 1 - (고유 n-gram 수 / 전체 n-gram 수)

  - word_rep: 단어 단위 trigram(n=3) 기준
  - char_rep: 문자 단위 5-gram(n=5) 기준

주의: 이 모듈은 원본 0808 세션 코드가 git에 커밋되지 않아 소실된 것을
2026-08-09 세션에서 정의 문서(메모리 기록)에 기반해 재구현한 것이다.
"""

import re
from typing import List, Tuple


def _word_tokens(text: str) -> List[str]:
    return text.split()


def _ngrams(tokens: List, n: int) -> List[Tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def word_repetition_ratio(text: str, n: int = 3) -> float:
    tokens = _word_tokens(text)
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    unique = len(set(grams))
    total = len(grams)
    return 1.0 - (unique / total)


def char_repetition_ratio(text: str, n: int = 5) -> float:
    if len(text) < n:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    unique = len(set(grams))
    total = len(grams)
    return 1.0 - (unique / total)


def is_collapsed(text: str, word_thresh: float = 0.3, char_thresh: float = 0.7) -> bool:
    return (word_repetition_ratio(text) > word_thresh) or (char_repetition_ratio(text) > char_thresh)


def collapse_report(text: str, word_thresh: float = 0.3, char_thresh: float = 0.7) -> dict:
    wr = word_repetition_ratio(text)
    cr = char_repetition_ratio(text)
    return {
        "word_rep": wr,
        "char_rep": cr,
        "collapsed": (wr > word_thresh) or (cr > char_thresh),
    }
