"""
core/dataset_loader.py
=======================
LongBench 데이터셋 로드 및 태스크별 프롬프트 구성.

LongBench HuggingFace 데이터셋: THUDM/LongBench
각 태스크별 올바른 입력 필드(context, input)와 메트릭(F1/ROUGE-L) 매핑.

[2026-08-12] 재현성 확보를 위해 revision을 고정(E1 결정, Plan v3 참고).
이전엔 revision 미고정으로 파드마다 다른 스냅샷을 로드해 세션 간 절대
수치가 흔들리는 문제가 있었음(2026-08-08 vs 08-09 파드에서 방법 무관
일괄 하락 관찰). 이후 모든 Phase 2/3 실험은 아래 고정된 revision을 사용.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset

logger = logging.getLogger(__name__)

# 재현성 고정: 2026-08-12 기준 THUDM/LongBench 최신 commit sha
# (huggingface_hub.HfApi().dataset_info("THUDM/LongBench").sha 로 확인,
#  원본 데이터셋 마지막 수정일 2024-12-18)
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"

# ─────────────────────────────────────────────
# 태스크 설정
# ─────────────────────────────────────────────
LONGBENCH_TASKS = {
    "narrativeqa": {
        "dataset_name": "THUDM/LongBench",
        "subset": "narrativeqa",
        "metric": "f1",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": "input",
        "answer_field": "answers",
        "task_type": "qa",
        "description": "장문 서사 이해 및 독해",
    },
    "qasper": {
        "dataset_name": "THUDM/LongBench",
        "subset": "qasper",
        "metric": "f1",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": "input",
        "answer_field": "answers",
        "task_type": "qa",
        "description": "과학 문서 QA",
    },
    "multifieldqa_en": {
        "dataset_name": "THUDM/LongBench",
        "subset": "multifieldqa_en",
        "metric": "f1",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": "input",
        "answer_field": "answers",
        "task_type": "qa",
        "description": "다중 도메인 사실 QA",
    },
    "hotpotqa": {
        "dataset_name": "THUDM/LongBench",
        "subset": "hotpotqa",
        "metric": "f1",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": "input",
        "answer_field": "answers",
        "task_type": "qa",
        "description": "다중 홉 추론",
    },
    "2wikimqa": {
        "dataset_name": "THUDM/LongBench",
        "subset": "2wikimqa",
        "metric": "f1",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": "input",
        "answer_field": "answers",
        "task_type": "qa",
        "description": "위키피디아 기반 다중 홉 QA",
    },
    "gov_report": {
        "dataset_name": "THUDM/LongBench",
        "subset": "gov_report",
        "metric": "rouge_l",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": None,     # 요약 태스크: 질문 없음
        "answer_field": "answers",
        "task_type": "summarization",
        "description": "정부 보고서 요약",
    },
    "qmsum": {
        "dataset_name": "THUDM/LongBench",
        "subset": "qmsum",
        "metric": "rouge_l",
        "max_new_tokens": 512,
        "context_field": "context",
        "question_field": "input",  # 쿼리 기반 요약
        "answer_field": "answers",
        "task_type": "summarization",
        "description": "회의록 요약",
    },
}

# 실험계획서의 태스크 이름 → LONGBENCH_TASKS 키 매핑
TASK_NAME_MAP = {
    "narrativeqa": "narrativeqa",
    "qasper": "qasper",
    "multifieldqa_en": "multifieldqa_en",
    "hotpotqa": "hotpotqa",
    "2wikimqa": "2wikimqa",
    "gov_report": "gov_report",
    "qmsum": "qmsum",
}

ALL_TASKS = list(LONGBENCH_TASKS.keys())


def load_longbench_task(
    task_name: str,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    LongBench 태스크 데이터 로드.
    
    Args:
        task_name: LONGBENCH_TASKS 키
        num_samples: None이면 전체, 정수면 해당 수만큼 샘플링
        seed: 재현성을 위한 랜덤 시드
    
    Returns:
        샘플 딕셔너리 리스트 [{"context", "question", "answers", "task_name", ...}]
    """
    task_cfg = LONGBENCH_TASKS[task_name]
    
    logger.info(f"Loading LongBench/{task_name} ...")
    
    try:
        dataset = load_dataset(
            task_cfg["dataset_name"],
            task_cfg["subset"],
            split="test",
            trust_remote_code=True,
            revision=LONGBENCH_REVISION,
        )
    except Exception as e:
        logger.error(f"Failed to load {task_name}: {e}")
        raise
    
    # 샘플 수 제한
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    
    samples = []
    for item in dataset:
        context = item.get(task_cfg["context_field"], "")
        
        # 컨텍스트 최대 길이 제한: 진짜 truncation은 tokenize_prompt()의
        # max_input_length(토큰 기준, 16000)에서 처리. 여기서는 극단적으로
        # 긴 문서의 불필요한 토크나이즈 비용만 줄이는 안전망으로 40000 단어 사용.
        words = context.split()
        if len(words) > 40000:
            context = " ".join(words[:30000]) + " ... " + " ".join(words[-10000:])
        
        # 질문 필드 처리
        if task_cfg["question_field"] is not None:
            question = item.get(task_cfg["question_field"], "")
        else:
            question = ""
        
        # 정답 처리 (리스트 또는 문자열)
        raw_answers = item.get(task_cfg["answer_field"], [])
        if isinstance(raw_answers, str):
            answers = [raw_answers]
        elif isinstance(raw_answers, list):
            answers = raw_answers
        else:
            answers = [str(raw_answers)]
        
        # 빈 정답 필터링
        answers = [a for a in answers if a.strip()]
        if not answers:
            continue
        
        samples.append({
            "context": context,
            "question": question,
            "answers": answers,
            "task_name": task_name,
            "task_type": task_cfg["task_type"],
            "metric": task_cfg["metric"],
            "max_new_tokens": task_cfg["max_new_tokens"],
        })
    
    logger.info(f"Loaded {len(samples)} samples from {task_name}")
    return samples


def build_prompt_for_sample(
    sample: Dict,
    model_key: str,
    tokenizer,
) -> str:
    """
    샘플에서 모델별 프롬프트 생성.
    make_prompt를 호출하되 태스크 타입에 따라 적절히 처리.
    """
    from core.model_loader import make_prompt
    
    prompt = make_prompt(
        model_key=model_key,
        tokenizer=tokenizer,
        context=sample["context"],
        question=sample["question"],
        task_type=sample["task_type"],
    )
    return prompt
