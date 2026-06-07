# KV Cache 실험 코드

**논문:** Adaptive Layer-wise Hybrid KV Cache Management for Resource-Constrained Small Language Models

---

## 디렉토리 구조

```
kv_cache_exp/
├── core/                        ← 공통 코드 (수정 불필요)
│   ├── model_loader.py            모델 로드, 프롬프트 생성
│   ├── dataset_loader.py          LongBench 7개 태스크
│   ├── metrics.py                 F1, ROUGE-L
│   ├── kv_base.py                 KV 캐시 기반 클래스
│   ├── kv_methods.py              7개 KV 방법
│   ├── evaluator.py               평가 엔진
│   └── results_manager.py         CSV/JSON 저장
│
├── experiments/                 ← 실험별 실행 파일
│   ├── sanity_check.py            환경 검증 (실험 전 필수)
│   ├── exp1_main_results.py       주요 성능 비교
│   ├── exp2_ablation_score.py     Ablation - Score 성분
│   ├── exp3_ablation_allocation.py Ablation - 레이어 할당
│   ├── exp4_budget_sensitivity.py  예산 민감도
│   ├── exp5_hyperparam_sensitivity.py 하이퍼파라미터 민감도
│   └── exp6_overhead.py           계산 오버헤드
│
├── notebooks/                   ← 환경 확인 및 결과 분석
│   ├── 01_환경확인.ipynb
│   ├── 02_결과분석.ipynb
│   └── 03_그래프작성.ipynb
│
├── configs/                     ← 하이퍼파라미터 설정
│   ├── default.yaml               공통 기본값
│   ├── qwen3-4b.yaml              주 모델 설정
│   ├── phi-3-mini.yaml            교차 검증 모델
│   └── gemma-2-2b.yaml            교차 검증 모델
│
├── scripts/                     ← RunPod 실행 스크립트
│   ├── run_all.sh                 전체 실험 순차 실행
│   ├── run_baselines.sh           실험 1: 베이스라인 비교
│   ├── run_ablation.sh            실험 2+3: Ablation
│   └── run_sensitivity.sh         실험 4+5: 민감도 분석
│
├── logs/                        ← 실험 로그 자동 저장
├── results/                     ← CSV 결과 자동 저장
│   ├── longbench/
│   ├── latency/
│   └── memory/
├── figures/                     ← 그래프 저장
├── setup.sh                     ← RunPod 초기 환경 설정
└── requirements.txt
```

---

## RunPod 실행 순서

### 1. 초기 환경 설정 (Pod 생성 후 최초 1회)
```bash
cd /workspace
git clone <repo_url> kv_cache_exp
cd kv_cache_exp
bash setup.sh
```

### 2. 환경 확인 (실험 전 필수)
```bash
python experiments/sanity_check.py --model qwen3-4b --full_check
```

### 3. 빠른 검증 (샘플 20개, ~1시간)
```bash
bash scripts/run_all.sh --quick
```

### 4. 전체 실험 (tmux 권장)
```bash
tmux new -s kvcache
bash scripts/run_all.sh
# Ctrl+B → D  (백그라운드로 분리)
# tmux attach -t kvcache  (재접속)
```

### 5. 개별 실험 실행
```bash
# 실험 1: 베이스라인 비교
bash scripts/run_baselines.sh --num_samples 20

# 실험 2+3: Ablation
bash scripts/run_ablation.sh --num_samples 20

# 실험 4+5: 민감도 분석
bash scripts/run_sensitivity.sh --num_samples 20

# 실험 6: 오버헤드
python experiments/exp6_overhead.py --model qwen3-4b
```

---

## 로컬 모델/데이터셋 전환 방법

현재는 HuggingFace Hub에서 자동 다운로드합니다.
로컬 경로로 전환 시 `# LOCAL:` 주석 부분을 수정하세요.

**파일 위치:**
- 모델 경로: `core/model_loader.py` → `MODEL_CONFIGS` 안의 `hf_name`
- 데이터셋 경로: `core/dataset_loader.py` → `load_longbench_task()` 안의 주석
- yaml 경로: `configs/qwen3-4b.yaml` → `model_source: local` 로 변경

**다운로드 명령어:**
```bash
# 모델
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir /workspace/models/Qwen3-4B

# 데이터셋
huggingface-cli download THUDM/LongBench \
    --repo-type dataset \
    --local-dir /workspace/datasets/longbench
```

---

## 결과 파일

실험 완료 후 `results/` 에 자동 저장:
```
results/
├── exp1_qwen3-4b_full_YYYYMMDD.csv    ← 논문 Table I
├── exp2_score_ablation_*.csv           ← 논문 Table III
├── exp3_allocation_*.csv               ← 논문 Table IV
├── exp4_budget_sensitivity_*.csv       ← 논문 Table V
└── exp5_*.csv                          ← 논문 Table VI
```
