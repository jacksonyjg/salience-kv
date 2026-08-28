#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""저장소 정비 패치 — /workspace/kv-cache-exp (또는 salience-kv 클론 루트)에서 실행.

    python3 repo_patch_20260828.py            # 미리보기(변경 없음)
    python3 repo_patch_20260828.py --apply    # 실제 적용

모든 치환은 count==1 검증 후 수행. 하나라도 어긋나면 아무것도 바꾸지 않고 중단.
"""
import io, os, sys

APPLY = '--apply' in sys.argv
edits = []          # (path, old, new, tag)


def E(path, old, new, tag):
    edits.append((path, old, new, tag))


# ══════════════════════════════════════════════════════════════
# P-1 [동작 변경] invert_norm 기본값 False -> True
#   근거: canonical 실행은 모두 invert_norm=True (JSON 메타데이터).
#         논문 §III-B.2 는 low-norm 우선(Devoto et al. 방향)을 유일 설정으로 서술.
#         그러나 scripts/run_all.sh 는 --invert_norm 을 넘기지 않아
#         그대로 실행하면 선택 방향이 반대가 되어 논문 수치가 재현되지 않음.
# ══════════════════════════════════════════════════════════════
E('core/kv_cache_hook.py',
  """    def __init__(self, budget_ratio: float, num_layers: int, model_config: Dict, sink_size: int = 0,
                 invert_norm: bool = False):""",
  """    def __init__(self, budget_ratio: float, num_layers: int, model_config: Dict, sink_size: int = 0,
                 invert_norm: bool = True):""",
  'P-1a Base 기본값')

for cls, sig in [
    ('StreamingLLMCache',
     """    def __init__(self, budget_ratio, num_layers, model_config, sink_size=4,
                 invert_norm=False):"""),
    ('SnapKVCache',
     """    def __init__(self, budget_ratio, num_layers, model_config, window_size=32, kernel_size=5, sink_size=0,
                 invert_norm=False):"""),
    ('PyramidKVCache',
     """    def __init__(self, budget_ratio, num_layers, model_config, min_ratio=0.1, sink_size=0,
                 invert_norm=False):"""),
]:
    E('core/kv_cache_hook.py', sig, sig.replace('invert_norm=False', 'invert_norm=True'),
      'P-1 %s' % cls)

E('core/kv_cache_hook.py',
  """                 use_attention=True, use_entropy=True, use_semantic=False, use_position=True,
                 sink_size=0, invert_norm=False):""",
  """                 use_attention=True, use_entropy=True, use_semantic=False, use_position=True,
                 sink_size=0, invert_norm=True):""",
  'P-1 OursHybridCache')

E('core/kv_cache_hook.py',
  "        self.invert_norm = invert_norm  # True면 low-key-norm 토큰 우선 (Devoto et al. 방향)",
  "        # True면 low-key-norm 토큰 우선 (Devoto et al. 방향). 논문의 모든 결과가 이 설정이며,\n"
  "        # 2026-08-28 이후 기본값이다. False 는 초기 개발용 legacy 방향이므로 논문 수치를 재현하지 않는다.\n"
  "        self.invert_norm = invert_norm",
  'P-1 주석')

# ── argparse help 및 기본값 정합 ────────────────────────────
E('experiments/exp1_main_results.py',
  """    parser.add_argument("--invert_norm", action="store_true",
                        help="key-norm 선택 방향을 corrected(low-norm 우선, Devoto et al. 방향)로 전환. "
                             "기본값(미지정)은 기존 legacy 방향(high-norm 우선) 유지.")""",
  """    parser.add_argument("--invert_norm", action="store_true", default=True,
                        help="key-norm 선택 방향을 low-norm 우선(Devoto et al. 방향)으로 둔다. "
                             "논문의 모든 결과가 이 설정이며 기본값이다.")
    parser.add_argument("--no_invert_norm", dest="invert_norm", action="store_false",
                        help="legacy 방향(high-norm 우선). 논문 수치를 재현하지 않는다.")""",
  'P-1 argparse')

# ── run_all.sh 에 명시 (기본값이 True 여도 의도를 남긴다) ────
E('scripts/run_all.sh',
  '''COMMON_ARGS="--model $MODEL"''',
  '''# invert_norm: low-norm 우선(Devoto et al. 방향). 논문의 모든 결과가 이 설정이다.
# core/kv_cache_hook.py 의 기본값도 True 이지만, 의도를 남기기 위해 명시한다.
COMMON_ARGS="--model $MODEL --invert_norm"''',
  'P-1 run_all.sh')


# ══════════════════════════════════════════════════════════════
# P-2 [주석] OursHybridCache docstring — 논문과 정면 충돌
#   현재: 4-signal + entropy 기반 레이어별 dynamic budget
#   실제: 3-signal(N+V+P) + 균일 예산. Sem 은 §V-D 절제 조건으로만 사용.
# ══════════════════════════════════════════════════════════════
E('core/kv_cache_hook.py',
  '''    """
    4-signal Hybrid Score + entropy 기반 레이어별 dynamic budget.

    신호 구성:
      A (α=0.40): key norm 기반 누적 importance (attention 근사)
      E (β=0.20): 헤드 간 key norm 분산 (정보 다양성)
      Sem (γ=0.20): prefill 평균 키와의 cosine similarity (의미적 관련성 근사)
      P (δ=0.20): 위치 감쇠 exp(-λ*(L-i)/L)

    레이어 budget:
      key norm variance 기반 동적 할당 (AdaKV와 동일 원리, 더 정교한 score)
    """''',
  '''    """
    SalienceKV — 3-signal 가중합 + 앞 m개 / 뒤 w개 예약 + 중간 구간 top-k.

    신호 구성 (논문 Eq. (5)):
      N (α=0.40): head-averaged key L2 norm. invert_norm=True 이므로 low-norm 우선
      V (β=0.20): 헤드 간 key norm 분산
      P (δ=0.20): 위치 감쇠 exp(-λ*(N-1-i)/N)

      Sem (γ): use_semantic=True 일 때만 활성. 논문의 주 수식에는 포함되지 않으며
               §V-D 의 4-signal 절제 조건에서만 사용된다.

    레이어 budget:
      균일. B_l = floor(r*N) 로 모든 레이어 동일. 레이어별 동적 할당(DLBA)은
      설계만 존재하고 구현/평가되지 않았다 (논문 §III-C, Appendix C).
    """''',
  'P-2 docstring')


# ══════════════════════════════════════════════════════════════
# P-3 [주석] 사문화된 모델 파라미터 — 로드 시 실제 값으로 덮어써짐
#   Qwen3-4B 실제: 36 layers / Q 32 / KV 8  (논문 §IV-C 와 일치)
# ══════════════════════════════════════════════════════════════
E('core/model_loader.py',
  '''    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B",
        "num_layers": 32,
        "num_heads": 8,          # num_attention_heads (GQA query heads)
        "num_kv_heads": 8,       # num_key_value_heads (GQA kv heads) — Qwen3-4B는 8로 동일''',
  '''    # NOTE: num_layers / num_heads / num_kv_heads 는 load_model_and_tokenizer() 가
    #       model.config 의 실제 값으로 덮어쓴다(아래 "CONFIG를 실제 값으로 업데이트").
    #       아래 값은 로드 전 참고용이며, 실제 실행에는 사용되지 않는다.
    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B",
        "num_layers": 36,        # Qwen3-4B 실제 값
        "num_heads": 32,         # num_attention_heads (GQA query heads)
        "num_kv_heads": 8,       # num_key_value_heads (GQA kv heads)''',
  'P-3 qwen3-4b config')


# ══════════════════════════════════════════════════════════════
# P-4 [주석] Transformers 버전 표기
# ══════════════════════════════════════════════════════════════
E('core/evaluator_v2.py',
  'Hook Cache 방식 평가 엔진. Transformers 5.10.2 DynamicCache 구조 대응.',
  'Hook Cache 방식 평가 엔진. Transformers 5.0.0 DynamicCache 구조 대응.',
  'P-4 버전 표기')


# ══════════════════════════════════════════════════════════════
# P-5 [보안] setup.sh 의 --add-to-git-credential 제거
# ══════════════════════════════════════════════════════════════
E('setup.sh',
  '''    hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \\
        || huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \\''',
  '''    hf auth login --token "$HF_TOKEN" 2>/dev/null \\
        || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null \\''',
  'P-5 setup.sh')


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
cache = {}
fail = 0
for path, old, new, tag in edits:
    if not os.path.exists(path):
        print('MISS %-22s %s (파일 없음)' % (tag, path)); fail += 1; continue
    if path not in cache:
        cache[path] = io.open(path, encoding='utf-8').read()
    c = cache[path].count(old)
    if c != 1:
        print('FAIL %-22s %s  count=%d' % (tag, path, c))
        print('     anchor: %r' % old[:70]); fail += 1; continue
    cache[path] = cache[path].replace(old, new)
    print('ok   %-22s %s' % (tag, path))

if fail:
    print('\n%d건 실패 — 아무것도 변경하지 않았습니다.' % fail); sys.exit(1)

print('\n검증 통과: %d건 / 파일 %d개' % (len(edits), len(cache)))
if not APPLY:
    print('미리보기 모드입니다. 실제 적용하려면 --apply 를 붙이세요.')
    sys.exit(0)

for path, text in cache.items():
    io.open(path, 'w', encoding='utf-8').write(text)
    print('  wrote %s' % path)
print('\n완료. git diff 로 확인 후 커밋하세요.')
