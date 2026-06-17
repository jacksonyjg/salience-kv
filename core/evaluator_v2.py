"""
core/evaluator_v2.py (v3)
===========================
Hook Cache 방식 평가 엔진. Transformers 5.10.2 DynamicCache 구조 대응.

변경사항 (v2 → v3):
  - _collect_prefill_attn() 완전 제거 (output_attentions=True OOM 문제 해결)
  - KV 복사 단계에서 set_prefill_keys() 호출로 대체
    → 이미 GPU에 올라온 key tensor를 CPU로 옮겨 저장, 별도 forward 없음
  - 코드 흐름: prefill → KV 복사 + key 저장 → 압축 → generate
"""

import time
import gc
import logging
from typing import Dict, List, Optional

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np

from core.kv_cache_hook import BaseHookCache, FullKVCache, make_hook_cache
from core.metrics import compute_score, aggregate_scores
from core.model_loader import tokenize_prompt, make_prompt

logger = logging.getLogger(__name__)


class EvaluatorV2:
    def __init__(self, model, tokenizer, model_config, device="cuda", seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = model_config
        self.device = device
        self.model_key = model_config["model_key"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        # RoPE re-indexing용: 압축 후 캐시의 토큰 position을 0..N-1로 재정렬하기 위해
        # 모델의 rotary embedding(inv_freq)을 hook cache에 전달한다.
        try:
            self.cfg["rotary_emb"] = self.model.model.rotary_emb
        except AttributeError:
            logger.warning("model.model.rotary_emb 접근 실패 - RoPE re-indexing 비활성화됨")
            self.cfg["rotary_emb"] = None

    def _clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clean_prediction(self, text: str) -> str:
        import re
        # <think>...</think> 블록 전체 제거
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # </think> 이후만 남기기 (블록이 잘린 경우)
        if "</think>" in text:
            text = text.split("</think>")[-1]
        # 고아 <think> 태그 제거
        text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()

    def _kv_size_mb(self, cache) -> float:
        total = 0
        dtype = self.cfg.get("dtype", torch.float16)
        bytes_per = 2 if dtype == torch.float16 else 4
        if hasattr(cache, 'layers'):
            for layer in cache.layers:
                if hasattr(layer, 'keys') and layer.keys is not None:
                    total += layer.keys.numel() * bytes_per
                if hasattr(layer, 'values') and layer.values is not None:
                    total += layer.values.numel() * bytes_per
        return total / (1024 * 1024)

    def evaluate_sample(self, sample, method_name, budget_ratio, measure_efficiency=True):
        self._clear_cache()

        prompt = make_prompt(
            model_key=self.model_key, tokenizer=self.tokenizer,
            context=sample["context"], question=sample["question"],
            task_type=sample["task_type"],
        )
        inputs = tokenize_prompt(prompt, self.tokenizer, self.model_key, device=self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_length = input_ids.shape[1]

        hook_cache = make_hook_cache(method_name, budget_ratio, self.cfg)

        # ── Step 1: Prefill ──────────────────────────────────
        prefill_start = time.perf_counter()
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
            prefill_kv = prefill_out.past_key_values
        prefill_time_ms = (time.perf_counter() - prefill_start) * 1000

        kv_before = self._kv_size_mb(prefill_kv) if measure_efficiency else 0.0

        # ── Step 2: KV 복사 + key 저장 + 압축 ───────────────
        compress_start = time.perf_counter()

        if isinstance(hook_cache, BaseHookCache):
            if hasattr(prefill_kv, 'layers'):
                for i, layer in enumerate(prefill_kv.layers):
                    # KV 복사
                    if i >= len(hook_cache.layers):
                        hook_cache.update(layer.keys, layer.values, i)
                    else:
                        hook_cache.layers[i].keys = layer.keys.clone()
                        hook_cache.layers[i].values = layer.values.clone()
                    # key tensor 저장 (importance 계산용, CPU)
                    hook_cache.set_prefill_keys(i, layer.keys)

            # 원본 KV 즉시 해제
            del prefill_kv
            gc.collect()
            torch.cuda.empty_cache()

            hook_cache.mark_prefill_done(input_length)
            hook_cache.apply_compression_all_layers()
            compressed_cache = hook_cache
        else:
            # FullKV: past_key_values 없이 generate (position 중복 방지)
            del prefill_kv
            gc.collect()
            torch.cuda.empty_cache()
            compressed_cache = None

        compress_time_ms = (time.perf_counter() - compress_start) * 1000

        if not measure_efficiency:
            kv_after = 0.0
            mem_red = 0.0
        elif compressed_cache is None:
            # FullKV: 압축 없음 → mem_red=0%
            kv_after = kv_before
            mem_red = 0.0
        else:
            kv_after = self._kv_size_mb(compressed_cache)
            mem_red = (1 - kv_after / kv_before) * 100 if kv_before > 0 else 0.0
        # ── Step 3: Generate ──────────────────────────────────
        gen_start = time.perf_counter()
        try:
            if compressed_cache is not None:
                # 레이어별 seq_len 중 최솟값으로 모든 레이어 통일
                min_len = min(
                    layer.keys.shape[2]
                    for layer in compressed_cache.layers
                    if layer.keys is not None
                )
                for layer in compressed_cache.layers:
                    if layer.keys is not None and layer.keys.shape[2] > min_len:
                        # 앞부분(과거 컨텍스트)이 아니라 뒷부분(최근 토큰, 질문 포함)을
                        # 유지해야 한다. 레이어별 budget이 다른 방법(PyramidKV, AdaKV,
                        # OursHybrid)은 각 레이어의 idx가 이미 "질문이 포함된 마지막
                        # window"를 보존하도록 정렬되어 있는데, 여기서 [:min_len]으로
                        # 앞에서 자르면 그 보존된 마지막 부분이 잘려나가 질문이 캐시에서
                        # 사라지는 문제가 있었다 (실측: PyramidKV/AdaKV 모든 레이어가
                        # truncate 후 input_length-1 position을 누락 -> 반복루프/EOS).
                        layer.keys = layer.keys[:, :, -min_len:, :]
                        layer.values = layer.values[:, :, -min_len:, :]
                # ── 캐시 1-토큰 롤백 ──────────────────────────────
                # 압축된 캐시는 원본 시퀀스에서 sparse하게 선택된 토큰들이므로,
                # "캐시 길이(min_len)만큼의 prefix가 이미 처리되었다"는 generate()의
                # 기본 가정(_cache_dependant_input_preparation의 input_ids[:, cache_position]
                # 슬라이싱)과 양립하지 않는다. input_ids 전체를 다시 넣으면 캐시에 없는
                # 임의의 중간 구간이 입력되어 컨텍스트가 깨진다.
                # 해결: 캐시의 마지막 1토큰을 롤백(제거)하고, 그 토큰을 input_ids로 다시
                # 넣어 cache_position을 캐시 길이에 정확히 맞춰 첫 logit을 재계산한다.
                # (transformers generate() 코드 주석에도 명시된 표준 패턴:
                #  "we need to roll back the cache 1 token to recompute the logits
                #   for the first token to be generated")
                for layer in compressed_cache.layers:
                    if layer.keys is not None and layer.keys.shape[2] > 0:
                        layer.keys = layer.keys[:, :, :-1, :]
                        layer.values = layer.values[:, :, :-1, :]
                rollback_len = min_len - 1
                gen_input_ids = input_ids[:, -1:]
                past_mask = torch.ones(
                    1, rollback_len,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                gen_attention_mask = torch.cat(
                    [past_mask, attention_mask[:, -1:]], dim=1
                )
                cache_position = torch.tensor(
                    [rollback_len], device=gen_input_ids.device
                )
            else:
                gen_input_ids = input_ids
                gen_attention_mask = attention_mask
                cache_position = None
            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=compressed_cache,
                    max_new_tokens=sample["max_new_tokens"],
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                if cache_position is not None:
                    gen_kwargs["cache_position"] = cache_position
                output = self.model.generate(**gen_kwargs)
            gen_prompt_len = gen_input_ids.shape[1]
            new_tokens = output[0, gen_prompt_len:]
        except Exception as e:
            logger.error(f"generate() failed: {e}")
            return {
                "score": 0.0, "prediction": "", "ttft_ms": 0.0,
                "throughput": 0.0, "kv_size_before_mb": kv_before,
                "kv_size_after_mb": kv_after, "memory_reduction_pct": mem_red,
            }

        gen_elapsed = time.perf_counter() - gen_start
        num_new = new_tokens.shape[0]
        throughput = num_new / gen_elapsed if gen_elapsed > 0 else 0.0
        ttft_ms = prefill_time_ms + compress_time_ms + (gen_elapsed * 1000 / max(num_new, 1))

        prediction = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        prediction = self._clean_prediction(prediction)
        logger.info(f"[{method_name}] raw_new_tokens={new_tokens.tolist()[:40]} prediction={prediction!r}")
        score = compute_score(prediction, sample["answers"], sample["metric"])

        return {
            "score": score, "prediction": prediction, "ttft_ms": ttft_ms,
            "throughput": throughput, "kv_size_before_mb": kv_before,
            "kv_size_after_mb": kv_after, "memory_reduction_pct": mem_red,
        }

    def evaluate_task(self, samples, method_name, budget_ratio, max_samples=None):
        if max_samples is not None:
            samples = samples[:max_samples]
        scores, ttfts, throughputs, mem_reds = [], [], [], []
        task_name = samples[0]["task_name"] if samples else "unknown"
        logger.info(f"[{method_name}] {task_name} ({len(samples)}샘플, budget={budget_ratio:.0%})")

        for i, sample in enumerate(samples):
            try:
                r = self.evaluate_sample(sample, method_name, budget_ratio)
                scores.append(r["score"]); ttfts.append(r["ttft_ms"])
                throughputs.append(r["throughput"]); mem_reds.append(r["memory_reduction_pct"])
                if (i + 1) % 5 == 0:
                    logger.info(f"  [{i+1}/{len(samples)}] avg={aggregate_scores(scores):.2f}")
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
                scores.append(0.0); ttfts.append(0.0)
                throughputs.append(0.0); mem_reds.append(0.0)

        return {
            "avg_score": aggregate_scores(scores), "scores": scores,
            "avg_ttft_ms": aggregate_scores(ttfts),
            "avg_throughput": aggregate_scores(throughputs),
            "avg_memory_reduction_pct": aggregate_scores(mem_reds),
            "num_samples": len(scores), "method": method_name,
            "task": task_name, "budget_ratio": budget_ratio,
        }
