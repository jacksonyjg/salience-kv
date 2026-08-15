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
from core.collapse_metrics import is_collapsed

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

    def _clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clean_prediction(self, text: str) -> str:
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if "</think>" in text:
            text = text.split("</think>")[-1]
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

    def evaluate_sample(self, sample, method_name, budget_ratio, measure_efficiency=True,
                         method_kwargs=None, max_input_length=None):
        self._clear_cache()

        prompt = make_prompt(
            model_key=self.model_key, tokenizer=self.tokenizer,
            context=sample["context"], question=sample["question"],
            task_type=sample["task_type"],
        )
        model_max_len = self.cfg.get("max_length", 16000)
        if max_input_length is None:
            # 기본값: 기존 16k 하드캡 동작 유지 (TABLE III~VIII 회귀 방지)
            safe_max_input = min(16000, model_max_len - 1000)
        else:
            # TABLE IX 등에서 명시적으로 캡을 조정/해제하고 싶을 때 사용.
            # max_input_length=-1 이면 모델 실제 max_length까지 허용.
            safe_max_input = (model_max_len - 1000) if max_input_length == -1 else max_input_length
        inputs = tokenize_prompt(prompt, self.tokenizer, self.model_key, max_input_length=safe_max_input, device=self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_length = input_ids.shape[1]

        hook_cache = make_hook_cache(method_name, budget_ratio, self.cfg, **(method_kwargs or {}))

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
            last_logits = prefill_out.logits[:, -1, :].detach().clone()
        prefill_time_ms = (time.perf_counter() - prefill_start) * 1000

        kv_before = self._kv_size_mb(prefill_kv) if measure_efficiency else 0.0

        # ── Step 2: KV 복사 + key 저장 + 압축 ───────────────
        compress_start = time.perf_counter()

        if isinstance(hook_cache, BaseHookCache):
            if hasattr(prefill_kv, 'layers'):
                for i, layer in enumerate(prefill_kv.layers):
                    if i >= len(hook_cache.layers):
                        hook_cache.update(layer.keys, layer.values, i)
                    else:
                        hook_cache.layers[i].keys = layer.keys.clone()
                        hook_cache.layers[i].values = layer.values.clone()
                    hook_cache.set_prefill_keys(i, layer.keys)

            del prefill_kv
            gc.collect()
            torch.cuda.empty_cache()

            hook_cache.mark_prefill_done(input_length)
            hook_cache.apply_compression_all_layers()
            compressed_cache = hook_cache
        else:
            # [2026-08-11 수정] FullKV: 이전엔 prefill_kv를 버리고 model.generate()가
            # 처음부터 다시 prefill하게 방치했음 - 실측(ttft_check.py)으로 이중 prefill
            # 확인됨(forward 호출 길이가 [N, N, 1] 패턴). 이제 prefill_kv를 그대로
            # compressed_cache로 재사용해 압축 경로와 동일한 단일 prefill 방식으로 통일.
            compressed_cache = prefill_kv

        compress_time_ms = (time.perf_counter() - compress_start) * 1000

        kv_after = 0.0 if not measure_efficiency else self._kv_size_mb(compressed_cache)
        mem_red = (1 - kv_after / kv_before) * 100 if (measure_efficiency and kv_before > 0) else 0.0
        # ── Step 3: Generate ──────────────────────────────────
        gen_start = time.perf_counter()
        try:
            eos_id = self.tokenizer.eos_token_id
            max_new = sample["max_new_tokens"]

            cache_len = compressed_cache.layers[0].keys.shape[2]

            next_token = last_logits.argmax(dim=-1, keepdim=True)
            generated_ids = [next_token.item()]

            cur_pos = input_length
            step = 0
            first_decode_step_ms = None
            step_times = []  # 첫 스텝 제외한 이후 스텝들의 소요시간(초)
            while step < max_new - 1 and generated_ids[-1] != eos_id:
                step_start = time.perf_counter()
                cache_position = torch.tensor([cur_pos], device=self.device)
                attn_mask_step = torch.ones(
                    1, cache_len + step + 1,
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                with torch.no_grad():
                    step_out = self.model(
                        input_ids=next_token,
                        past_key_values=compressed_cache,
                        cache_position=cache_position,
                        attention_mask=attn_mask_step,
                        use_cache=True,
                        return_dict=True,
                    )
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                cur_pos += 1
                step_elapsed = time.perf_counter() - step_start
                if first_decode_step_ms is None:
                    first_decode_step_ms = step_elapsed * 1000
                else:
                    step_times.append(step_elapsed)
                step += 1

            new_tokens = torch.tensor(generated_ids, device=self.device)
            decode_throughput = (len(step_times) / sum(step_times)) if step_times else 0.0
        except Exception as e:
            logger.error(f"generate() failed: {e}")
            return {
                "score": 0.0, "prediction": "", "ttft_ms": 0.0,
                "throughput": 0.0, "kv_size_before_mb": kv_before,
                "kv_size_after_mb": kv_after, "memory_reduction_pct": mem_red,
                "prefill_ms": prefill_time_ms, "compress_ms": compress_time_ms,
                "first_decode_step_ms": 0.0, "decode_throughput": 0.0,
            }

        gen_elapsed = time.perf_counter() - gen_start
        num_new = new_tokens.shape[0]
        throughput = num_new / gen_elapsed if gen_elapsed > 0 else 0.0
        ttft_ms = prefill_time_ms + compress_time_ms + (gen_elapsed * 1000 / max(num_new, 1))

        prediction = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        prediction = self._clean_prediction(prediction)
        logger.info(f"[{method_name}] pred={prediction[:120]!r}")
        score = compute_score(prediction, sample["answers"], sample["metric"])

        return {
            "score": score, "prediction": prediction, "ttft_ms": ttft_ms,
            "throughput": throughput, "kv_size_before_mb": kv_before,
            "kv_size_after_mb": kv_after, "memory_reduction_pct": mem_red,
            # TABLE IX용 세분화 지표 (기존 ttft_ms/throughput은 하위호환 위해 유지)
            "prefill_ms": prefill_time_ms,
            "compress_ms": compress_time_ms,
            "first_decode_step_ms": first_decode_step_ms if first_decode_step_ms is not None else 0.0,
            "decode_throughput": decode_throughput,
        }

    def evaluate_task(self, samples, method_name, budget_ratio, max_samples=None,
                       method_kwargs=None, max_input_length=None):
        if max_samples is not None:
            samples = samples[:max_samples]
        scores, ttfts, throughputs, mem_reds = [], [], [], []
        prefill_mss, compress_mss, first_decode_mss, decode_throughputs = [], [], [], []
        task_name = samples[0]["task_name"] if samples else "unknown"
        logger.info(f"[{method_name}] {task_name} ({len(samples)}샘플, budget={budget_ratio:.0%}, kwargs={method_kwargs})")

        collapses = []
        for i, sample in enumerate(samples):
            try:
                r = self.evaluate_sample(sample, method_name, budget_ratio, method_kwargs=method_kwargs,
                                          max_input_length=max_input_length)
                scores.append(r["score"]); ttfts.append(r["ttft_ms"])
                throughputs.append(r["throughput"]); mem_reds.append(r["memory_reduction_pct"])
                prefill_mss.append(r.get("prefill_ms", 0.0))
                compress_mss.append(r.get("compress_ms", 0.0))
                first_decode_mss.append(r.get("first_decode_step_ms", 0.0))
                decode_throughputs.append(r.get("decode_throughput", 0.0))
                collapses.append(1.0 if is_collapsed(r["prediction"]) else 0.0)
                if (i + 1) % 5 == 0:
                    logger.info(f"  [{i+1}/{len(samples)}] avg={aggregate_scores(scores):.2f}")
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
                scores.append(0.0); ttfts.append(0.0)
                throughputs.append(0.0); mem_reds.append(0.0)
                prefill_mss.append(0.0); compress_mss.append(0.0)
                first_decode_mss.append(0.0); decode_throughputs.append(0.0)
                collapses.append(1.0)

        return {
            "avg_score": aggregate_scores(scores), "scores": scores,
            "avg_ttft_ms": aggregate_scores(ttfts),
            "avg_throughput": aggregate_scores(throughputs),
            "avg_memory_reduction_pct": aggregate_scores(mem_reds),
            # TABLE IX용 세분화 집계 지표
            "avg_prefill_ms": aggregate_scores(prefill_mss),
            "avg_compress_ms": aggregate_scores(compress_mss),
            "avg_first_decode_step_ms": aggregate_scores(first_decode_mss),
            "avg_decode_throughput": aggregate_scores(decode_throughputs),
            "avg_collapse_rate_pct": (sum(collapses) / len(collapses) * 100) if collapses else 0.0,
            "collapse_count": int(sum(collapses)),
            "collapse_total": len(collapses),
            "num_samples": len(scores), "method": method_name,
            "task": task_name, "budget_ratio": budget_ratio,
        }
