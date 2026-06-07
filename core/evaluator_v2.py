"""
core/evaluator_v2.py (new)
===========================
Hook Cache 방식 평가 엔진. Transformers 5.10.2 DynamicCache 구조 대응.
"""

import time
import gc
import logging
from typing import Dict, List, Optional, Tuple, Any

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

    def _clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clean_prediction(self, text: str) -> str:
        # </think> 있으면 그 이후만 사용
        if "</think>" in text:
            parts = text.split("</think>")
            text = parts[-1].strip()
        # <think>만 있고 내용이 있으면 태그 제거 후 반환
        elif "<think>" in text:
            text = text.replace("<think>", "").replace("</think>", "").strip()
        return text.strip()

    def _collect_prefill_attn(self, hook_cache, input_ids, attention_mask):
        """forward hook으로 prefill attention 수집."""
        if not isinstance(hook_cache, BaseHookCache):
            return
        hooks = []
        def make_hook(idx):
            def fn(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    hook_cache.set_prefill_attn(idx, out[1])
            return fn
        for i in range(self.cfg["num_layers"]):
            try:
                h = self.model.model.layers[i].self_attn.register_forward_hook(make_hook(i))
                hooks.append(h)
            except Exception:
                pass
        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    output_attentions=False,
                )
        finally:
            for h in hooks:
                h.remove()

    def _kv_size_mb(self, cache) -> float:
        """캐시 크기 MB 계산."""
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

        # Hook cache 생성
        hook_cache = make_hook_cache(method_name, budget_ratio, self.cfg)

        # ── Step 1: Prefill ──────────────────────────────────
        # attention 수집 (BaseHookCache인 경우)
        self._collect_prefill_attn(hook_cache, input_ids, attention_mask)

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

        # KV 크기 (압축 전)
        kv_before = self._kv_size_mb(prefill_kv) if measure_efficiency else 0.0

        # ── Step 2: 압축 ──────────────────────────────────────
        compress_start = time.perf_counter()

        if isinstance(hook_cache, BaseHookCache):
            # prefill KV → hook_cache의 layers로 복사
            if hasattr(prefill_kv, 'layers'):
                for i, layer in enumerate(prefill_kv.layers):
                    if i >= len(hook_cache.layers):
                        hook_cache.update(layer.keys, layer.values, i)
                    else:
                        hook_cache.layers[i].keys = layer.keys.clone()
                        hook_cache.layers[i].values = layer.values.clone()

            hook_cache.mark_prefill_done(input_length)
            hook_cache.apply_compression_all_layers()
            compressed_cache = hook_cache
        else:
            # FullKV: past_key_values 없이 generate (position 중복 방지)
            compressed_cache = None

        compress_time_ms = (time.perf_counter() - compress_start) * 1000

        # KV 크기 (압축 후)
        kv_after = self._kv_size_mb(compressed_cache) if measure_efficiency else 0.0
        mem_red = (1 - kv_after / kv_before) * 100 if kv_before > 0 else 0.0

        # ── Step 3: Generate ──────────────────────────────────
        gen_start = time.perf_counter()
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=compressed_cache,
                    max_new_tokens=sample["max_new_tokens"],
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # thinking 억제: /no_think 접미사 또는 토큰 억제
                    # suppress_tokens 제거: thinking 억제 시 모델이 EOS 즉시 생성
                    # thinking 내용은 _clean_prediction에서 후처리로 제거
                )
            new_tokens = output[0, input_length:]
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
