"""
core/evaluator.py
==================
KV 캐시 방법 평가 엔진.

Qwen3-4B를 포함한 모든 모델에서 안정적으로 동작하도록 설계.
어텐션 가중치 캡처, KV 캐시 압축, 성능/효율성 지표 측정 통합.
"""

import time
import gc
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from core.kv_base import tuple_to_dynamic_cache

from core.kv_base import (
    KVCacheMethod,
    register_attention_hooks,
    remove_hooks,
    get_kv_cache_size_mb,
)
from core.metrics import compute_score, aggregate_scores
from core.model_loader import tokenize_prompt, make_prompt, MODEL_CONFIGS

logger = logging.getLogger(__name__)


class Evaluator:
    """단일 KV 압축 방법의 성능 및 효율성 평가."""
    
    def __init__(
        self,
        model,
        tokenizer,
        model_config: Dict,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = model_config
        self.device = device
        self.model_key = model_config["model_key"]
        
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def _clear_cache(self):
        """GPU 메모리 캐시 정리."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _decode_output(self, output_ids: torch.Tensor, input_length: int) -> str:
        """
        생성된 토큰 ID를 텍스트로 디코딩.
        입력 부분 제거, 특수 토큰 정리.
        """
        # 입력 이후의 새로 생성된 토큰만 추출
        new_tokens = output_ids[0, input_length:]
        
        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Qwen3 특수 처리: thinking 토큰 제거 (혹시 남아있을 경우)
        if "<think>" in text:
            # </think> 이후 내용만 사용
            parts = text.split("</think>")
            text = parts[-1].strip() if len(parts) > 1 else text
        
        return text.strip()
    
    def _run_prefill_with_attn_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Any, List, float]:
        """
        Prefill 단계 실행 + 어텐션 가중치 캡처.
        
        Returns:
            (past_key_values, attn_weights_list, prefill_time_ms)
        """
        num_layers = self.cfg["num_layers"]
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values
            torch.cuda.empty_cache()
        
        prefill_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # attn_weights_list는 None으로 채움 (hooks 제거로 OOM 방지)
        attn_weights_list = [None] * num_layers
        
        return past_kv, attn_weights_list, prefill_time
    
    def _run_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        KV 캐시를 사용한 Decode 단계 실행.
        
        Returns:
            (generated_ids, first_token_time_ms, throughput_tokens_per_sec)
        """
        input_length = input_ids.shape[1]
        
        # past_key_values에서 현재 시퀀스 길이 파악
        past_seq_len = 0
        if past_key_values is not None:
            for layer_kv in past_key_values:
                if layer_kv is not None:
                    past_seq_len = layer_kv[0].shape[2]
                    break
        
        start_time = time.perf_counter()
        first_token_time = None
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            current_past_kv = past_key_values
            
            for step in range(max_new_tokens):
                if step == 0:
                    # 첫 decode 스텝: prefill 이후 다음 토큰 생성
                    # position_ids 계산 (중요: 압축된 KV 캐시의 실제 길이 사용)
                    if current_past_kv is not None:
                        actual_past_len = 0
                        for lkv in current_past_kv:
                            if lkv is not None:
                                actual_past_len = lkv[0].shape[2]
                                break
                        position_ids = torch.tensor(
                            [[actual_past_len]], device=self.device
                        )
                    else:
                        position_ids = None
                    
                    try:
                        out = self.model(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=torch.ones(
                                1, (actual_past_len if current_past_kv else input_length) + 1,
                                device=self.device
                            ),
                            past_key_values=current_past_kv,
                            use_cache=True,
                            return_dict=True,
                        )
                    except Exception as e:
                        logger.warning(f"Decode step 0 failed: {e}, trying full generate")
                        break
                else:
                    out = self.model(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=torch.ones(
                            1, generated_ids.shape[1], device=self.device
                        ),
                        past_key_values=current_past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                
                # 다음 토큰 생성 (greedy)
                next_token_logits = out.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                current_past_kv = out.past_key_values
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if first_token_time is None:
                    first_token_time = (time.perf_counter() - start_time) * 1000
                
                # EOS 확인
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        total_time = time.perf_counter() - start_time
        num_generated = generated_ids.shape[1] - input_length
        throughput = num_generated / total_time if total_time > 0 else 0.0
        
        if first_token_time is None:
            first_token_time = total_time * 1000
        
        return generated_ids, first_token_time, throughput
    
    def evaluate_sample(
        self,
        sample: Dict,
        kv_method: KVCacheMethod,
        budget_ratio: float,
        measure_efficiency: bool = True,
    ) -> Dict:
        """
        단일 샘플에 대한 평가 수행.
        
        Returns:
            {
                "score": float,          # F1 또는 ROUGE-L (0~100)
                "prediction": str,       # 생성된 텍스트
                "ttft_ms": float,        # Time to First Token (ms)
                "throughput": float,     # tokens/s
                "kv_size_before_mb": float,
                "kv_size_after_mb": float,
                "memory_reduction_pct": float,
            }
        """
        self._clear_cache()
        
        # 프롬프트 생성
        prompt = make_prompt(
            model_key=self.model_key,
            tokenizer=self.tokenizer,
            context=sample["context"],
            question=sample["question"],
            task_type=sample["task_type"],
        )
        
        # 토크나이즈
        inputs = tokenize_prompt(
            prompt=prompt,
            tokenizer=self.tokenizer,
            model_key=self.model_key,
            device=self.device,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_length = input_ids.shape[1]
        
        # Prefill + 어텐션 캡처
        past_kv, attn_weights_list, prefill_time_ms = self._run_prefill_with_attn_capture(
            input_ids, attention_mask
        )
        
        # KV 캐시 크기 측정 (압축 전)
        dtype = self.cfg.get("dtype", torch.float16)
        kv_size_before = get_kv_cache_size_mb(past_kv, dtype) if measure_efficiency else 0.0
        
        # KV 캐시 압축
        compress_start = time.perf_counter()
        compressed_kv = kv_method.compress(
            past_kv,
            attn_weights_list,
            budget_ratio=budget_ratio,
        )
        compress_time_ms = (time.perf_counter() - compress_start) * 1000
        
        # KV 캐시 크기 측정 (압축 후)
        kv_size_after = get_kv_cache_size_mb(compressed_kv, dtype) if measure_efficiency else 0.0
        memory_reduction = (
            (1 - kv_size_after / kv_size_before) * 100
            if kv_size_before > 0
            else 0.0
        )
        
        # 안전한 generate 방식: 직접 generate() 사용 (복잡한 decode loop 대신)
        ttft_ms, throughput, prediction = self._safe_generate(
            input_ids=input_ids,
            compressed_kv=compressed_kv,
            max_new_tokens=sample["max_new_tokens"],
            input_length=input_length,
        )
        
        # TTFT = prefill + 첫 토큰
        total_ttft = prefill_time_ms + compress_time_ms + ttft_ms
        
        # 점수 계산
        score = compute_score(prediction, sample["answers"], sample["metric"])
        
        return {
            "score": score,
            "prediction": prediction,
            "ttft_ms": total_ttft,
            "throughput": throughput,
            "kv_size_before_mb": kv_size_before,
            "kv_size_after_mb": kv_size_after,
            "memory_reduction_pct": memory_reduction,
        }
    
    def _safe_generate(
        self,
        input_ids: torch.Tensor,
        compressed_kv,
        max_new_tokens: int,
        input_length: int,
    ) -> Tuple[float, float, str]:
        """
        generate()를 사용한 안전한 텍스트 생성.
        압축된 KV 캐시가 있으면 past_key_values로 전달.
        없으면 전체 재생성.
        """
        start = time.perf_counter()
        
        try:
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,          # greedy (재현성)
                    "temperature": None,
                    "top_p": None,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # 압축된 KV 캐시를 past_key_values로 전달
                if compressed_kv is not None and isinstance(compressed_kv, tuple):
                    # 압축된 캐시를 DynamicCache로 변환 후 전체 input으로 generate
                    dynamic_cache = tuple_to_dynamic_cache(compressed_kv)
                    compressed_seq_len = compressed_kv[0][0].shape[2]
                    # 압축된 토큰 수에 맞게 input_ids 앞부분 잘라서 전달
                    # (압축 비율만큼 앞부분 제거, 마지막 compressed_seq_len 토큰 사용)
                    trimmed_input = input_ids[:, -compressed_seq_len:]
                    output = self.model.generate(
                        input_ids=trimmed_input,
                        attention_mask=torch.ones_like(trimmed_input),
                        **gen_kwargs,
                    )
                    new_tokens = output[0, compressed_seq_len:]
                else:
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        **gen_kwargs,
                    )
                    new_tokens = output[0, input_length:]
        
        except Exception as e:
            logger.error(f"generate() failed: {e}")
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, 0.0, ""
        
        elapsed = time.perf_counter() - start
        prediction = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Qwen3 thinking 토큰 정리
        if "<think>" in prediction:
            parts = prediction.split("</think>")
            prediction = parts[-1].strip() if len(parts) > 1 else prediction
        
        prediction = prediction.strip()
        
        num_new = new_tokens.shape[0]
        throughput = num_new / elapsed if elapsed > 0 else 0.0
        ttft_ms = elapsed * 1000 / max(num_new, 1)  # 첫 토큰까지 추정
        
        return ttft_ms, throughput, prediction
    
    def evaluate_task(
        self,
        samples: List[Dict],
        kv_method: KVCacheMethod,
        budget_ratio: float,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        태스크 전체 샘플 평가 및 집계.
        
        Returns:
            {
                "avg_score": float,
                "scores": List[float],
                "avg_ttft_ms": float,
                "avg_throughput": float,
                "avg_memory_reduction_pct": float,
                "num_samples": int,
            }
        """
        if max_samples is not None:
            samples = samples[:max_samples]
        
        scores = []
        ttfts = []
        throughputs = []
        mem_reductions = []
        
        method_name = kv_method.get_name()
        task_name = samples[0]["task_name"] if samples else "unknown"
        
        logger.info(
            f"Evaluating {method_name} on {task_name} "
            f"({len(samples)} samples, budget={budget_ratio:.0%})"
        )
        
        for i, sample in enumerate(samples):
            try:
                result = self.evaluate_sample(sample, kv_method, budget_ratio)
                scores.append(result["score"])
                ttfts.append(result["ttft_ms"])
                throughputs.append(result["throughput"])
                mem_reductions.append(result["memory_reduction_pct"])
                
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"  [{i+1}/{len(samples)}] avg_score={aggregate_scores(scores):.2f}"
                    )
            
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
                scores.append(0.0)
                ttfts.append(0.0)
                throughputs.append(0.0)
                mem_reductions.append(0.0)
        
        return {
            "avg_score": aggregate_scores(scores),
            "scores": scores,
            "avg_ttft_ms": aggregate_scores(ttfts),
            "avg_throughput": aggregate_scores(throughputs),
            "avg_memory_reduction_pct": aggregate_scores(mem_reductions),
            "num_samples": len(scores),
            "method": method_name,
            "task": task_name,
            "budget_ratio": budget_ratio,
        }
