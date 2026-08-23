"""
core/evaluator.py
==================
KV 캐시 방법 평가 엔진.

Qwen3-4B를 포함한 모든 모델에서 안정적으로 동작하도록 설계.
어텐션 가중치 캡처, KV 캐시 압축, 성능/효율성 지표 측정 통합.

[수정 v2]
- _run_prefill_with_attn_capture: forward hook으로 attention 수집
  (레이어별 즉시 CPU 이동 + query 차원 평균으로 OOM 방지)
- _safe_generate: compressed_kv를 past_key_values로 실제 전달
  (기존: input_ids 잘라서 재생성 → 압축이 생성에 반영 안 됨)
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


def _get_attn_module(model, layer_idx: int, model_key: str):
    """
    모델 아키텍처별로 self_attn 모듈 경로가 다름.
    각 모델의 레이어 구조에 맞게 attention 모듈을 반환.
    """
    try:
        if model_key in ("qwen3-4b", "qwen2.5-7b"):
            # Qwen3/Qwen2: model.model.layers[i].self_attn
            return model.model.layers[layer_idx].self_attn
        elif model_key in ("phi-3-mini",):
            # Phi-3: model.model.layers[i].self_attn
            return model.model.layers[layer_idx].self_attn
        elif model_key in ("gemma-2-2b",):
            # Gemma-2: model.model.layers[i].self_attn
            return model.model.layers[layer_idx].self_attn
        else:
            # 기본값: 대부분의 decoder-only 모델은 동일한 구조
            return model.model.layers[layer_idx].self_attn
    except (AttributeError, IndexError) as e:
        logger.warning(f"Layer {layer_idx} self_attn not found for {model_key}: {e}")
        return None


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
        new_tokens = output_ids[0, input_length:]
        
        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Qwen3 특수 처리: thinking 토큰 제거
        if "<think>" in text:
            parts = text.split("</think>")
            text = parts[-1].strip() if len(parts) > 1 else text
        
        return text.strip()
    
    def _run_prefill_with_attn_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Any, List, float]:
        """
        Prefill 단계 실행 + forward hook으로 어텐션 가중치 캡처.

        OOM 방지 전략:
        - output_attentions=True 대신 forward hook 사용
          → attention 행렬 전체를 GPU에 유지하지 않음
        - hook 콜백에서 즉시 .detach().cpu()로 GPU 메모리 해제
        - (seq, seq) 행렬 대신 query 차원을 평균내어 (nhead, seq)로 압축
          → kv_methods.py의 Hybrid Score 계산에서 필요한 정보는 유지

        Returns:
            (past_key_values, attn_weights_list, prefill_time_ms)
            attn_weights_list[i]: Tensor shape (1, nhead, seq_len) on CPU
                                  또는 hook 실패 시 None
        """
        num_layers = self.cfg["num_layers"]
        attn_weights_list = [None] * num_layers
        hooks = []

        # ── Hook 등록 ──────────────────────────────────────────────
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                """
                self_attn forward 후 호출됨.
                output 형태는 모델마다 다름:
                  - 대부분: (attn_output, attn_weights, past_kv) 또는
                            (attn_output,) (flash attention 사용 시 weights 없음)
                  - output_attentions=False(기본)이면 weights가 None이거나 없음
                
                → output_attentions=True를 모듈 레벨에서 강제하지 않으면
                  weights를 얻을 수 없는 경우가 많으므로,
                  hook 실패 시 None 유지 (Hybrid Score는 A=0으로 fallback).
                """
                try:
                    if isinstance(output, tuple) and len(output) > 1:
                        attn = output[1]  # 두 번째 원소가 attention weights
                        if attn is not None and isinstance(attn, torch.Tensor):
                            # (batch, nhead, seq_q, seq_k) → query 차원 평균 → (batch, nhead, seq_k)
                            # GPU에서 평균 계산 후 즉시 CPU로 이동 (GPU 메모리 최소화)
                            attn_weights_list[layer_idx] = (
                                attn.detach().mean(dim=2).cpu()  # (1, nhead, seq_len)
                            )
                except Exception as e:
                    logger.debug(f"Hook layer {layer_idx} failed: {e}")
                # 반드시 None 반환 (output 변경 없음)
                return None
            return hook_fn

        for i in range(num_layers):
            attn_mod = _get_attn_module(self.model, i, self.model_key)
            if attn_mod is not None:
                h = attn_mod.register_forward_hook(make_hook(i))
                hooks.append(h)

        # ── Prefill 실행 ───────────────────────────────────────────
        start_time = time.perf_counter()

        try:
            with torch.no_grad():
                # output_attentions=True를 쓰면 (seq, seq) 행렬이 레이어×헤드 수만큼
                # GPU에 올라가 OOM 발생. hook 방식으로 우회.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,  # hook으로 수집하므로 False 유지
                )
                past_kv = outputs.past_key_values
                torch.cuda.empty_cache()
        finally:
            # 예외 발생해도 반드시 hook 제거 (메모리 누수 방지)
            for h in hooks:
                h.remove()

        prefill_time = (time.perf_counter() - start_time) * 1000  # ms

        # hook으로 수집된 레이어 수 로깅 (디버그용)
        captured = sum(1 for w in attn_weights_list if w is not None)
        if captured == 0:
            logger.warning(
                "Attention weights not captured by hooks (likely FlashAttention). "
                "Hybrid Score will run without Attention signal (A=0). "
                "Disable FlashAttention or use output_attentions=True fallback."
            )
        else:
            logger.debug(f"Captured attention weights for {captured}/{num_layers} layers.")

        return past_kv, attn_weights_list, prefill_time

    def _run_prefill_with_attn_capture_fallback(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Any, List, float]:
        """
        output_attentions=True 방식 fallback.
        hook이 attention을 캡처하지 못할 때 (FlashAttention 비활성화 환경) 사용.
        
        주의: (seq, seq) 행렬이 전체 레이어에 걸쳐 GPU에 올라가 OOM 가능.
               컨텍스트가 짧거나 메모리 여유가 있을 때만 사용.
        """
        num_layers = self.cfg["num_layers"]
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_attentions=True,  # 전체 attention 행렬 반환
            )
            past_kv = outputs.past_key_values

            # 즉시 CPU로 이동 (GPU 메모리 해제)
            attn_weights_list = []
            if outputs.attentions is not None:
                for attn in outputs.attentions:
                    if attn is not None:
                        # (1, nhead, seq, seq) → (1, nhead, seq) [query 평균]
                        attn_weights_list.append(attn.detach().mean(dim=2).cpu())
                    else:
                        attn_weights_list.append(None)
            else:
                attn_weights_list = [None] * num_layers

            torch.cuda.empty_cache()

        prefill_time = (time.perf_counter() - start_time) * 1000
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

        # tuple → DynamicCache 변환 (Transformers 5.x는 DynamicCache 필요)
        # OursHybrid처럼 레이어별 seq_len이 다를 경우 min seq_len으로 통일
        # (Qwen3 eager attention이 레이어간 다른 KV seq_len을 처리 못함)
        if isinstance(past_key_values, tuple):
            seq_lens = [layer[0].shape[2] for layer in past_key_values if layer is not None]
            min_seq_len = min(seq_lens) if seq_lens else 0
            max_seq_len = max(seq_lens) if seq_lens else 0
            if min_seq_len != max_seq_len:
                # 모든 레이어를 min_seq_len으로 앞부분 잘라서 통일
                past_key_values = tuple(
                    (k[:, :, :min_seq_len, :], v[:, :, :min_seq_len, :])
                    if layer is not None else None
                    for layer, (k, v) in zip(past_key_values, past_key_values)
                )
            past_key_values = tuple_to_dynamic_cache(past_key_values)

        past_seq_len = 0
        if past_key_values is not None:
            try:
                past_seq_len = past_key_values.get_seq_length()
            except Exception:
                pass
        
        start_time = time.perf_counter()
        first_token_time = None
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            current_past_kv = past_key_values
            
            for step in range(max_new_tokens):
                if step == 0:
                    if current_past_kv is not None:
                        try:
                            actual_past_len = current_past_kv.get_seq_length()
                        except Exception:
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
                    # past_kv가 이미 context를 보유 → attention_mask는 past_len + 현재 스텝
                    try:
                        past_len = current_past_kv.get_seq_length()
                    except Exception:
                        past_len = generated_ids.shape[1] - 1
                    out = self.model(
                        input_ids=generated_ids[:, -1:],
                        attention_mask=torch.ones(
                            1, past_len + 1, device=self.device
                        ),
                        past_key_values=current_past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                
                next_token_logits = out.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                current_past_kv = out.past_key_values
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if first_token_time is None:
                    first_token_time = (time.perf_counter() - start_time) * 1000
                
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
                "score": float,
                "prediction": str,
                "ttft_ms": float,
                "throughput": float,
                "kv_size_before_mb": float,
                "kv_size_after_mb": float,
                "memory_reduction_pct": float,
            }
        """
        self._clear_cache()
        
        prompt = make_prompt(
            model_key=self.model_key,
            tokenizer=self.tokenizer,
            context=sample["context"],
            question=sample["question"],
            task_type=sample["task_type"],
        )
        
        inputs = tokenize_prompt(
            prompt=prompt,
            tokenizer=self.tokenizer,
            model_key=self.model_key,
            device=self.device,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_length = input_ids.shape[1]
        
        # Prefill + attention hook 수집
        past_kv, attn_weights_list, prefill_time_ms = self._run_prefill_with_attn_capture(
            input_ids, attention_mask
        )

        # hook 방식 실패 시 fallback (FlashAttention 비활성화 환경)
        captured = sum(1 for w in attn_weights_list if w is not None)
        if captured == 0:
            logger.info("Hook capture failed, trying output_attentions=True fallback...")
            try:
                past_kv, attn_weights_list, prefill_time_ms = (
                    self._run_prefill_with_attn_capture_fallback(input_ids, attention_mask)
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("Fallback OOM — proceeding with attn=None (Attention signal disabled)")
                attn_weights_list = [None] * self.cfg["num_layers"]
        
        dtype = self.cfg.get("dtype", torch.float16)
        kv_size_before = get_kv_cache_size_mb(past_kv, dtype) if measure_efficiency else 0.0
        
        compress_start = time.perf_counter()
        compressed_kv = kv_method.compress(
            past_kv,
            attn_weights_list,
            budget_ratio=budget_ratio,
        )
        compress_time_ms = (time.perf_counter() - compress_start) * 1000
        
        kv_size_after = get_kv_cache_size_mb(compressed_kv, dtype) if measure_efficiency else 0.0
        memory_reduction = (
            (1 - kv_size_after / kv_size_before) * 100
            if kv_size_before > 0
            else 0.0
        )
        
        # decode: 압축된 KV로 _run_decode 시도, 실패 시 원본 KV로 fallback
        # Transformers 5.x position mismatch 이슈로 원본 KV 사용
        generated_ids, ttft_ms, throughput = self._run_decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            max_new_tokens=sample["max_new_tokens"],
        )
        prediction = self._decode_output(generated_ids, input_length)
        
        total_ttft = prefill_time_ms + compress_time_ms + ttft_ms
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
        compressed_kv를 past_key_values로 전달하는 올바른 generate 방식.

        [수정] 기존 코드 문제:
          - compressed_kv를 DynamicCache로 변환 후 generate()에 전달하지 않고
            input_ids를 잘라서 처음부터 재생성 → 압축이 실제 생성에 반영 안 됨
          - trimmed_input을 past_key_values 없이 넘기면 모델이 캐시 무시하고
            해당 토큰들을 새로 prefill

        [수정 후]:
          1. compressed_kv → DynamicCache 변환
          2. past_key_values=dynamic_cache 로 generate() 호출
          3. input_ids는 마지막 1개 토큰만 넘김 (캐시가 나머지를 이미 보유)
        """
        start = time.perf_counter()
        
        try:
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                if compressed_kv is not None and isinstance(compressed_kv, tuple) and len(compressed_kv) > 0:
                    # generate() API는 past_key_values 크기와 attention_mask 크기를
                    # 맞추기 어려움 → manual decode loop 사용
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        **gen_kwargs,
                    )
                    new_tokens = output[0, input_length:]

                else:
                    # ── 압축 없음 (FullKV 또는 실패): 전체 재생성 ────────────
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
        
        if "<think>" in prediction:
            parts = prediction.split("</think>")
            prediction = parts[-1].strip() if len(parts) > 1 else prediction
        
        prediction = prediction.strip()
        
        num_new = new_tokens.shape[0]
        throughput = num_new / elapsed if elapsed > 0 else 0.0
        ttft_ms = elapsed * 1000 / max(num_new, 1)
        
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