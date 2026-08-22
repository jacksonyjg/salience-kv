"""
Phi-3-mini 다중 EOS 토큰([32000,32001,32007]) 미처리 버그 수정 검증.
core/evaluator_v2.py는 건드리지 않고, evaluate_sample() 로직을 그대로 복사한 뒤
eos_id 비교 부분만 리스트(set) 기준으로 바꾼 서브클래스로 검증.
(Gemma-2 EOS 버그 수정 때와 동일한 안전 원칙 - 원본 파일 무변경)
"""
import sys, gc, time, torch
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2, compute_score
from core.kv_cache_hook import make_hook_cache, BaseHookCache
from core.collapse_metrics import is_collapsed, word_repetition_ratio

class EvaluatorV2MultiEOS(EvaluatorV2):
    """evaluate_sample과 완전히 동일, eos_id 비교만 set 멤버십으로 교체."""
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
            safe_max_input = min(16000, model_max_len - 1000)
        else:
            safe_max_input = (model_max_len - 1000) if max_input_length == -1 else max_input_length
        inputs = tokenize_prompt(prompt, self.tokenizer, self.model_key, max_input_length=safe_max_input, device=self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_length = input_ids.shape[1]

        hook_cache = make_hook_cache(method_name, budget_ratio, self.cfg, **(method_kwargs or {}))

        prefill_start = time.perf_counter()
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=True, output_attentions=False,
            )
            prefill_kv = prefill_out.past_key_values
            last_logits = prefill_out.logits[:, -1, :].detach().clone()
        prefill_time_ms = (time.perf_counter() - prefill_start) * 1000
        kv_before = self._kv_size_mb(prefill_kv) if measure_efficiency else 0.0

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
            compressed_cache = prefill_kv
        compress_time_ms = (time.perf_counter() - compress_start) * 1000
        kv_after = 0.0 if not measure_efficiency else self._kv_size_mb(compressed_cache)
        mem_red = (1 - kv_after / kv_before) * 100 if (measure_efficiency and kv_before > 0) else 0.0

        gen_start = time.perf_counter()
        try:
            # ★ 수정 지점: 단일 eos_id 대신 generation_config의 전체 eos 리스트 사용
            raw_eos = self.model.generation_config.eos_token_id
            eos_ids = set(raw_eos if isinstance(raw_eos, (list, tuple)) else [raw_eos])

            max_new = sample["max_new_tokens"]
            cache_len = compressed_cache.layers[0].keys.shape[2]
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            generated_ids = [next_token.item()]
            cur_pos = input_length
            step = 0
            first_decode_step_ms = None
            step_times = []
            # ★ 수정 지점: != eos_id  →  not in eos_ids
            while step < max_new - 1 and generated_ids[-1] not in eos_ids:
                step_start = time.perf_counter()
                cache_position = torch.tensor([cur_pos], device=self.device)
                attn_mask_step = torch.ones(
                    1, cache_len + step + 1,
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                with torch.no_grad():
                    step_out = self.model(
                        input_ids=next_token, past_key_values=compressed_cache,
                        cache_position=cache_position, attention_mask=attn_mask_step,
                        use_cache=True, return_dict=True,
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
            print(f"generate() failed: {e}")
            return {"score": 0.0, "prediction": "", "error": str(e)}

        gen_elapsed = time.perf_counter() - gen_start
        num_new = new_tokens.shape[0]
        prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction = self._clean_prediction(prediction)
        score = compute_score(prediction, sample["answers"], sample["metric"])
        return {"score": score, "prediction": prediction, "num_new_tokens": num_new}


model, tok, cfg = load_model_and_tokenizer("phi-3-mini")
ev = EvaluatorV2MultiEOS(model, tok, cfg)

samples = load_longbench_task("qmsum", num_samples=2, seed=42)
targets = [
    ("fullkv", {}),
    ("adakv", {"invert_norm": False}),  # legacy 방향 - EOS 수정만으로 충분한지, key-norm도 별도 영향인지 분리
    ("adakv", {"invert_norm": True}),
    ("ours",  {"sink_size": 4, "invert_norm": True}),
]

print("=== Phi-3-mini, multi-EOS 수정 후 재검증 ===")
for method, kwargs in targets:
    print(f"\n--- method={method} kwargs={kwargs} ---")
    for i, sample in enumerate(samples):
        r = ev.evaluate_sample(sample, method, 0.20, method_kwargs=kwargs)
        pred = r["prediction"]
        c = is_collapsed(pred)
        wr = word_repetition_ratio(pred)
        print(f"  [{i}] score={r['score']:6.2f}  word_rep={wr:.3f}  collapsed={c}  "
              f"num_new_tokens={r.get('num_new_tokens','?')}  len={len(pred)}")
        print(f"      원문: {pred[:200]!r}")
