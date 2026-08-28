import sys, os, time, gc
sys.path.insert(0, os.getcwd())

import torch
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2, compute_score
from core.kv_cache_hook import make_hook_cache, BaseHookCache
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EvaluatorV2FixedEOS(EvaluatorV2):
    def evaluate_sample(self, sample, method_name, budget_ratio, measure_efficiency=True, method_kwargs=None):
        self._clear_cache()
        prompt = make_prompt(
            model_key=self.model_key, tokenizer=self.tokenizer,
            context=sample["context"], question=sample["question"],
            task_type=sample["task_type"],
        )
        model_max_len = self.cfg.get("max_length", 16000)
        safe_max_input = min(16000, model_max_len - 1000)
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
            raw_eos = self.model.generation_config.eos_token_id
            if raw_eos is None:
                raw_eos = self.tokenizer.eos_token_id
            eos_ids = set(raw_eos) if isinstance(raw_eos, (list, tuple)) else {raw_eos}

            max_new = sample["max_new_tokens"]
            cache_len = compressed_cache.layers[0].keys.shape[2]
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            generated_ids = [next_token.item()]
            cur_pos = input_length
            step = 0
            while step < max_new - 1 and generated_ids[-1] not in eos_ids:
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
                step += 1
            new_tokens = torch.tensor(generated_ids, device=self.device)
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
        logger.info(f"[{method_name}] pred={prediction[:150]!r}")
        score = compute_score(prediction, sample["answers"], sample["metric"])
        return {
            "score": score, "prediction": prediction, "ttft_ms": ttft_ms,
            "throughput": throughput, "kv_size_before_mb": kv_before,
            "kv_size_after_mb": kv_after, "memory_reduction_pct": mem_red,
        }


def main():
    MODEL_KEY = "gemma-2-2b"
    TASK = "qmsum"
    NUM_SAMPLES = 3
    BUDGET_RATIO = 0.20

    model, tokenizer, model_config = load_model_and_tokenizer(MODEL_KEY)
    print(f"model.generation_config.eos_token_id = {model.generation_config.eos_token_id}")
    print(f"tokenizer.eos_token_id = {tokenizer.eos_token_id}")

    evaluator = EvaluatorV2FixedEOS(model, tokenizer, model_config)
    samples = load_longbench_task(TASK, num_samples=NUM_SAMPLES, seed=42)

    for method_name, label in [("fullkv", "FullKV"), ("adakv", "AdaKV"), ("ours", "SalienceKV-Sink4")]:
        print(f"\n--- {label} (EOS 수정판) ---")
        method_kwargs = {"sink_size": 4} if method_name == "ours" else {}
        for i, sample in enumerate(samples):
            r = evaluator.evaluate_sample(
                sample, method_name, BUDGET_RATIO,
                measure_efficiency=False, method_kwargs=method_kwargs,
            )
            print(f"[{i}] score={r['score']:.2f} pred={r['prediction'][:150]!r}")


if __name__ == "__main__":
    main()
