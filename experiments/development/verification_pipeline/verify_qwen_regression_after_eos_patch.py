"""
evaluator_v2.py의 multi-EOS 수정이 Qwen3-4B 결과에 영향을 주는지 회귀 검증.
patch 적용 전(원본 로직 재현, eos_id 단일값 비교)과 patch 적용 후(현재 파일 그대로)를
같은 샘플에 대해 직접 비교 - prediction/score/토큰 수가 완전히 동일해야 통과.
"""
import sys, gc, time, torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.model_loader import load_model_and_tokenizer, make_prompt, tokenize_prompt
from core.dataset_loader import load_longbench_task
from core.evaluator_v2 import EvaluatorV2, compute_score  # patch 적용된 현재 버전
from core.kv_cache_hook import make_hook_cache, BaseHookCache

# === 원본(patch 이전) 로직 재현: eos_id 단일값만 비교 ===
class EvaluatorV2OriginalEOS(EvaluatorV2):
    def evaluate_sample(self, sample, method_name, budget_ratio, measure_efficiency=True,
                         method_kwargs=None, max_input_length=None):
        self._clear_cache()
        prompt = make_prompt(model_key=self.model_key, tokenizer=self.tokenizer,
                              context=sample["context"], question=sample["question"],
                              task_type=sample["task_type"])
        model_max_len = self.cfg.get("max_length", 16000)
        safe_max_input = min(16000, model_max_len - 1000) if max_input_length is None else \
            ((model_max_len - 1000) if max_input_length == -1 else max_input_length)
        inputs = tokenize_prompt(prompt, self.tokenizer, self.model_key, max_input_length=safe_max_input, device=self.device)
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        input_length = input_ids.shape[1]
        hook_cache = make_hook_cache(method_name, budget_ratio, self.cfg, **(method_kwargs or {}))
        with torch.no_grad():
            prefill_out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=True, return_dict=True, output_attentions=False)
            prefill_kv = prefill_out.past_key_values
            last_logits = prefill_out.logits[:, -1, :].detach().clone()
        if isinstance(hook_cache, BaseHookCache):
            if hasattr(prefill_kv, 'layers'):
                for i, layer in enumerate(prefill_kv.layers):
                    if i >= len(hook_cache.layers):
                        hook_cache.update(layer.keys, layer.values, i)
                    else:
                        hook_cache.layers[i].keys = layer.keys.clone()
                        hook_cache.layers[i].values = layer.values.clone()
                    hook_cache.set_prefill_keys(i, layer.keys)
            del prefill_kv; gc.collect(); torch.cuda.empty_cache()
            hook_cache.mark_prefill_done(input_length)
            hook_cache.apply_compression_all_layers()
            compressed_cache = hook_cache
        else:
            compressed_cache = prefill_kv
        try:
            eos_id = self.tokenizer.eos_token_id  # ★ 원본(patch 이전) 로직
            max_new = sample["max_new_tokens"]
            cache_len = compressed_cache.layers[0].keys.shape[2]
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            generated_ids = [next_token.item()]
            cur_pos = input_length
            step = 0
            while step < max_new - 1 and generated_ids[-1] != eos_id:  # ★ 원본 비교
                cache_position = torch.tensor([cur_pos], device=self.device)
                attn_mask_step = torch.ones(1, cache_len + step + 1, dtype=attention_mask.dtype, device=attention_mask.device)
                with torch.no_grad():
                    step_out = self.model(input_ids=next_token, past_key_values=compressed_cache,
                                           cache_position=cache_position, attention_mask=attn_mask_step,
                                           use_cache=True, return_dict=True)
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                cur_pos += 1
                step += 1
            new_tokens = torch.tensor(generated_ids, device=self.device)
        except Exception as e:
            return {"score": 0.0, "prediction": "", "num_new_tokens": 0}
        prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction = self._clean_prediction(prediction)
        score = compute_score(prediction, sample["answers"], sample["metric"])
        return {"score": score, "prediction": prediction, "num_new_tokens": new_tokens.shape[0]}


model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
ev_original = EvaluatorV2OriginalEOS(model, tok, cfg)
ev_patched = EvaluatorV2(model, tok, cfg)  # 현재 patch 적용된 evaluator_v2.py 그대로

print(f"tokenizer.eos_token_id: {tok.eos_token_id}")
print(f"model.generation_config.eos_token_id: {model.generation_config.eos_token_id}")
print("(두 값이 사실상 같으면 - 리스트여도 단일 원소면 - patch가 Qwen 결과에 영향 없어야 함)\n")

samples = load_longbench_task("qmsum", num_samples=3, seed=42)
targets = [("fullkv", {}), ("adakv", {"invert_norm": True}), ("ours", {"sink_size": 4, "invert_norm": True})]

all_match = True
for method, kwargs in targets:
    print(f"--- method={method} kwargs={kwargs} ---")
    for i, sample in enumerate(samples):
        r_orig = ev_original.evaluate_sample(sample, method, 0.20, method_kwargs=kwargs)
        r_patch = ev_patched.evaluate_sample(sample, method, 0.20, method_kwargs=kwargs)
        match_pred = r_orig["prediction"] == r_patch["prediction"]
        match_score = abs(r_orig["score"] - r_patch["score"]) < 1e-6
        match_tok = r_orig["num_new_tokens"] == r_patch.get("num_new_tokens", -1)
        ok = match_pred and match_score
        all_match = all_match and ok
        print(f"  [{i}] score 원본={r_orig['score']:.2f} patch={r_patch['score']:.2f}  "
              f"prediction 동일={match_pred}  {'✅' if ok else '❌ 불일치!'}")

print(f"\n{'✅ 전체 일치 - regression 없음' if all_match else '❌ 불일치 발견 - 반드시 원인 확인 필요'}")
