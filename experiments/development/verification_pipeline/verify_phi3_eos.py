import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer

model, tok, cfg = load_model_and_tokenizer("phi-3-mini")

print("tokenizer.eos_token:", repr(tok.eos_token), "| id:", tok.eos_token_id)
print("model.generation_config.eos_token_id:", model.generation_config.eos_token_id)
print("model.config.eos_token_id (있다면):", getattr(model.config, "eos_token_id", "없음"))

# 챗 템플릿 적용 시 실제로 어떤 종료 토큰이 프롬프트에 쓰이는지 확인
messages = [{"role": "user", "content": "테스트"}]
formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\nchat_template 적용 결과:")
print(repr(formatted))
