import sys
sys.path.insert(0, "/workspace/kv-cache-exp")
from core.model_loader import load_model_and_tokenizer

model, tok, cfg = load_model_and_tokenizer("qwen3-4b")
print("151645 ->", repr(tok.decode([151645])))
print("151643 ->", repr(tok.decode([151643])))
print("tokenizer.eos_token ->", repr(tok.eos_token), "id:", tok.eos_token_id)

# 챗 템플릿 적용 시 실제로 어떤 토큰이 assistant 턴 종료 마커로 쓰이는지 확인
messages = [{"role": "user", "content": "테스트"}]
formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("\nchat_template 결과:")
print(repr(formatted))
