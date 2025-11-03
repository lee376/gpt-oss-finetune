# 版本一
在原github链接的基础上，按照gemini的建议，做了以下改动：
1.# --- ⬇️ 在这里添加新代码 ⬇️ ---
peft_model.enable_input_require_grads()
# --- ⬆️ 添加结束 ⬆️ ---
2. 原来是4.4，现在改成1,16

    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
3.
# --- ⬇️ 在这里添加新代码 ⬇️ ---
if tokenizer.pad_token is None:
    print("Tokenizer did not have a pad_token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
# --- ⬆️ 添加结束 ⬆️ ---

4. max_length = 2048,......