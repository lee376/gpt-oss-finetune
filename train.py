import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, Trainer, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling # 👈 添加这一行
# import matplotlib.pyplot as plt # 1. 导入绘图库


dataset = load_dataset("/home_data/home/ligx2025/tmp/t25_112/multilingual_thinking_local", split="train")
tokenizer = AutoTokenizer.from_pretrained("/public_bme2/bme-wangqian2/lgx2025/gpt-oss-20b") 

# --- ⬇️ 在这里添加新代码 ⬇️ ---
if tokenizer.pad_token is None:
    print("Tokenizer did not have a pad_token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
# --- ⬆️ 添加结束 ⬆️ ---
    
messages = dataset[0]["messages"]
conversation = tokenizer.apply_chat_template(messages, tokenize=False)
print(conversation)

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained("/public_bme2/bme-wangqian2/lgx2025/gpt-oss-20b", **model_kwargs)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        # MoE 专家层的投影，按需增减
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ]
)
peft_model = get_peft_model(model, peft_config)
# --- ⬇️ 在这里添加新代码 ⬇️ ---
peft_model.enable_input_require_grads()
# --- ⬆️ 添加结束 ⬆️ ---
peft_model.print_trainable_parameters()

# max_length = 4096
max_length = 16

def format_and_tokenize(example):
    # 期望存在 "messages" 字段（和你示例一致）
    messages = example["messages"]
    # 不加 generation_prompt；让模型学习到完整的对话展开
    text = tokenizer.apply_chat_template(
        messages, tokenize=False
    )
    # 直接整体 tokenization，labels=inputs（由 collator 处理）
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    return tokens

tokenized = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
# 简单划个验证集（可选）
splits = tokenized.train_test_split(test_size=0.01, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

# @dataclass
# class CausalDataCollator:
#     tokenizer: AutoTokenizer
#     mlm: bool = False # 这个参数在这里没用，是为MLM模型（如BERT）准备的
#     def __call__(self, features):
#         # 1. 使用默认 collator 将列表中的 feature (字典) 转换成 Pytorch 张量
#         # 比如，把 4 个 [512] 的 input_ids 列表，堆叠成 [4, 512] 的张量
#         batch = default_data_collator(features)
#         # 2. 这是因果语言模型 (Causal LM) 训练的关键
#         if "labels" not in batch:
#             # 3. 将 input_ids 复制一份作为 labels
#             batch["labels"] = batch["input_ids"].clone()
#         return batch
'''
什么是因果语言模型 (Causal LM)？ 模型的目标是预测下一个词。例如，给定 "A B C"，模型需要预测 "B C D"。

为什么 labels = input_ids？ 在 Trainer 内部，模型会自动将 labels 向右平移一位。

输入 (input_ids)： [<bos>, token_A, token_B, token_C]

标签 (labels)： [token_A, token_B, token_C, <eos>] (由 Trainer 自动处理平移和掩码)

作用： 这确保了模型在训练时，是拿 input_ids 去预测 labels，这正是 GPT 这类模型（Causal LM）的标准训练方式。

'''

# collator = CausalDataCollator(tokenizer)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# training_args = TrainingArguments(
#     output_dir="/public_bme2/bme-wangqian2/lgx2025/gpt-oss-20b-multilingual-reasoner",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     num_train_epochs=1.0,
#     learning_rate=2e-4,
#     lr_scheduler_type="cosine_with_min_lr",
#     lr_scheduler_kwargs={"min_lr_rate": 0.1},
#     warmup_ratio=0.03,
#     logging_steps=1,
#     save_steps=200,
#     save_total_limit=2,
#     bf16=True,
#     gradient_checkpointing=True,
#     report_to=[],
# )

training_args = TrainingArguments(
    output_dir="/public_bme2/bme-wangqian2/lgx2025/gpt-oss-20b-multilingual-reasoner",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1.0,
    learning_rate=2e-4,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    warmup_ratio=0.03,
    logging_steps=1,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    report_to=[],
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

# # --- 2. 添加绘图功能 ---

# print("正在绘制 Loss 曲线...")
# # 从 trainer.state.log_history 中提取训练日志
# logs = trainer.state.log_history
# train_logs = [log for log in logs if 'loss' in log] # 筛选训练日志
# eval_logs = [log for log in logs if 'eval_loss' in log] # 筛选评估日志

# # 提取训练 loss 和 steps
# train_steps = [log['step'] for log in train_logs]
# train_losses = [log['loss'] for log in train_logs]

# # 提取评估 loss 和 steps
# eval_steps = [log['step'] for log in eval_logs]
# eval_losses = [log['eval_loss'] for log in eval_logs]

# # --- 绘制图像 ---
# plt.figure(figsize=(10, 6))

# # 绘制训练 Loss
# plt.plot(train_steps, train_losses, label='Training Loss')

# # 绘制评估 Loss
# # 用 'o-' 样式让评估点更清晰
# if eval_steps: # 确保有评估日志
#     plt.plot(eval_steps, eval_losses, 'o-', label='Evaluation Loss', markersize=4)

# plt.title('Training and Evaluation Loss Curve')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True) # 添加网格

# # 3. 保存图像
# output_dir = training_args.output_dir
# plot_path = f"{output_dir}/loss_curve.png"
# plt.savefig(plot_path)

# print(f"Loss 曲线已保存至: {plot_path}")