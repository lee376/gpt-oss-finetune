# from datasets import load_dataset
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("/public_bme2/bme-wangqian2/lgx2025/gpt-oss-20b") 
# dataset = load_dataset("/home_data/home/ligx2025/tmp/t25_112/multilingual_thinking_local", split="train")
# # dataset = load_dataset("/home_data/home/ligx2025/tmp/t25_112/multilingual_thinking_local/data", split="train")

# messages = dataset[0]["messages"]
# conversation = tokenizer.apply_chat_template(messages, tokenize=False)
# print(conversation)

from datasets import load_dataset

dataset = load_dataset("/home_data/home/ligx2025/tmp/t25_112/multilingual_thinking_local", split="train")

# print(dataset[0].keys())
print(dataset[0]['reasoning_language'])
print('-'* 20)
print(dataset[0]['developer'])
print('-'* 20)
print(dataset[0]['user'])
print('-'* 20)
print(dataset[0]['analysis'])
print('-'* 20)
print(dataset[0]['final'])
print('-'* 20)
print(dataset[0]['messages'])