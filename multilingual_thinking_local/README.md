---
viewer: true
dataset_info:
  features:
  - name: reasoning_language
    dtype: string
  - name: developer
    dtype: string
  - name: user
    dtype: string
  - name: analysis
    dtype: string
  - name: final
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
    - name: thinking
      dtype: string
  splits:
  - name: train
    num_bytes: 8900623
    num_examples: 1000
  download_size: 5290171
  dataset_size: 8900623
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: apache-2.0
task_categories:
- text-generation
language:
- en
- de
- fr
- es
- it
pretty_name: Multilingual-Thinking
size_categories:
- 1K<n<10K
---

# Dataset summary

Multilingual-Thinking is a reasoning dataset where the chain-of-thought has been translated from English into one of 4 languages: Spanish, French, Italian, and German. The dataset was created by sampling 1k training samples from the [SystemChat subset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2/viewer/SFT/smoltalk_systemchats_Qwen3_32B_think) of [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) and translating the reasoning traces with another language model. 

This dataset was used in the [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) to fine-tune the OpenAI gpt-oss models.

You can load the dataset using:

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
```

The `gpt-oss` models were trained on the Harmony response format for defining conversation structures, generating reasoning output and structuring function calls. The format is designed to mimic the OpenAI Responses API, and the table below summarizes the different message types used in the dataset:

| `developer` | The developer message is used to provide custom instructions for the model (what we usually call the `system` role) |
| :---- | :--|
| `user` | The user message is used to provide the input to the model |
| `assistant` | Output by the model which can either be a tool call or a message output. The output might also be associated with a particular “channel” identifying what the intent of the message is. |
| `analysis` | These are messages that are being used by the model for its chain-of thought |
| `final` | Messages tagged in the final channel are messages intended to be shown to the end-user and represent the responses from the model. |
| `messages` | The list of messages that combine the content of the above to produce a full conversation. This is the input to the model. |

If you're familiar with [OpenAI's messages format](https://platform.openai.com/docs/api-reference/messages/object), you will recognise this as being quite similar, but with an important difference:

> The `assistant` turn contains two special fields: a `thinking` one which contains the model's reasoning process, and a `content` one which contains the final response to the user.