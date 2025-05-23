from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import os
from tqdm import tqdm
import gc
import torch
import sys

from llmlingua.prompt_compressor import PromptCompressor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=float, help='Compression rate')
args = parser.parse_args()

# /data1/users/yuanh/saves/qwen2.5-7B-Instruct/sft_full/math220k_first_value_4096
# /data0/models/Llama-2-7b-hf
llm_lingua = PromptCompressor(
    model_name="/data1/users/yuanh/saves/qwen2.5-7B-Instruct/sft_full/math220k_first_value_4096"    
)

tokenizer = AutoTokenizer.from_pretrained(
    "/data0/models/Qwen2.5-14B-Instruct/"
)

with open("/data0/users/yuanh/CoT-compression/math220k/math_first_train.json", 'r') as f:
    data = json.load(f)

cutoff_len = 5000
total_tokens = 0
compressed_tokens = 0

for idx, item in tqdm(enumerate(data), total=len(data)):

    thinking = item['answer']

    thinking_ids = tokenizer.encode(thinking)

    if len(thinking_ids) > cutoff_len:
        thinking_ids = thinking_ids[:cutoff_len]

    truncated_thinking = tokenizer.decode(thinking_ids)
    
    # llmlingua的压缩方法
    compressed_prompt = llm_lingua.compress_prompt(
        context=truncated_thinking,
        instruction="",
        concate_question=False,     # Whether to concatenate the question to the compressed prompt. 默认是true       
        rate=args.rate,     
        use_context_level_filter=False, # 启用上下文级压缩，可以考虑分隔Thinking时使用                 
        use_token_level_filter=True,    # 启用token级压缩
        condition_compare=False,         # 启用条件比较
        condition_in_question="none",
        iterative_size=1800,           # 4000比较好
        # force_tokens=['\n', '.' ,':'], # 保留这些标点符号
        force_reserve_digit=True,      # 保留数字
        rank_method="llmlingua"
    )

    total_tokens += compressed_prompt['origin_tokens']
    compressed_tokens += compressed_prompt['compressed_tokens']
    item['lingua_res'] = compressed_prompt['compressed_prompt']

with open(f'/data0/users/yuanh/CoT-compression/math220k_llmlingua_RM/llmlingua_RM_actual_{args.rate}.txt', 'w', encoding='utf-8') as f:
    f.write(f"total_tokens: {total_tokens}\n")
    f.write(f"compressed_tokens: {compressed_tokens}\n")
    f.write(f"compression_rate: {compressed_tokens / total_tokens}\n")


with open(f'/data0/users/yuanh/CoT-compression/math220k_llmlingua_RM/math_llmlingua_RM_{args.rate}.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("The result has been saved")
