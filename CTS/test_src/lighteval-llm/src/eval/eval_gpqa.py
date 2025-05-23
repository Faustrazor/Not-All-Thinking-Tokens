"""
评估 GPQA 数据集
"""

import random

from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import numpy as np
from transformers import AutoTokenizer



########################################################
# prompt_fn
# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
########################################################
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def gpqa_prompt_fn(line: dict, task_name: str = None):
    """
    参数中的 `line` 是 HuggingFace dataset 中的一个 data point，类型是一个 dict
    
    函数返回一个 Doc 对象，作为一次评测的输入
    """
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )


########################################################
# metric：gpqa_metric
# 
# 这段代码的作用是创建一个名为 gpqa_metric 的指标，用于评估模型的输出与参考答案之间的匹配程度。它通过调用 multilingual_extractive_match_metric 函数来实现，传入了一些特定的参数配置。
########################################################
gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


########################################################
# metric：mean_length_metric
########################################################
def compute_sample_length(predictions: list[str], **kwargs) -> float:
    """计算模型 response的长度（以字符为单位）"""
    response = predictions[0]
    return len(response)


def agg_mean_length(items: list[float], **kwargs) -> float:
    flat_items = [item for item in items]
    return float(np.mean(flat_items))


mean_length_metric = SampleLevelMetric(
    metric_name="mean_length",
    higher_is_better=False,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=compute_sample_length,
    corpus_level_fn=agg_mean_length,
)


########################################################
# metric：mean_token_count_metric
########################################################
# 初始化tokenizer，使用Qwen2.5-7B-Instruct模型的tokenizer
_tokenizer = None

def get_tokenizer():
    """获取或创建tokenizer的辅助函数"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("/data0/models/Qwen2.5-7B-Instruct")
    return _tokenizer


def compute_token_count(predictions: list[str], **kwargs) -> float:
    """计算模型 response的token数量"""
    response = predictions[0]
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(response)
    return len(tokens)


def agg_mean_token_count(items: list[float], **kwargs) -> float:
    flat_items = [item for item in items]
    return float(np.mean(flat_items))


mean_token_count_metric = SampleLevelMetric(
    metric_name="mean_token_count",
    higher_is_better=False,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=compute_token_count,
    corpus_level_fn=agg_mean_token_count,
)


########################################################
# task
########################################################
gpqa_diamond_task = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[gpqa_metric, mean_length_metric, mean_token_count_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
)


TASKS_TABLE = [gpqa_diamond_task]


if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
