from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
import numpy as np
from transformers import AutoTokenizer

# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


########################################################
# metric：expr_gold_metric
########################################################
expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
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
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric, mean_length_metric, mean_token_count_metric],
    version=1,
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric, mean_length_metric, mean_token_count_metric],
    version=1,
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)


if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
