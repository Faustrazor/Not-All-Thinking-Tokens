#!/bin/bash

MODEL=/path/to/your/model
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=4,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=results/gpqa

TASK=gpqa:diamond

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/eval/eval_gpqa.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details