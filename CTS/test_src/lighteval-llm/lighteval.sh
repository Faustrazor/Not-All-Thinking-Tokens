MODEL=/data1/users/yuanh/saves/Llama-3.1-8B-Instruct/sft_full/math220k_first_train_4096
NUM_GPUS=8
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
target=/data0/users/yuanh/lighteval-llm/llama8B/math_first_train

OUTPUT_DIR=$target/math500
TASK=math_500
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/eval/eval_math500.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

OUTPUT_DIR=$target/aime24
TASK=aime24
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/eval/eval_aime.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

OUTPUT_DIR=$target/gpqa
TASK=gpqa:diamond
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/eval/eval_gpqa.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details