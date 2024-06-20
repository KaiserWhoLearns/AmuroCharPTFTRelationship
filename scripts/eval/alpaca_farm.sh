# Please make sure OPENAI_API_KEY is set in your environment variables

# Use V1 of alpaca farm evaluation.
export IS_ALPACA_EVAL_2=False
export OPENAI_API_KEY=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

# use vllm for generation
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --save_dir results/alpaca_farm/tulu_v1_7B/ \
#     --eval_batch_size 20 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

export model_path="${base_dir}/output/olmo1b_lima_15epoch_2e-6/"
# use normal huggingface generation function
python -m eval.alpaca_farm.run_eval \
    --model_name_or_path $model_path \
    --save_dir ${base_dir}/output/eval_res/alpaca_farm_tulu/ \
    --eval_batch_size 20 \
    --max_new_tokens 300 \
    --use_chat_format \
    --load_in_8bit
    # --reference_path ${base_dir}/data/eval/alpaca_farm/davinci_003_outputs.json \
    