# example scripts for toxigen

# export model_path="${base_dir}/output/olmo1b_lima_15epoch_2e-4"
# # evaluate an open-instruct model with chat format
# python -m eval.toxigen.run_eval \
#     --data_dir ${data_dir}/eval/toxigen/ \
#     --save_dir ${base_dir}/output/eval_res/toxigen_2e-4 \
#     --model_name_or_path ${model_path}
#     # --use_vllm

# export model_path="${base_dir}/output/olmo1b_tulu_0.05_data_4epoch_2e-6"
# python -m eval.toxigen.run_eval \
#     --data_dir ${data_dir}/eval/toxigen/ \
#     --save_dir ${base_dir}/output/eval_res/toxigen_tulu005 \
#     --model_name_or_path ${model_path}

export lr=2e-6
export model_path="${base_dir}/output/olmo1b_main_tulu_5epoch_${lr}"
python -m eval.toxigen.run_eval \
    --data_dir ${data_dir}/eval/toxigen/ \
    --save_dir ${base_dir}/output/eval_res/toxigen_tulu_${lr}_5epoch \
    --model_name_or_path ${model_path}

# export model_path="${base_dir}/output/hp_tuning/olmo1b_tulu_2e-7"
# python -m eval.toxigen.run_eval \
#     --data_dir ${data_dir}/eval/toxigen/ \
#     --save_dir ${base_dir}/output/eval_res/toxigen_tulu_2e-7 \
#     --model_name_or_path ${model_path}

# # evaluate chatGPT
# python -m eval.toxigen.run_eval \
#     --data_dir data/eval/toxigen/ \
#     --save_dir results/toxigen/chatgpt \
#     --openai_engine gpt-3.5-turbo-0301 \
#     --max_prompts_per_group 100 \
#     --eval_batch_size 20


# # evaluate gpt4
# python -m eval.toxigen.run_eval \
#     --data_dir data/eval/toxigen/ \
#     --save_dir results/toxigen/gpt4 \
#     --openai_engine gpt-4-0314 \
#     --max_prompts_per_group 100 \
#     --eval_batch_size 20