# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0
# export model_path="${base_dir}/output/hp_tuning/olmo1b_tulu_2e-6"
lr=2e-7
export model_path="${base_dir}/output/olmo1b_main_tulu_5epoch_${lr}"

# Evaluating llama 7B model using chain-of-thought
python -m eval.gsm.run_eval \
    --data_dir $data_dir \
    --max_num_examples 200 \
    --save_dir $base_dir/results/gsm/olmo1b_tulu_${lr}_5epoch \
    --model $model_path \
    --tokenizer $model_path \
    --n_shot 8 


# Evaluating llama 7B model using direct answering (no chain-of-thought)
python -m eval.gsm.run_eval \
    --data_dir $data_dir \
    --max_num_examples 200 \
    --save_dir $base_dir/results/gsm/olmo1b_tulu_${lr}_5epoch_no_cot \
    --model $model_path \
    --tokenizer $model_path \
    --n_shot 8 \
    --no_cot


# Evaluating tulu 7B model using chain-of-thought and chat format
python -m eval.gsm.run_eval \
    --data_dir $data_dir \
    --max_num_examples 200 \
    --save_dir $base_dir/results/gsm/olmo1b_tulu_${lr}_5epoch_cot_chat \
    --model $model_path \
    --tokenizer $model_path \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
