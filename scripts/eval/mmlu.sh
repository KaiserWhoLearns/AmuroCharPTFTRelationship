# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

# Change chat formatting function if using chat format using llama or olmo
export lr=2e-6
# export model_path="${base_dir}/output/olmo1b_main_tulu_5epoch_${lr}"
# export model_name=olmo1b_hp_5epoch_${lr}
export model_name=olmo-7b-instruct

export model_path=allenai/OLMo-7B-Instruct


# # cd $base_dir/eval
# # Evaluating llama 7B model using 0 shot directly
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir $data_dir/eval/mmlu \
    --save_dir $base_dir/results/mmlu/${model_name}_1shot_chat \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format

python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir $data_dir/eval/mmlu \
    --save_dir $base_dir/results/mmlu/${model_name}_1shot \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --eval_batch_size 4 \
    --load_in_8bit

# Use chat format
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $data_dir/eval/mmlu \
    --save_dir $base_dir/results/mmlu/${model_name}_5shot_chat \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format

python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $data_dir/eval/mmlu \
    --save_dir $base_dir/results/mmlu/${model_name}_5shot \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --eval_batch_size 4 \
    --load_in_8bit

# # Use Prompt completion format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir $data_dir/eval/mmlu \
#     --save_dir $base_dir/results/mmlu/olmo1b_hp_2e-7_5shot \
#     --model_name_or_path $model_path \
#     --tokenizer_name_or_path $model_path \
#     --eval_batch_size 4 \
#     --load_in_8bit

# # Use chat format
# export model_path="${base_dir}/output/olmo1b_tulu_0.1_data_4epoch_2e-6"
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir $data_dir/eval/mmlu \
#     --save_dir $base_dir/results/mmlu/olmo1b_lima_2e-4 \
#     --model_name_or_path $model_path \
#     --tokenizer_name_or_path $model_path \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format

# # Use Prompt completion format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir $data_dir/eval/mmlu \
#     --save_dir $base_dir/results/mmlu/olmo1b_lima_2e-4 \
#     --model_name_or_path $model_path \
#     --tokenizer_name_or_path $model_path \
#     --eval_batch_size 4 \
#     --load_in_8bit

# # Evaluating llama 7B model using 5 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-5shot \
#     --model_name_or_path  /workspace/open-instruct/llama/hf_llama_models/7B \
#     --tokenizer_name_or_path /workspace/open-instruct/llama/hf_llama_models/7B \
#     --eval_batch_size 4 \
#     --load_in_8bit


# # Evaluating Tulu 7B model using 0 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-0shot \
#     --model_name_or_path ../checkpoints/tulu_7B \
#     --tokenizer_name_or_path ../checkpoints/tulu_7B \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating Tulu 7B model using 5 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-5shot \
#     --model_name_or_path ../checkpoints/tulu_7B \
#     --tokenizer_name_or_path ../checkpoints/tulu_7B \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model using 0-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating llama2 chat model using 5-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating chatgpt using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating gpt4 using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # Evaluating gpt4 using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20