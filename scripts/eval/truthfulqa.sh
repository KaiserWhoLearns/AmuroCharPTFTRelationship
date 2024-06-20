# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating llama 7B model, getting the truth and info scores and multiple choice accuracy

# To get the truth and info scores, the original TruthfulQA paper trained 2 judge models based on GPT3 curie engine.
# If you have or trained those judges, you can specify the `gpt_truth_model_name`` and `gpt_info_model_name`,
# which are the names of the GPT models trained following https://github.com/sylinrl/TruthfulQA#fine-tuning-gpt-3-for-evaluation
# But recently Openai has deprecated the GPT3 curie engine, so we also provide the option to use the HF models as the judges.
# The two models provided here are trained based on the llama2 7B model.
# We have the training details in https://github.com/allenai/truthfulqa_reeval, and verified these two models have similar performance as the original GPT3 judges.

# export model_path="${base_dir}/output/hp_tuning/olmo1b_tulu_2e-6"
# python -m eval.truthfulqa.run_eval \
#     --data_dir $data_dir/eval/truthfulqa \
#     --save_dir $base_dir/results/trutufulqa/hp_tulu_chat_2e-6 \
#     --model_name_or_path $model_path \
#     --tokenizer_name_or_path $model_path \
#     --metrics truth info mc \
#     --preset chat \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format

# export model_path="${base_dir}/output/hp_tuning/olmo1b_tulu_2e-5"
# python -m eval.truthfulqa.run_eval \
#     --data_dir $data_dir/eval/truthfulqa \
#     --save_dir $base_dir/results/trutufulqa/hp_tulu_chat_2e-5 \
#     --model_name_or_path $model_path \
#     --tokenizer_name_or_path $model_path \
#     --metrics truth info mc \
#     --preset chat \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format

### non-chat
export lr=2e-6
export model_path="${base_dir}/output/olmo1b_main_tulu_5epoch_${lr}"
python -m eval.truthfulqa.run_eval \
    --data_dir $data_dir/eval/truthfulqa \
    --save_dir $base_dir/results/trutufulqa/hp_tulu_5epoch_${lr} \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --metrics truth info mc \
    --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 20 \
    --load_in_8bit 

# export model_path="${base_dir}/output/hp_tuning/olmo1b_tulu_2e-7"
python -m eval.truthfulqa.run_eval \
    --data_dir $data_dir/eval/truthfulqa \
    --save_dir $base_dir/results/trutufulqa/hp_tulu_chat_5epoch_${lr} \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --metrics truth info mc \
    --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format

# # # Evaluating Tulu 7B model using chat format, getting the truth and info scores and multiple choice accuracy
# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/tulu2-7B/ \
#     --model_name_or_path ../checkpoints/tulu2/7B/ \
#     --tokenizer_name_or_path ../checkpoints/tulu2/7B/ \
#     --metrics truth info mc \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
    # --use_chat_format \
    # --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model using chat format, getting the truth and info scores and multiple choice accuracy
# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/llama2-chat-7B \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --metrics truth info mc \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt, getting the truth and info scores
# # Multiple choice accuracy is not supported for chatgpt, since we cannot get the probabilities from chatgpt
# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/chatgpt \
#     --openai_engine gpt-3.5-turbo-0301 \
#     --metrics truth info \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20

# # Evaluating gpt-4, getting the truth and info scores
# # Multiple choice accuracy is not supported for gpt-4, since we cannot get the probabilities from gpt-4
# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/gpt4 \
#     --openai_engine gpt-4-0314 \
#     --metrics truth info \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20