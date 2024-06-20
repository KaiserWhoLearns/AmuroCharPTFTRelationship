#!/bin/bash
export base_dir=""
export data_dir=""
export num_shots=4
export data_file_name=llmbar_Adversarial_Neighbor
export model_name=olmo1b_hf_ckpt342000_tulu_5epoch_2e-6
export model_path="${base_dir}/output/${model_name}"


export exp_name=pred_${data_file_name}_${model_name}

if [[ $num_shots == 0 ]]; then
        export output_file_name="${data_file_name}_${model_name}"
else
        export output_file_name="${data_file_name}_${model_name}_${num_shots}shots"
fi

export input_file_name="${data_dir}/evaluation/${data_file_name}.jsonl"
export uncons_output_file="${base_dir}/output/predictions/${uncons_output_file_name}.jsonl"
export output_file="${base_dir}/output/predictions/${output_file_name}.jsonl"


cd $base_dir

conda activate eval-pipeline

if [[ $data_file_name == *"nli"* || $data_file_name == *"rte"* || $data_file_name == *"paws"* || $data_file_name == *"stsb"* || $data_file_name == *"qqp"* ]]
then
        echo "Running prediction for classification unconstrained"
        # python -m eval.predict \
        #         --model_name_or_path $model_path \
        #         --input_files $input_file_name \
        #         --num_shots ${num_shots} \
        #         --max_new_tokens 5 \
        #         --output_file $uncons_output_file

        # echo "Running prediction for classification"
        # python -m eval.predict_classification \
        #         --model_name_or_path $model_path \
        #         --input_files $input_file_name \
        #         --num_shots ${num_shots} \
        #         --constraint_type constraint \
        #         --max_new_tokens 5 \
        #         --output_file $output_file
        
        python -m eval.predict_classification \
                --model_name_or_path $model_path \
                --input_files $input_file_name \
                --num_shots ${num_shots} \
                --max_new_tokens 5 \
                --constraint_type next_word \
                --output_file $output_file
elif [[ $data_file_name == *"llmbar"* || $data_file_name == *"instruction"* ]]
then
        echo "Running prediction for generation"
        python -m eval.predict_classification \
                --model_name_or_path $model_path \
                --input_files $input_file_name \
                --max_new_tokens 5 \
                --constraint_type next_word \
                --output_file $output_file \
                --num_shots ${num_shots} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format
else
        echo "Running prediction for generation"
        python -m eval.predict \
                --model_name_or_path $model_path \
                --input_files $input_file_name \
                --max_new_tokens 60 \
                --num_shots ${num_shots} \
                --output_file $output_file
fi
