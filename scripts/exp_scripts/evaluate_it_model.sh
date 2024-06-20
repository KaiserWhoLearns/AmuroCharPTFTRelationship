#!/bin/bash
export base_dir=""
export data_dir=""
conda activate eval-pipeline

export model_name=checkpoint-738000
export model_path="${base_dir}/output/olmo1b_tf_ckpts/${model_name}"

for data_name in "openbookqa" "hellaswag" "boolq" "sciq" "arc_challenge"  "arc_easy"
do
        export num_shots=4

        if [[ $num_shots == 0 ]]; then
                export output_file_name="${data_file_name}_${model_name}"
        else
                export output_file_name="${data_file_name}_${model_name}_${num_shots}shots"
        fi

        export input_file_name="${data_dir}/evaluation/${data_file_name}.jsonl"
        export uncons_output_file="${base_dir}/output/predictions/${uncons_output_file_name}.jsonl"
        export output_file="${base_dir}/output/predictions/${output_file_name}.jsonl"

        python -m olmo_eval.run_lm_eval --model-path ${model_path} \
            --task $data_name --split validation \
            --num-shots $num_shots \
            --metrics-file ${base_dir}/eval_results/eval_pipeline/${data_name}_${model_name}.json


done