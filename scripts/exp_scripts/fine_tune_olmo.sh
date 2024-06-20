#!/bin/bash
export base_dir=""
export data_dir=""

export data_file_name=xsum
export ckpt=592000
export model_name=${base_dir}/output/olmo1b_tf_ckpts/checkpoint-${ckpt}
export lr=2e-7
export epoch=3
export output_folder_name=olmo1b_hf_ckpt${ckpt}_${data_file_name}_${epoch}epoch_${lr}
export exp_name=ft_${output_folder_name}

export train_file=${data_dir}/${data_file_name}.jsonl
export output=output/${output_folder_name}/

export BNB_CUDA_VERSION=118
export HF_HOME=/scratch4/mdredze1/huggingface_cache/
export TRANSFORMERS_CACHE=/scratch4/mdredze1/huggingface_cache/transformers/
export HF_DATASETS_CACHE=/scratch4/mdredze1/huggingface_cache/datasets/

echo "Running with BASE_DIR=${base_dir}"
export MODEL_SIZE=1B
export NUM_GPUS=1
export BATCH_SIZE_PER_GPU=1
export TOTAL_BATCH_SIZE=8
export GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

conda activate eval-pipeline
cd $base_dir

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    open_instruct/finetune.py \
    --model_name_or_path ${model_name} \
    --train_file ${train_file} \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${lr} \
    --config_name ${model_name} \
    --tokenizer_name ${model_name} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $epoch \
    --output_dir $output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 10
