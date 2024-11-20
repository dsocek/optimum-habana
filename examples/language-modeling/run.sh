#!/bin/bash
python3 run_lora_clm.py \
    --model_name_or_path mistralai/Mistral-Nemo-Instruct-2407 \
    --dataset_name tatsu-lab/alpaca \
    --bf16 True \
    --output_dir ./model_lora_mistral \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm  0.3 \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" "lm_head" \
    --dataset_concatenation \
    --max_seq_length 1024 \
    --validation_split_percentage 4 \
    --adam_epsilon 1e-08

#--deepspeed mistral_ds_zero3_config.json
#--low_cpu_mem_usage True \
#python3 ../gaudi_spawn.py --world_size 8 --use_mpi run_lora_clm.py \

