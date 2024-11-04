#!/bin/bash

lr=2e-5
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config=configs/data/ft_laion_aes_1m.yaml

exp_name=7B
mkdir -p workdir/"$exp_name"


python -u finetune_solver.py \
--model_size 7B \
--batch_size 8 \
--accum_iter 1 \
--epochs 2 \
--warmup_epochs 0.01 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--clip_grad 4 \
--data_config $data_config \
--cache_ann_on_disk \
--num_workers 8 \
--output_dir workdir/"$exp_name" \
--save_iteration_interval 1000 \
--checkpointing \
--max_seq_len 4096 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
2>&1 | tee -a workdir/"$exp_name"/output.log

echo "exp name: $exp_name"
