torchrun --standalone --nnodes=1 --nproc-per-node=8 ./lumina_mgpt/finetune_solver.py  --model_size 7B --batch_size 8 --accum_iter 1 --epochs 2 --warmup_epochs 0.01 --lr 2e-5 --min_lr 2e-5 --wd 0.1 --clip_grad 4 --data_config "./lumina_mgpt/configs/data/ft_laion_aes_1m.yaml" --cache_ann_on_disk --num_workers 8 --output_dir "./workdir/7B-laion-1m" --save_iteration_interval 1000 --checkpointing --max_seq_len 4096 --unmask_image_logits --dropout 0.05 --z_loss_weight 1e-5  2>&1 | tee -a "./workdir/7B-laion-1m/output.log"

torchrun --standalone --nnodes=1 --nproc-per-node=8 ./lumina_mgpt/finetune_solver.py  --model_size 7B --batch_size 8 --accum_iter 1 --epochs 2 --warmup_epochs 0.01 --lr 2e-5 --min_lr 2e-5 --wd 0.1 --clip_grad 4 --data_config "./lumina_mgpt/configs/data/ft_civi.yaml" --cache_ann_on_disk --num_workers 8 --output_dir "./workdir/7B-civi-1m" --save_iteration_interval 1000 --checkpointing --max_seq_len 4096 --unmask_image_logits --dropout 0.05 --z_loss_weight 1e-5  2>&1 | tee -a "./workdir/7B-civi-1m/output.log"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10021 tools/train.py  --launcher pytorch

