export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python od_main.py \
    --dataset lvis \
    --dataset_root data/coco2017 \
    \
    --num_epochs 60 \
    --batch_size 6 \
    --patience_ratio 0.1 \
    --eval_interval_ratio 0.01 \
    --accumulation_steps 1 \
    --train_subset_ratio 1 \
    \
    --lr_min_ratio 1e-8 \
    --warmup_ratio 0.05 \
    \
    --optimizer GSAM \
    --learning_rate 0.001 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --rho 0.05 \
    --gsam_alpha 0.02 \
    --gsam_lr_scheduler cosine \
    --gsam_rho_scheduler cosine \
    --rho_min_ratio 1e-8 \




