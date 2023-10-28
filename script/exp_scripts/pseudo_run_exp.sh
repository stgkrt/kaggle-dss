epoch=3
python src/exp/run_exp.py --exp_category pseudo_making \
                        --exp_name pseudo_exp008_002_epoch${epoch} \
                        --train-mode pseudo \
                        --pseudo_weight_exp exp008_meanstd_nonavepool \
                        --model_type mean_stds \
                        --input_channels 14 \
                        --folds 0 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --lr 0.0005
wandb sync --sync-all
