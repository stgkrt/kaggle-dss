
epoch=10
python src/exp/run_exp.py --exp_category baseline \
                        --exp_name exp009_normal \
                        --model_type normal \
                        --input_channels 2 \
                        --folds 0 1 2 3 4 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --lr 0.001

# python src/exp/run_exp.py --exp_category baseline \
#                         --exp_name exp008_meanstd_nonavepool \
#                         --model_type mean_stds \
#                         --input_channels 14 \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001

# python src/exp/run_exp.py --exp_category baseline_fold0 \
#                         --exp_name exp008_meanstd_noavepool_fold0 \
#                         --input_channels 14 \
#                         --model_type mean_stds \
#                         --folds 0 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001

wandb sync --sync-all
