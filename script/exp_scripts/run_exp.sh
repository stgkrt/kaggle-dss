
epoch=10

# python src/exp/run_exp.py --exp_category baseline \
#                         --exp_name exp007_addch \
#                         --input_channels 6 \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001 \

python src/exp/run_exp.py --exp_category baseline_fold0 \
                        --exp_name exp007_addch \
                        --input_channels 6 \
                        --folds 0 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --lr 0.0001

wandb sync --sync-all
