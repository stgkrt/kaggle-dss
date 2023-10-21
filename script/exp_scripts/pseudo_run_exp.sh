
epoch=10
python src/exp/run_exp.py --exp_category pseudo \
                        --exp_name make_pseudo \
                        --input_channels 4 \
                        --folds 0 1 2 3 4 \
                        --n_epoch $epoch \
                        --T_0 $epoch

wandb sync --sync-all
