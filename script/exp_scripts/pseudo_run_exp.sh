
epoch=1
python src/exp/run_exp.py --exp_category pseudo_making \
                        --exp_name pseudo_exp006_000 \
                        --train-mode pseudo \
                        --pseudo_weight_exp exp006_addlayer \
                        --folds 0 1 2 3 4 \
                        --n_epoch $epoch \
                        --T_0 $epoch

wandb sync --sync-all
