

python src/exp/run_exp.py --exp_category build_baseline \
                        --exp_name baseline000 \
                        --n_epoch 10 \
                        --T_0 10

wandb sync --sync-all
