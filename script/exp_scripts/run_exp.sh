

python src/exp/run_exp.py --exp_category build_baseline_alldata \
                        --exp_name baseline_alldata_allfold_000 \
                        --model_type add_rolldiff \
                        --input_channels 4 \
                        --folds 0 1 2 3 4 \
                        --n_epoch 10 \
                        --T_0 10

wandb sync --sync-all
