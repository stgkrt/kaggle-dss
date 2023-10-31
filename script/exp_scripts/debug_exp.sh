
# python src/exp/run_exp.py --exp_name debug \
#                         --folds 0 \
#                         --n_epoch 1 \
#                         --input_channels 2 \
#                         --model_type normal

# python src/exp/run_exp.py --exp_name debug \
#                         --folds 0 \
#                         --n_epoch 1 \
#                         --input_channels 14 \
#                         --model_type mean_stds \
#                         --folds 0 \
#                         --lr 0.001

df="/kaggle/input/downsample_train_series_fold_zerosec.parquet"
python src/exp/run_exp.py --exp_name debug \
                        --folds 0 \
                        --n_epoch 1 \
                        --series_df $df \
                        --input_channels 2 \
                        --model_type down_sample \
                        --folds 0 \
                        --lr 0.001

# python src/exp/run_exp.py  --train-mode pseudo \
#                             --exp_name pseudo_debug

# oof_df_path="/kaggle/working/_oof/exp008_meanstd_nonavepool/oof_df_secondstg.parquet"
# python src/exp/run_exp.py --exp_category debug_secondstage \
#                         --exp_name debug_secondstg \
#                         --model_type event_detect \
#                         --input_channels 3 \
#                         --output_channels 2 \
#                         --series_df $oof_df_path \
#                         --folds 0  \
#                         --n_epoch 5 \
#                         --T_0 1 \
#                         --lr 0.001
