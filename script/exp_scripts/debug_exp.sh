
python src/exp/run_exp.py --exp_name debug \
                        --folds 0 \
                        --n_epoch 1 \
                        --input_channels 2 \
                        --model_type normal


# python src/exp/run_exp.py  --train-mode pseudo \
#                             --exp_name pseudo_debug

# oof_df_path="/kaggle/working/_oof/exp006_addlayer/oof_df.parquet"
# python src/exp/run_exp.py --exp_category debug_secondstage \
#                         --exp_name debug_secondstg \
#                         --model_type event_detect \
#                         --input_channels 3 \
#                         --output_channels 2 \
#                         --series_df $oof_df_path \
#                         --folds 0  \
#                         --n_epoch 10 \
#                         --T_0 1 \
#                         --lr 0.001
