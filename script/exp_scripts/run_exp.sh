
epoch=30

df="/kaggle/input/train_series_alldata_skffold.parquet"
python src/exp/run_exp.py --exp_category earlysave \
                        --exp_name exp025_alldata_skf_lstmenchead_epoch${epoch} \
                        --folds 0 1 2 3 4 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --series_df $df \
                        --input_channels 6 \
                        --model_type dense_lstm_enc_head \
                        --lr 0.001

# epoch=10
# df="/kaggle/input/train_series_alldata_skffold.parquet"
# python src/exp/run_exp.py --exp_category earlysave \
#                         --exp_name exp023_alldata_skf_dense_epoch${epoch}_lr5e-4 \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type dense2ch \
#                         --lr 0.0005

# python src/exp/run_exp.py --exp_category earlysave \
#                         --exp_name exp023_alldata_skf_dense_epoch${epoch}_lr75e-4 \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type dense2ch \
#                         --lr 0.00075

wandb sync --sync-all
