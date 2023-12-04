
# epoch=15
# df="/kaggle/input/train_series_alldata_skffold.parquet"
# df="/kaggle/input/train_series_alldata_elapseddate_fold.parquet"
# python src/exp/run_exp.py  --exp_category earlysave \
#                          --exp_name exp037_elapseddate_epoch${epoch} \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --enc-kernelsize-list 12 24 48 96 192 384 \
#                         --series_df $df \
#                         --input_channels 7 \
#                         --model_type elapsed_date \
#                         --lr 0.001

# python src/exp/run_exp.py  --exp_category earlysave \
#                          --exp_name exp042_epoch${epoch}_fold0\
#                         --folds 0 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --enc-kernelsize-list 12 24 48 96 192 384 \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type dense_lstm_enc_head \
#                         --lr 0.001
epoch=30
# epoch=10
# epoch=5
df="/kaggle/input/train_series_alldata_duplicate_fold.parquet"
# python src/exp/run_exp.py  --exp_category sleep_onset_wakeup \
#                          --exp_name exp038_epoch${epoch}_fold0_sigma_tuning \
#                         --folds 0 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --enc-kernelsize-list 12 24 48 96 192 384 \
#                         --series_df $df \
#                         --input_channels 7 \
#                         --model_type dense3ch \
#                         --warmup_t 0 \
#                         --lr 0.001

python src/exp/run_exp.py  --exp_category earlysave \
                         --exp_name exp045_randomslide_epoch${epoch} \
                        --folds 0 1 2 3 4 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --enc-kernelsize-list 12 24 48 96 192 384 \
                        --series_df $df \
                        --input_channels 7 \
                        --model_type dense_lstm_enc_head \
                        --warmup_t 2 \
                        --lr 0.001
wandb sync --sync-all
