
epoch=30

df="/kaggle/input/train_series_alldata_skffold.parquet"
python src/exp/run_exp.py --exp_category earlysave \
                        --exp_name exp030_lstm_enchead_enckernelto384_removekeys_trnvld_epoch${epoch} \
                        --folds 0 1 2 3 4 \
                        --enc-kernelsize-list 12 24 48 96 192 384 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --series_df $df \
                        --input_channels 6 \
                        --model_type dense_lstm_enc_head \
                        --lr 0.001

wandb sync --sync-all

# df="/kaggle/input/train_series_alldata_halflabel_fold.parquet"
# python src/exp/run_exp.py --exp_category earlysave \
#                         --exp_name exp029_lstm_enchead_enckernelto192_halflabel_epoch${epoch} \
#                         --folds 0 1 2 3 4 \
#                         --enc-kernelsize-list 12 24 48 96 192 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type dense_lstm_enc_head \
#                         --lr 0.001

# wandb sync --sync-all
