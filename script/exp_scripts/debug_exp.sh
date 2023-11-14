df="/kaggle/input/train_series_alldata_skffold.parquet"
python src/exp/run_exp.py --exp_name debug \
                        --folds 0 \
                        --n_epoch 2 \
                        --T_0 2 \
                        --series_df $df \
                        --input_channels 6 \
                        --model_type dense_lstm_enc_head \
                        --lr 0.001
                                    
