df="/kaggle/input/train_series_alldata_skffold.parquet"
python src/exp/run_1ststage_exp.py --exp_name debug \
                                --folds 0 \
                                --n_epoch 2 \
                                --enc-kernelsize-list 12 24 48 96 192 384 \
                                --T_0 2 \
                                --series_df $df \
                                --input_channels 2 \
                                --lr 0.001

                      
