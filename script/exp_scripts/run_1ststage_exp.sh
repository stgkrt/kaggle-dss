epoch=5
df="/kaggle/input/train_series_alldata_skffold.parquet"
python src/exp/run_1ststage_exp.py --exp_category fisrt_stage_baseline \
                                    --exp_name exp1st000_baseline_epoch${epoch} \
                                    --folds 0 1 2 3 4 \
                                    --n_epoch ${epoch} \
                                    --T_0 ${epoch} \
                                    --series_df $df \
                                    --enc-kernelsize-list 12 24 48 96 192 384 \
                                    --input_channels 2 \
                                    --lr 0.001

                      
