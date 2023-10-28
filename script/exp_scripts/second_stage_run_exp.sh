
epoch=10

oof_df_path="/kaggle/working/_oof/exp006_addlayer/oof_df.parquet"
python src/exp/run_exp.py --exp_category baseline_secondstage \
                        --exp_name secondstg_exp000 \
                        --model_type event_det \
                        --input_channels 3 \
                        --folds 0 1 2 3 4 \
                        --series_df $oof_df_path \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --lr 0.001

wandb sync --sync-all
