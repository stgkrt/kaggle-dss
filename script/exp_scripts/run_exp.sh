
epoch=10

# oof_df_path="/kaggle/working/_oof/exp008_meanstd_nonavepool/oof_df_secondstg.parquet"
# for epoch in 50 30 20 10
#     do
#     python src/exp/run_exp.py --exp_category secondstage_baseline \
#                             --exp_name secondstg_making_epoch${epoch} \
#                             --model_type event_detect \
#                             --input_channels 3 \
#                             --output_channels 2 \
#                             --series_df $oof_df_path \
#                             --folds 0  \
#                             --n_epoch $epoch \
#                             --T_0 1 \
#                             --lr 0.001
#     done

# python src/exp/run_exp.py --exp_category baseline \
#                         --exp_name exp010_meanstd_norminput_div \
#                         --model_type mean_stds \
#                         --input_channels 14 \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001

# python src/exp/run_exp.py --exp_category baseline_fold0 \
#                         --exp_name exp010_meanstd_norminput_fold0 \
#                         --input_channels 14 \
#                         --model_type mean_stds \
#                         --folds 0 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001

# df="/kaggle/input/downsample_train_series_fold_zerosec.parquet"
# python src/exp/run_exp.py --exp_category baseline \
#                         --exp_name exp011_downsample_norm \
#                         --folds 0 1 2 3 4\
#                         --series_df $df \
#                         --input_channels 2 \
#                         --model_type downsample \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001

df="/kaggle/input/targetdownsample_train_series_fold.parquet"
python src/exp/run_exp.py --exp_category baseline \
                        --exp_name exp012_targetdownsample \
                        --folds 0 1 2 3 4\
                        --series_df $df \
                        --input_channels 10 \
                        --model_type target_downsample \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --lr 0.001


wandb sync --sync-all
