
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

# df="/kaggle/input/targetdownsample_train_series_fold.parquet"
# python src/exp/run_exp.py --exp_category baseline \
#                         --exp_name exp013_td_ch6_epoch${epoch} \
#                         --folds 0 1 2 3 4 \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type target_downsample \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001


# df="/kaggle/input/targetdownsample_train_series_fold.parquet"
# python src/exp/run_exp.py --exp_category baseline_fold0 \
#                         --exp_name exp017_inputtargettd_flip_epoch${epoch}_lr05_fold0 \
#                         --folds 0 \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type input_target_downsample \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.0005

# df="/kaggle/input/targetdownsample_train_series_hour_fold.parquet"
# python src/exp/run_exp.py --exp_category baseline \
#                         --exp_name exp018_inputtargetd_hour_notflip_epoch${epoch} \
#                         --folds 0 1 2 3 4 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --series_df $df \
#                         --input_channels 8 \
#                         --model_type input_target_downsample \
#                         --lr 0.001
    
# df="/kaggle/input/targetdownsample_train_series_hour_fold.parquet"
# python src/exp/run_exp.py --exp_category baseline_fold0 \
#                         --exp_name exp019_deriv_meanstd_epoch${epoch}_fold0 \
#                         --folds 0 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --series_df $df \
#                         --input_channels 8 \
#                         --model_type input_target_downsample_dt \
#                         --lr 0.001

df="/kaggle/input/targetdownsample_train_series_skffold.parquet"
# df="/kaggle/input/targetdownsample_train_series_hour_fold.parquet"
python src/exp/run_exp.py --exp_category earlysave \
                        --exp_name exp020_dense_chh_skffold_epoch${epoch} \
                        --folds 0 1 2 3 4 \
                        --n_epoch $epoch \
                        --T_0 $epoch \
                        --series_df $df \
                        --input_channels 6 \
                        --model_type input_target_downsample_dense \
                        --lr 0.001


# df="/kaggle/input/targetdownsample_train_series_fold.parquet"
# df="/kaggle/input/targetdownsample_train_series_3ch_fold.parquet"
# python src/exp/run_exp.py --exp_category baseline_fold0 \
#                         --exp_name exp016_inputtargetd3ch_bcewithlogits_epoch${epoch}_fold0 \
#                         --folds 0 \
#                         --series_df $df \
#                         --input_channels 6 \
#                         --model_type input_target_downsample_3ch \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --lr 0.001


# df="/kaggle/input/targetdownsample_train_series_fold.parquet"
# python src/exp/run_exp.py --exp_category baseline_fold0 \
#                         --exp_name exp014_td_event_bcemax_fold0 \
#                         --folds 0 \
#                         --n_epoch $epoch \
#                         --T_0 $epoch \
#                         --series_df $df \
#                         --input_channels 10 \
#                         --model_type target_downsample_event \
#                         --lr 0.001

wandb sync --sync-all
