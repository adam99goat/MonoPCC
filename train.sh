# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --data_path /nas/zhouying/AF-SfMLearner/endovis_data/scared_mono --num_epochs 20 --batch_size 24 \
#     --log_dir endovis_log_150_train_val_files \
#     --train_files train_val --trainer trainer_feature_frozen_vgg \
#     --model_name feature_loss_Res50_weight1e-3_align_corners_no_normal \
#     --split endovis --dataset endovis \
#     --png \
#     --max_depth 150 --height 256 --width 320 --learning_rate 5e-5 #\
#     # --load_weights_folder /home/zhouying/MonoViT/endovis_log_150_train_files/feature_loss_Res50_weight1e-3_align_corners/best_rm_model
#     # --load_weights_folder /home/zhouying/MonoViT/endovis_log_150_train_more_files/feature_loss_resnet_weight1e-3_align_corners/best_sq_model
#     # --load_weights_folder /home/zhouying/MonoViT/endovis_log_150_train_files/feature_loss_Res50_weight1e-3_4/best_sq_model
#     # --load_weights_folder /home/zhouying/MonoViT/endovis_log_150_train_more_files/feature_loss_resnet_weight1e-3/best_sq_model
#     # /home/zhouying/MonoViT/endovis_log_150_train_more_files/feature_loss_SAM_weight1e-3/best_sq_model
#     # /home/zhouying/MonoViT/endovis_log_150_train_more_files/mono_model/best_sq_model #/home/zhouying/MonoViT/endovis_log_150_3/mono_model/models/weights_14 #/home/zhouying/MonoViT/endovis_log_150_more_train+val/mono_model/best_sq_model
#      #/home/zhouying/MonoViT/endovis_log_step_best_model_4/mono_model/best_sq_model_from_A60001

CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_path /nas/zhouying/AF-SfMLearner/endovis_data/scared_mono --num_epochs 20 \
    --log_dir endovis_log_150_demo_demo_demo \
    --train_files train_more --trainer trainer_feature_forward_loss_frozen_EMA_bilateral_mix_style \
    --model_name from_train_more_files_feature_loss_Res50_weight1e-3_alpha_025 \
    --split endovis --dataset endovis \
    --png \
    --max_depth 150 --height 256 --width 320 --learning_rate 5e-5 \
    --load_weights_folder /home/zhouying/MonoViT/endovis_log_150_train_more_files/feature_loss_resnet_weight1e-3/best_sq_model

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --data_path /nas/zhouying/AF-SfMLearner/endovis_data/scared_mono --num_epochs 10 \
#     --log_dir endovis_log_150_feature_forward_loss_frozen_bilateral_mix_style \
#     --train_files train_val --trainer trainer_feature_forward_loss_frozen_bilateral_mix_style \
#     --model_name from_train_val_files_feature_loss_Res50_weight1e-3_align_corners_sq \
#     --split endovis --dataset endovis \
#     --png \
#     --max_depth 150 --height 256 --width 320 --learning_rate 5e-5 \
#     --load_weights_folder /home/zhouying/MonoViT/endovis_log_150_train_val_files/feature_loss_Res50_weight1e-3_align_corners/best_sq_model
















# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --data_path /nas/zhouying/AF-SfMLearner/endovis_data/scared_mono \
#     --log_dir endovis_log_mpvit_small_pose_2 \
#     --model_name mono_model \
#     --split endovis --dataset endovis \
#     --png \
#     --max_depth 150 --height 256 --width 320 --learning_rate 5e-5 #\
#     # --load_weights_folder /home/zhouying/AF-SfMLearner/Model_MIA_2stages \
#     # --models_to_load "pose_encoder" "pose"


# if awk 'NR==3 {val=substr($0, 16, 5)+0; if(val>0.402) exit 0; else exit 1}' /home/zhouying/MonoViT/endovis_log_150_demo/feature_loss_resnet_weight1e-3_2/best_sq_model/result.txt
# then
#     echo "1"
# else
#     echo "2"
# fi