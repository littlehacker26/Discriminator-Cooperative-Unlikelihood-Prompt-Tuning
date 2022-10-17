gpu=3
temperature=1.0
batch_size=300
model_name_or_path="/home/zhanghanqing/pretrained_model/gpt2/large"
evaluate_outfile="../eval/detoxic_temp/detoxic_result.csv"
mode="classifer"
task_name="detoxic"
evaluate_file="../eval/detoxic_temp"

disc_embedding_checkpoint="../detoxic_model/small/prompt_model/prompt_tuning_positive_lr_0.3_temperature0.01_scope_50_epoch_26_f1_0.81_(2,3).ckpt"
template_disc="(2,3)"

CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --model_name_or_path $model_name_or_path  --evaluate_file $evaluate_file  --evaluate_outfile $evaluate_outfile --mode $mode --disc_embedding_checkpoint $disc_embedding_checkpoint --task_name $task_name --template_disc $template_disc