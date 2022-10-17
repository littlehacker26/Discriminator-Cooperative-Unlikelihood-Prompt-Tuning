gpu=0
batch_size=200
file_name='eval'
out_dir='../sentiment_model/small'
model_name_or_path="/home/zhanghanqing/pretrained_model/gpt2/small"
mode="train"
data_path="../datasets/pos_neg/"
task_name="sentiment"
template="(2,3)"
tuning_name="disc_tuning"
epoch=30
use_lm_finetune=True


CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --file_name $file_name --out_dir $out_dir --epoch $epoch --model_name_or_path $model_name_or_path --mode $mode --data_path=$data_path --template $template  --tuning_name $tuning_name --use_lm_finetune $use_lm_finetune 





