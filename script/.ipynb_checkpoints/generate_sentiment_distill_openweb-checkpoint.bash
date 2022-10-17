gpu=1
temperature=1.0
batch_size=100
file_name='../eval'
prompt_type="neutral"
target_type="negative"
model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
data_path="../datasets/sentiment_prompts-10k/"

# embedding_checkpoint="../sentiment_model/large/prompt_model/epoch_6_f1_0.94_(1,2).ckpt"
embedding_checkpoint="../sentiment_model/openweb10000/prompt_model/distill_tuning_negative_temperature0.01_scope_70_epoch_7_f1_407.95_(5,5).ckpt"
# distill_tuning_positive_lr_0.1_temperature0.01_scope_30_epoch_3_f1_548.31_(8,8).ckpt


template="(5,5)"
beta=0.2
tuning_name="distill_tuning"


mode="ctg"
iter_num=20
top_p=1.0

#21 20 19 18 17 16 15 14 13 12
for ranking_scope in 10
do 
    echo  ---ranking_scope--$ranking_scope---------
    
     CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size --ranking_scope $ranking_scope --file_name $file_name  --prompt_type $prompt_type --target_type $target_type --model_name_or_path $model_name_or_path  --embedding_checkpoint $embedding_checkpoint --iter_num $iter_num --top_p $top_p --beta $beta --template $template --tuning_name $tuning_name --data_path $data_path 
    wait

done


