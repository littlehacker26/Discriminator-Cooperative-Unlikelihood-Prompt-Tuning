gpu=0
temperature=1.0
batch_size=100
file_name='../eval'
prompt_type="negative"
target_type="positive"

model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
data_path="../datasets/sentiment_prompts-10k/"


embedding_checkpoint="../check_point/sentiment/vanilla_prompt_tuning_negative_(5,5).ckpt"
embedding_checkpoint="../check_point/sentiment/vanilla_prompt_tuning_positive_(5,5).ckpt"


template="(5,5)"
beta=0.999
tuning_name="prompt_tuning"


mode="ctg"
iter_num=20
top_p=1.0

for ranking_scope in 10
do 
    echo  ---ranking_scope--$ranking_scope---------
    
    CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size --ranking_scope $ranking_scope --file_name $file_name  --prompt_type $prompt_type --target_type $target_type --model_name_or_path $model_name_or_path  --embedding_checkpoint $embedding_checkpoint --iter_num $iter_num --top_p $top_p --beta $beta --template $template --tuning_name $tuning_name --data_path $data_path
    wait

done