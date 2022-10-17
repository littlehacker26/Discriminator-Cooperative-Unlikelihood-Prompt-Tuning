gpu=3
temperature=1.0
batch_size=20
file_name='../eval'
prompt_type="negative"
target_type="positive"
model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
data_path="../datasets/sentiment_prompts-10k/"

disc_embedding_checkpoint="../check_point/sentiment/disc_gpt2_small/gpt_small_disc.ckpt"
template="(2,3)"
beta=1.0
max_length=30
tuning_name="disc_tuning"
mode="ctg"
top_p=1.0


#21 20 19 18 17 16 15 14 13 12
for ranking_scope in 100
do 
    echo  ---ranking_scope--$ranking_scope---------
    CUDA_VISIBLE_DEVICES=$gpu  python ../main.py  --batch_size $batch_size --ranking_scope $ranking_scope --file_name $file_name  --prompt_type $prompt_type --target_type $target_type --model_name_or_path $model_name_or_path  --disc_embedding_checkpoint $disc_embedding_checkpoint --top_p $top_p --beta $beta --template $template --max_length  $max_length  --tuning_name $tuning_name  --data_path $data_path
    wait

done