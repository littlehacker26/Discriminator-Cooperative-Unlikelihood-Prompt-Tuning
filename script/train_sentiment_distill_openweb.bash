gpu=0
top_p=1.0
temperature=0.01
batch_size=150
file_name='eval'
out_dir='../sentiment_model/openweb10000'
model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
mode="train"
data_path="../datasets/openweb10000/"
task_name="sentiment"
template="(5,5)"
corpus_type="negative"
tuning_name="distill_tuning"
epoch=10
ranking_scope=70
disc_embedding_checkpoint="../check_point/sentiment/disc_gpt2_small/gpt_small_disc.ckpt"
template_disc="(2,3)"

for template in "(5,5)" 
do
    echo  ---epoch--$epoch---------
    echo  ---template--$template---------
    CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --file_name $file_name --out_dir $out_dir --epoch $epoch --model_name_or_path $model_name_or_path --mode $mode --data_path=$data_path --template $template --corpus_type $corpus_type  --tuning_name $tuning_name --disc_embedding_checkpoint $disc_embedding_checkpoint --template_disc $template_disc  --ranking_scope $ranking_scope --top_p $top_p  --temperature $temperature
    wait
done