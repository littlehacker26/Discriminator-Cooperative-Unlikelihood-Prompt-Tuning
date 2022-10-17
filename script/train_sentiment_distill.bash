gpu=1
top_p=1.0
temperature=0.01
batch_size=100
file_name='eval'
out_dir='../sentiment_model/gen'
model_name_or_path="/home/zhanghanqing/pretrained_model/gpt2/large"
mode="train"
data_path="../datasets/pos_neg/"
task_name="sentiment"
template="(6,6)"
corpus_type="negative"
tuning_name="distill_tuning"
epoch=7
ranking_scope=110
disc_embedding_checkpoint="../sentiment_model/small/prompt_model/prompt_tuning_positive_lr_0.3_temperature0.01_scope_50_epoch_15_f1_0.96_(2,3).ckpt"
template_disc="(2,3)"

for template in "(6,6)" 
do
    echo  ---epoch--$epoch---------
    echo  ---template--$template---------
    CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --file_name $file_name --out_dir $out_dir --epoch $epoch --model_name_or_path $model_name_or_path --mode $mode --data_path=$data_path --template $template --corpus_type $corpus_type  --tuning_name $tuning_name --disc_embedding_checkpoint $disc_embedding_checkpoint --template_disc $template_disc  --ranking_scope $ranking_scope --top_p $top_p  --temperature $temperature
    wait
done