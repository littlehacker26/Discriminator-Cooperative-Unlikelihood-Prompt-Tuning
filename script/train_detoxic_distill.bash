gpu=0
top_p=1.0
temperature=0.01
batch_size=150
file_name='eval'
out_dir='../detoxic_model/gen'
model_name_or_path="/home/zhanghanqing/pretrained_model/gpt2/large"
mode="train"
data_path="../datasets/detoxic"
task_name="detoxic"
template="(5,5)"
corpus_type="positive"
tuning_name="distill_tuning"
epoch=6
ranking_scope=30
disc_embedding_checkpoint="../check_point/detoxic/disc_gpt2_small/detoxic_classifier.ckpt"
template_disc="(2,3)"
# 

for ranking_scope in  110
do
    echo  ---epoch--$epoch---------
    echo  ---ranking_scope--$ranking_scope---------
    CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --file_name $file_name --out_dir $out_dir --epoch $epoch --model_name_or_path $model_name_or_path --mode $mode --data_path=$data_path --template $template --corpus_type $corpus_type  --tuning_name $tuning_name --disc_embedding_checkpoint $disc_embedding_checkpoint --template_disc $template_disc  --ranking_scope $ranking_scope --top_p $top_p  --temperature $temperature
    wait
done