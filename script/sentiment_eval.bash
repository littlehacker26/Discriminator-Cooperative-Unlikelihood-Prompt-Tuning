gpu=2
temperature=1.0
batch_size=600
model_name_or_path="/home/zhanghanqing/pretrained_model/gpt2/large"
embedding_checkpoint="../sentiment_model/large/prompt_model/epoch_6_f1_0.94_(1,2).ckpt"
evaluate_outfile="../../other_method/eval/sentiment_result.csv"
mode="classifer"
task_name="sentiment"
evaluate_file="../../other_method/text_data"




echo $embedding_checkpoint

CUDA_VISIBLE_DEVICES=$gpu  python ../main.py   --batch_size $batch_size  --model_name_or_path $model_name_or_path  --evaluate_file $evaluate_file  --evaluate_outfile $evaluate_outfile --mode $mode --embedding_checkpoint $embedding_checkpoint --task_name $task_name
