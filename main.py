import json
import os
import torch
import argparse
import numpy as np

from collections import Counter
from os.path import join, abspath, dirname

from torch.utils.data import DataLoader
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from transformers import pipeline, set_seed


from trainer import Trainer
from control_generation import  CTG
from classifier import Classifier

from data import Classification_Dataset, SentimentPrompt
from utils import seed_everything
from utils import addCsv, findAllFile



def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--model_name_or_path", type=str, default='/home/zhanghanqing/pretrained_model/gpt2/large')

    parser.add_argument("--data_path", type=str, default='../data/pos_neg')
    
    parser.add_argument("--embedding_checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="sentiment",choices = ["detoxic","sentiment"])
                        
    

    parser.add_argument("--pseudo_token", type=str, default='xxx')
    
    parser.add_argument("--batch_size", type=int, default= 600)
    parser.add_argument("--epoch", type=int, default= 50)


    parser.add_argument("--template", type=str, default="(2, 2)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)

    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), './checkpoint'))
    # MegatronLM 11B
    
    ## generation configure
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--max_prompt_length", type=int, default=10)

    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--prompt_type", type=str, default="negative")
    parser.add_argument("--target_type", type=str, default="positive")

    parser.add_argument("--prompt_pad_length", type=int, default= 10)
    # parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--ranking_scope", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    
    parser.add_argument("--file_name", type=str, default="./eval")
    parser.add_argument("--mode", type=str, default="ctg", choices=["ctg","train","classifer"])
    parser.add_argument("--evaluate_file", type=str, default="../our_text")
    parser.add_argument("--evaluate_outfile", type=str, default="./eval/our/result.csv")
    parser.add_argument("--iter_num", type=int, default=10)
    parser.add_argument("--corpus_type", type=str, default="positive")
    parser.add_argument("--tuning_name", type=str, default="disc_tuning", choices=["prompt_tuning","disc_tuning","distill_tuning"])
    
    ## discriminator information for distilled tuning
    parser.add_argument("--disc_embedding_checkpoint", type=str, default= None)
    parser.add_argument("--template_disc", type=str, default="(2, 3)")

    args = parser.parse_args()
    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    args.template_disc = eval(args.template_disc) if type(args.template_disc) is not tuple else args.template_disc

    assert type(args.template) is tuple

    seed_everything(args.seed)

    return args

                    

def result_evaluation(args, file_dir, save_file):
    
    path = findAllFile(file_dir)
    
    for p in path:
        classifier = Classifier(args, p)
        result = classifier.evaluate()
        
        if args.task_name =="sentiment":
            c = dict(Counter(result))
            res = {}
            pos = c[11274]
            pos_rate = pos/len(result)

            res["path"] = str(p)
            res["pos_rate"] = pos_rate
            res["neg_rate"] = 1- pos_rate
            addCsv(save_file, res)
            
        elif args.task_name =="detoxic":
            res = {}
            res["path"] = str(p)
            res["toxic_prob"] = np.nanmean(result)
            addCsv(save_file, res)
            

def main(relation_id=None):
    args = construct_generation_args()
    
    print("the task is:", args.mode)
    ## generation mode
    if args.mode =="ctg":
        gen =  CTG(args)
        gen.test()
    
    ## train classifier or prompt-learning
    elif args.mode =="train":
        trainer = Trainer(args)
        trainer.train()
        
    ## evaluation mode using offline classifier
    elif args.mode =="classifer":
        result_evaluation(args, args.evaluate_file, args.evaluate_outfile)
        
    else:
        raise Exception("the task is out of scope!") 


if __name__ == '__main__':
    main()
