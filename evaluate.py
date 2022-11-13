"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""

import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from utils import addCsv, findAllFile
from  itertools import zip_longest
from torch.nn import CrossEntropyLoss
import jsonlines

review_data = ["movie", "movies", "films", "film", "story", "director" ,"directors" ,"comedy" , "audience"  "drama"]

def _sample(data, count):
    
    res = []
    sum_c = 0
    for d in data:
        if sum_c%count==1:
            res.append(d)
        sum_c+=1
    return res


def remove_prompt(prompts, text):
    for p in prompts:
        if p in text:
            return text.replace(p, '')
    return text


def distinctness(generations_data):
    dist1, dist2, dist3 = [], [], []
    total_words = 0
    unigrams, bigrams, trigrams = set(), set(), set()
    
    for gen in generations_data:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
                
    if total_words == 0:
        return 0.0, 0.0, 0.0
    
    dist1 = len(unigrams) / total_words
    dist2 = len(bigrams) / total_words
    dist3 = len(trigrams) / total_words
    
    return dist1, dist2, dist3


def cal_ppl_bygpt2(tokenizer, model, max_length, sentence):
    
    tokenizer.padding_side = "right"
    inputs = tokenizer(sentence, padding='max_length', max_length = max_length, truncation=True, return_tensors="pt").to(model.device)
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    
    loss = loss.mul(shift_attentions.type(torch.uint8))
    
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).cpu().numpy().tolist()

    return ppl


def get_review_rate(data):
    
    count =0
    sum_count = len(data)
    if sum_count==0:
        return 0.0
    
    for gen in data:
        # print("gen",gen)
        for review in review_data:
            if review in gen:
                count+=1
     
    return count/sum_count
        

def main(generations_dir, model_path, save_path):
    
    path = findAllFile(generations_dir)
    for generations_file in path:
        
        data = []
        res = {}

        with open(generations_file, 'r') as f:
            data = f.read().splitlines()
            
        res["review_rate"] = str(get_review_rate(data))
        
        # if "#" not in generations_file:
        #     data = _sample(data, 25)
            
        print("data rows:", len(data))
        dist1, dist2, dist3 = distinctness(data)
        res["path"] = str(generations_file)
        res["dist1"] = dist1
        res["dist2"] = dist2
        res["dist3"] = dist3
        print("Dist calculation finish!!!")
        print(res)
                
        eval_model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
        eval_tokenizer = AutoTokenizer.from_pretrained(model_path)
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        torch.cuda.empty_cache()
        ppls = []
        for i in zip_longest(*([iter(data)] * 120), fillvalue= "xxx"):
            i = list(i)
            while "xxx" in i:
                i.remove("xxx")
            
            if len(i)<=1:
                continue
            with torch.no_grad():
                ppl = cal_ppl_bygpt2(eval_tokenizer, eval_model, 30, i)
                ppls += ppl
        
        res["ppl"] = np.nanmean(ppls)
        
        print(res)
        addCsv(save_path, res)


if __name__ == '__main__':
    
    ## the direction of  evaluated files
    generations_file = "./eval/openweb/"
    
    ##the direction of pretrained language model(i.e., GPT2-large)
    model_path =  "/home2/xxx/pretrained_model/gpt2/large"
    
    ## the path to save the evaluated results
    save_path  = "./eval/openweb/result.csv"
    

    main(generations_file, model_path, save_path)
    

