import logging
import csv
import random
import torch
import os
import sys
from torch.nn import CrossEntropyLoss
import numpy as np

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
    tokenizer.padding_side = "left"

    return ppl



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
    
    
def classification_log(test_label,test_pred):
        test_label = [j for i in test_label for j in i]
        test_pred = [j for i in test_pred for j in i]

        new_preds = []
        new_labels = []
        
        for l, pred in zip(test_label,test_pred):
            if l != -1:
                new_labels.append(l)
                new_preds.append(pred)
                
        return classification_report(new_labels, new_preds, output_dict=True)
    
    

import pandas as pd
def save_csv_to_text(filename, csv_name, usecols):
    '''
    read csv data，convert the data in the specific column to txt file
    '''
    data = pd.read_csv(csv_name, usecols=[usecols])
    data_list = data.values.tolist()
    result = []
    for item in data_list:
        result.append(item[0])
    print("start process {}".format(filename))
    with open(filename, 'w', encoding='utf-8') as f:
        for item in result:
            f.write(item + '\n')
    f.close()
    print('save {} done!'.format(filename))
    print("---------------------")
        


def addCsv(filename,dict_inf):
    """
        add dict information to csv file
        filename: log file, if not exists,then creat 
        dict_inf: key,value 
    """
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = dict_inf.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict_inf)
    else:
        with open(filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=dict_inf.keys())
                    writer.writerow(dict_inf)
                                        