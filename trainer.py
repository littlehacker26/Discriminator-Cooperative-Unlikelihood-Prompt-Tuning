import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from transformers import pipeline, set_seed


from os.path import join, abspath, dirname
from data import Classification_Dataset, SentimentPrompt, DetoxicDataset, Sentiment_Suffix

from discriminator import PTuneForLAMA
from prompt_tuning import Prompt_tuning
from distill_tuning import Distill_Tuning

      
        
class Trainer(object):
    def __init__(self, args):
        self.args = args

        # self.label_token ={
        #   "positive":'good',
        #   "negative":'bad'
        # }
        
        self.label_token ={
          "positive":'positive',
          "negative":'negative'
        }
            
        if self.args.tuning_name == "prompt_tuning":
            self.label_token ={"positive":'good',"negative":'bad'}
            self.model = Prompt_tuning(args, args.template, label_token = self.label_token)
            
        elif self.args.tuning_name == "distill_tuning":
            self.label_token ={"positive":'good',"negative":'bad'}
            self.model = Distill_Tuning(args, args.template, label_token = self.label_token)           
            
        elif self.args.tuning_name == "disc_tuning":
            self.label_token ={"positive":'good',"negative":'bad'}
            self.model = PTuneForLAMA(args, args.template, label_token = self.label_token)

        self.tokenizer = self.model.tokenizer
        data_path = args.data_path

        if self.args.task_name == "sentiment":
            print(self.args.tuning_name)

            if self.args.tuning_name == "disc_tuning" or self.args.tuning_name == "distill_tuning":
                all_dataset = Classification_Dataset(tokenizer = self.tokenizer, data_dir=data_path, max_length = 30, type_path="train", label_token =  self.label_token)
                
            else:
                
                all_dataset = Sentiment_Suffix(tokenizer = self.tokenizer, data_dir=data_path, max_length = 30, task_type= self.args.corpus_type, label_token =  self.label_token)
              
        elif self.args.task_name == "detoxic":
            print("load detoxic dataset!!!")
            
            if self.args.tuning_name == "disc_tuning" or self.args.tuning_name == "distill_tuning":

                all_dataset = DetoxicDataset(tokenizer = self.tokenizer, data_dir=data_path, max_length = 30, type_path="train", label_token =  self.label_token)
                
            else:
                 all_dataset = Sentiment_Suffix(tokenizer = self.tokenizer, data_dir=data_path, max_length = 30, task_type= self.args.corpus_type, label_token =  self.label_token)
                
            
        train_size = int(len(all_dataset) * 0.9)
        test_size = len(all_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, args.batch_size, num_workers=2,  shuffle=True)
        self.test_loader  = DataLoader(test_dataset, args.batch_size, num_workers=2,  shuffle=True)
    

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
        else:
            loader = self.dev_loader
        labels = []
        preds  = []
        with torch.no_grad():
            self.model.eval()
            for batch in loader:
                self.model.eval()
                x = batch[0].cuda().squeeze(1)
                musk = batch[1].cuda().long().squeeze(1)
                y = batch[2]
                
                pred_ids  = self.model.predict(x,musk)
                
                preds += pred_ids
                labels += y.tolist()  
                
            result = classification_report(labels, preds, output_dict = True)
            print(result)
        
        return result


    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model')
    

    def get_checkpoint(self, epoch_idx, f1_score):
        ckpt_name = "{}_{}_temperature{}_scope_{}_epoch_{}_f1_{}_{}.ckpt".format(self.args.tuning_name ,self.args.corpus_type, self.args.temperature, self.args.ranking_scope, epoch_idx, str(f1_score), str(self.args.template).replace(" ",""))
        return {'embedding': self.model.prompt_encoder.state_dict(),
                'ckpt_name': ckpt_name,
                'args': self.args}
    
    

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        
        if self.args.use_lm_finetune:
            self.model.model.save_pretrained(str(join(path, ckpt_name))[:-5])
        print("Checkpoint {} saved.".format(ckpt_name))
        

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters(), 'lr':self.args.lr}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 1e-5})
            
            
        optimizer = torch.optim.Adam(params,  weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        stop_count = 0
        best_result = 0.0 
        for epoch_idx in range(self.args.epoch):
                            
            tot_loss = 0
            count =0 
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                x = batch[0].cuda().squeeze(1)
                musk = batch[1].long().cuda().squeeze(1)
                y = batch[2].long().cuda()   
                
                loss = self.model(x, y, musk)
                
                tot_loss += loss.item()
                
                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                
                
            print(f"epoch index is {epoch_idx}, and total loss is {tot_loss}")
            
            my_lr_scheduler.step()
                
            
#             if epoch_idx > -1:
#                 result = self.evaluate(epoch_idx, 'Test')
#                 weight_avg =result["weighted avg"]
#                 f1_score = weight_avg["f1-score"] 
                
#                 if f1_score > best_result:
#                     best_ckpt = self.get_checkpoint(epoch_idx,best_result)
#                     best_result =  f1_score
#                     stop_count = 0
#                     continue
#                 else:
#                     stop_count += 1
#                     if stop_count>5:
#                         self.save(best_ckpt)
#                         break

            if epoch_idx >= -1:
                
                if self.args.tuning_name == "prompt_tuning":
                    result = self.evaluate(epoch_idx, 'Test')
                    weight_avg =result["weighted avg"]
                    f1_score = round(weight_avg["f1-score"],2)
                    best_ckpt = self.get_checkpoint(epoch_idx,round(f1_score,2))
                else:
                    best_ckpt = self.get_checkpoint(epoch_idx,round(tot_loss,2))
                    
                self.save(best_ckpt)


def main(relation_id=None):
    args = construct_generation_args()
    
    #train stage
    trainer = Trainer(args)
    trainer.train()
    
    ## generation stage

if __name__ == '__main__':
    main()
