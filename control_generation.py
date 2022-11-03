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
import numpy as np

from os.path import join, abspath, dirname
from data import Classification_Dataset, SentimentPrompt, ToxicPrompt

from utils import addCsv
from utils import cal_ppl_bygpt2


import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss


from prompt_tuning import Prompt_tuning
from discriminator import PTuneForLAMA
from distill_tuning import Distill_Tuning



class CTG(object):
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
    
        if  self.args.tuning_name == "prompt_tuning":
            self.model = Prompt_tuning(args, args.template, label_token = self.label_token)
            
        elif self.args.tuning_name == "distill_tuning":
            self.model = Distill_Tuning(args, args.template, label_token = self.label_token)
            
        elif self.args.tuning_name == "disc_tuning":
            self.label_token ={"positive":'good',"negative":'bad'}
            self.model = PTuneForLAMA(args, args.template, label_token = self.label_token)
        else:
            raise Exception("the tuning mode is not existing!")
        
        self.tokenizer = self.model.tokenizer
        
        # init the prompt encoder's parameters
        if args.embedding_checkpoint!= None:
            self.model.prompt_encoder.load_state_dict(self.load_prompt(args.embedding_checkpoint))
        
        
        print("The running task is: ", self.args.task_name)
        print("tuning name is :", self.args.tuning_name)
    
        # load datasets and dataloaders
        data_path = args.data_path
        
        if self.args.task_name=="sentiment":
            print("sentiment control generation!")
            # data_path =  args.data_path
            pos_dataset = SentimentPrompt(tokenizer=self.tokenizer, data_dir=data_path, max_length=self.args.max_prompt_length,  prompt_type= self.args.prompt_type)
            
        elif self.args.task_name=="detoxic":
            
            print("detoxic generation!")
            
            # data_path =  args.data_path
            pos_dataset = ToxicPrompt(tokenizer=self.tokenizer, data_dir=data_path, max_length=self.args.max_prompt_length)
            
        else:
            
            raise Exception("the task is out of scope!")
            
            
        self.pos_loader = DataLoader(pos_dataset, args.batch_size, num_workers=2,  shuffle=True)
        self.prompt_pad_length = self.args.prompt_pad_length
        
        self.generator_model = self.model.model        
        self.generateor_embedding = self.generator_model.get_input_embeddings()
        self.discrimirator_embedding = self.generateor_embedding

    
#     def statistic(self,dataSentiment, dataPPL, dataText, desired_label):
#         sentiment = np.array(dataSentiment).T
#         ppl = np.array(dataPPL).T
#         text = np.array(dataText).T
#         res = []
#         for s, p, t in zip(sentiment, ppl, text):

#             res.append(self._statistic(s,p,t,desired_label))
#         return res


#     def _statistic(self, dataSentiment, dataPPL, dataText, desired_label):

#         sentiment = np.array(dataSentiment)

#         ppl = np.array(dataPPL)

#         sentiment_target = np.argwhere(sentiment==desired_label)  #get desired text's ppl 

#         if len(sentiment_target)>1:
#             _ppl = ppl[sentiment_target] ##get ppl
#             min_index = _ppl.argsort()[0] ##get the index of min value of ppl
#             target_index =sentiment_target[min_index][0][0] ## get target value
#             # print(target_index)
#             if dataPPL[target_index]<=500:
#                 return (dataSentiment[target_index], dataPPL[target_index], dataText[target_index],desired_label)

#         min_index = ppl.argsort()[0] ##get the index of min value of ppl

#         return (dataSentiment[min_index], dataPPL[min_index], dataText[min_index],desired_label)
            
        
    def test(self):
        
        att = self.args.target_type
        print("the desired att is:", att)
        desired_att_token = self.model.label_token_ids[att]

        if self.args.task_name =="sentiment":
            file_name = f"{self.args.file_name}/result_beta_{self.args.beta}_ranking_scope_{self.args.ranking_scope}_{self.args.top_p}_{self.args.prompt_type}_to_{att}_{desired_att_token}.csv"
            
        elif self.args.task_name =="detoxic":
            file_name = f"{self.args.file_name}/ranking_scope_{self.args.ranking_scope}_{self.args.top_p}_detoxic.csv"
            
        else:
            raise Exception("the task is not specific!")
        
        count = 0
        for data in self.pos_loader:
            
                count+= 1
            
                x = data[0].squeeze(1).to(self.args.device)
                musk = data[1].long().squeeze(1).to(self.args.device)
                desired_att = torch.tensor([desired_att_token]).expand(x.shape[0],-1).to(self.args.device)
                    
                output_seq = self.model.generate(prompts_ids = x, max_length = self.args.max_length, desired_att=desired_att, beta = self.args.beta)
                    
                text = self.tokenizer.batch_decode(output_seq["generated_tokens"], skip_special_tokens= True)
                text = [t.replace('\n', '') for t in text]
                print("generated:", text)
                    
                ppl = cal_ppl_bygpt2(self.tokenizer, self.model.model, self.args.max_length, text)
                print("ppl is :", cal_ppl_bygpt2(self.tokenizer, self.model.model, self.args.max_length, text))
                    
                    
                for i in range(len(ppl)):
                    dict_csv={}
                    # dict_csv["result"] = result_eval[i]
                    dict_csv["ppl"] = ppl[i]
                    dict_csv["text"] = text[i]
                    addCsv(file_name, dict_csv)

                    
                        
        
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding        
        

            

                   

def main(relation_id=None):
    args = construct_generation_args()
    
    # train stage
    trainer = Trainer(args)
    trainer.train()
    
    # generation stage
    
    gen =  CTG(args)
    gen.test()
    
    


if __name__ == '__main__':
    main()
