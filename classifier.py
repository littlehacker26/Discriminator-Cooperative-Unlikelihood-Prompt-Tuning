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
from data import Classification_Dataset, SentimentPrompt
from data import TextDataset


from discriminator import PTuneForLAMA
      
        
class Classifier(object):
    def __init__(self, args, data_path):
        self.args = args

        #load model and tokenizer
        self.label_token ={
          "positive":'good',
          "negative":'bad'}
        
        self.model = PTuneForLAMA(args, args.template, label_token = self.label_token)
        self.tokenizer = self.model.tokenizer
        # self.model.prompt_encoder.load_state_dict(self.load_prompt(args.embedding_checkpoint))
        self.data_path = data_path        


        # if self.args.task_name=="sentiment":
        all_dataset = TextDataset(tokenizer = self.tokenizer, data_dir= self.data_path, max_length = 35)
        self.data_loader = DataLoader(all_dataset, args.batch_size, num_workers=2,  shuffle=True)
            
        # elif self.args.task_name=="detoxic":
        #     all_dataset = ToxicPrompt(tokenizer = self.tokenizer, data_dir= self.data_path, max_length = 35)
        #     self.data_loader = DataLoader(all_dataset, args.batch_size, num_workers=2,  shuffle=True)
    
    # def load_prompt(self, embedding_checkpoint):
    #     checkpoint = torch.load(embedding_checkpoint)
    #     prompt_embedding = checkpoint['embedding']
    #     return prompt_embedding
    
    def cal_ppl_bygpt2(tokenizer, model, max_length, sentence): 
        # tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        # model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        
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
 

    def evaluate(self):
        loader = self.data_loader
        labels = []
        preds  = []
        with torch.no_grad():
            self.model.eval()
            for batch in loader:
                x = batch[0].cuda().squeeze(1)
                musk = batch[1].cuda().long().squeeze(1)
                if self.args.task_name=="sentiment":
                    pred_ids  = self.model.predict(x,musk)
                    preds += pred_ids
                elif self.args.task_name=="detoxic":
                    print(x.shape, musk.shape)
                    pred_ids  = self.model._predict_scores(x,musk).cpu().tolist()
                    preds += pred_ids
        return preds




                        
                    
            


