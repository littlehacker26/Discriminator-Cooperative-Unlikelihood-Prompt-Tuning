import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import re

from transformers import AutoTokenizer

from models import get_embedding_layer, create_model
from prompt_encoder import PromptEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_CONST = 1e-10
BIG_CONST = -1e15

class Prompt_tuning(torch.nn.Module):

    def __init__(self, args, template, label_token = None):
        super().__init__()
        self.args = args

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # model setting
        self.model = create_model(self.args)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = self.model.cuda()
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
                # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()
            
        if self.args.use_lm_finetune == True:
            self.generative_model = self.create_model(self.args).cuda()
            for param in self.generative_model.parameters():
                param.requires_grad = False 
        else:
            self.generative_model = self.model
            
        # label information
        self.label_token = label_token
        self.label_token_ids = {}
        
        for k, v in self.label_token.items():
            print(k,v,self.tokenizer.encode(v))
            
            self.label_token_ids[k] = self.tokenizer.encode(v)

        self.template = template
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        self.prompt_encoder = self.prompt_encoder.cuda()
                
        # self.fc_loss = CrossEntropyLoss(ignore_index = self.tokenizer.eos_token_id)
        
        
    def get_query_head(self, x_h, prompt_tokens, x_t = None):
        
        prompt_tensor_head = torch.tensor(prompt_tokens* (self.spell_length)).to(x_h.device)
                
        trans_inputs = []
        
        index_musk =  (x_h == self.tokenizer.pad_token_id).type(torch.uint8) # only calculte the token which is not eos
        
        valid_number_length = torch.sum(index_musk, 1)
        
        for index, seq in zip(valid_number_length, x_h):
            # if index == 0:
            #     trans_inputs.append(torch.cat([seq, prompt_tensor_head]))
            if index == x_h.shape[1]:
                trans_inputs.append(torch.cat([prompt_tensor_head,seq]))
            else:
                trans_inputs.append(torch.cat([seq[:index], prompt_tensor_head, seq[index:]]))
                
        res = torch.stack(trans_inputs, dim=0)
                        
        if x_t != None:
            # x_t = x_t.unsqueeze(1)
            return  torch.cat([res, x_t], dim =1)
        else:
            return res        
        
        
    def embed_input_head(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).type(torch.uint8).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds
    
            
    def generate(self, prompts_ids, max_length, desired_att = None,  beta = 0.5):
        """
        generation forward based on given prompt tokens, 
        Args:
            prompt_ids: the  prompt tokens
            max_length: the max len of the generation
        Returns:
       
            generated_texts:[generated tokens]
        """
        cur_len = prompts_ids.shape[1]
        logits = []
        output_ids = prompts_ids
        return_dict = {}
        eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).cuda()
        
        
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(prompts_ids, prompt_tokens)
        inputs_embeds = self.embed_input_head(queries)

        
        # construct label ids
        attention_mask = torch.cat([prompts_ids!= self.tokenizer.pad_token_id , torch.ones([prompts_ids.shape[0], self.prompt_encoder.spell_length + max_length-prompts_ids.shape[1]]).long().to(prompts_ids.device)], dim=1)

        # get embedded input
        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        while cur_len <= max_length:
            
            outputs = self.generative_model(inputs_embeds=inputs_embeds,
                                                       attention_mask = attention_mask[:,:inputs_embeds.shape[1]],
                                                       position_ids = position_ids[:,:inputs_embeds.shape[1]],
                                                       return_dict=True)
                
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits_ = self.top_k_top_p_filtering(next_token_logits,  top_k=self.args.ranking_scope, top_p=1.0, filter_value=BIG_CONST)
            
            next_token_logits_prob = torch.softmax(next_token_logits_, dim=1)
            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            
            eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(torch.uint8))# if flag = 0, it means the generated is over 
            next_tokens = next_tokens.mul(eos_flag)
            next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id            
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, self.embeddings(next_tokens).unsqueeze(1)], dim=1)

            print("cur_len is:",cur_len)
            cur_len = cur_len + 1
            
            
        
        return_dict = {"generated_tokens":output_ids}
        return return_dict
    
    
    def top_k_top_p_filtering(self,
        logits,
        top_k = 0,
        top_p = 1.0,
        filter_value = -1e15 ,
        min_tokens_to_keep = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
            
        return logits
   
    
    def forward(self, x_hs, x_ts, att_mask):
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(x_hs, prompt_tokens)
        # construct label ids
        attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.prompt_encoder.spell_length]).long().to(att_mask.device)], dim=1)
        
        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        labels = torch.clone(queries)
        
        labels.masked_fill_(attention_mask==0, -100)
        labels.masked_fill_(queries == self.pseudo_token_id, -100)
                
        # get embedded input
        inputs_embeds = self.embed_input_head(queries)

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels= labels)
        
        return output.loss