import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering

import re
import datetime

from transformers import AutoTokenizer

from models import get_embedding_layer, create_model, _create_model
from prompt_encoder import PromptEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_CONST = 1e-10
BIG_CONST = -1e15

class Distill_Tuning(torch.nn.Module):

    def __init__(self, args, template, label_token = None):
        super().__init__()
        self.args = args

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # model setting
        self.model = create_model(self.args)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.args.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
            
        # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()
            

        # label information
        self.label_token = label_token
        self.label_token_ids ={}
        
        for k, v in self.label_token.items():
            print(k,v,self.tokenizer.encode(v))
            
            self.label_token_ids[k] = self.tokenizer.encode(v)

        self.template = template
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        self.prompt_encoder = self.prompt_encoder.to(self.args.device)
                
        # self.fc_loss = CrossEntropyLoss(ignore_index = self.tokenizer.eos_token_id)
        
        ### load discriminator
        if self.args.disc_embedding_checkpoint != None:
            
            self.disc_model = _create_model(self.args.disc_embedding_checkpoint[:-5]).to(self.args.device)
            self.spell_length_disc = sum(self.args.template_disc)
            self.disc_embedding = self.disc_model.get_input_embeddings()
            self.prompt_encoder_disc = PromptEncoder(self.args.template_disc, self.disc_embedding.embedding_dim, self.tokenizer, args)
            self.prompt_encoder_disc = self.prompt_encoder_disc.to(self.args.device)
            self.prompt_encoder_disc.load_state_dict(self.load_prompt(self.args.disc_embedding_checkpoint))
        else :
            self.disc_model = self.model
            self.prompt_encoder_disc  =  self.prompt_encoder
            
            
    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding
    
    def generate_soft_tokens(self, generated_tokens, past_key_values= None, position = None):
        
        if past_key_values!= None:
            last_embeds =self.embeddings(generated_tokens[:, -1]).unsqueeze(1)#get its embeddings
            with torch.no_grad():
                outputs = self.model(inputs_embeds=last_embeds,
                                               past_key_values = past_key_values,
                                               position_ids = position[:, past_key_values[0][0].shape[-2]],
                                               return_dict=True)
                
        else:
            attention_mask = (generated_tokens!=self.tokenizer.eos_token_id).type(torch.uint8)
            position_ids = attention_mask.long().cumsum(-1)- 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            last_embeds =self.embeddings(generated_tokens) #get its embeddings
            with torch.no_grad():
                outputs = self.model(inputs_embeds=last_embeds,
                                               past_key_values = past_key_values,
                                               attention_mask = attention_mask,
                                               position_ids = position_ids,
                                               return_dict=True)
                
        next_token_logits = outputs.logits[:, -1, :]
                        
        next_token_logits = self.top_k_top_p_filtering(next_token_logits.squeeze(1), top_k=self.args.ranking_scope, top_p=self.args.top_p, filter_value=BIG_CONST)
                         
        return next_token_logits, outputs.past_key_values
    
    
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
        
    

    
    def _predict_scores(self, x_hs, att_mask):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        
        queries = self.get_query(x_hs, prompt_tokens)
        # construct label ids
        attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.spell_length_disc]).long().to(self.args.device)], dim=1)
        # get embedded input
        inputs_embeds = self.embed_input(queries)
        
        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        with torch.no_grad():
            output = self.disc_model(inputs_embeds = inputs_embeds,
                            attention_mask = attention_mask,
                            position_ids = position_ids,
                            labels=None)

        logits = output.logits[:,-1,:].squeeze(1)
        
        binary_prob = torch.softmax(logits[:,[11274,14774]], dim=-1)
        
        if self.args.corpus_type == "negative":
            return binary_prob[:,1]
        else:
            return binary_prob[:,0]    
    
    def get_query(self, x_h, prompt_tokens, x_t = None):
        
        prompt_tensor = torch.tensor(prompt_tokens* (self.spell_length_disc)).to(self.args.device)
        prompt_tensor = prompt_tensor.expand(x_h.shape[0],-1)
        if x_t != None:
            x_t = x_t.unsqueeze(1)
            return  torch.cat([x_h, prompt_tensor, x_t], dim =1)
        else:
            return  torch.cat([x_h, prompt_tensor], dim =1)
        

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        raw_embeds = self.disc_embedding(queries_for_embedding)
        
        replace_embeds = self.prompt_encoder_disc()
        
        replace_embeds = replace_embeds.unsqueeze(0).expand(bz,-1, -1)
        
        raw_embeds[:,-self.prompt_encoder_disc.spell_length:,: ] = replace_embeds
                
        return raw_embeds
    
    
    def get_query_head(self, x_h, prompt_tokens, x_t = None):
        
        prompt_tensor_head = torch.tensor(prompt_tokens* (self.spell_length)).to(self.args.device)
                
        trans_inputs = []
        
        index_musk =  (x_h == self.tokenizer.pad_token_id).type(torch.uint8) # only calculte the token which is not eos
        
        valid_number_length = torch.sum(index_musk, 1)
        
        for index, seq in zip(valid_number_length, x_h):
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
        eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)
        
        
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(prompts_ids, prompt_tokens)
        inputs_embeds = self.embed_input_head(queries)

        attention_mask = torch.cat([prompts_ids!= self.tokenizer.pad_token_id , torch.ones([prompts_ids.shape[0], self.prompt_encoder.spell_length + max_length-prompts_ids.shape[1]]).long().to(self.args.device)], dim=1)

        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # start = datetime.datetime.now()
        # test generation time

        while cur_len <= max_length:
            outputs = self.model(inputs_embeds=inputs_embeds,
                                                       attention_mask = attention_mask[:,:inputs_embeds.shape[1]],
                                                       position_ids = position_ids[:,:inputs_embeds.shape[1]],
                                                       return_dict=True)
                
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits_ = self.top_k_top_p_filtering(next_token_logits,  top_k=self.args.ranking_scope, top_p=1.0, filter_value=BIG_CONST)
            
            next_token_logits_prob = torch.softmax(next_token_logits_, dim=1)
            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            
            eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(torch.uint8))# if flag = 0, it means the generation is over 
            next_tokens = next_tokens.mul(eos_flag)
            next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id            
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, self.embeddings(next_tokens).unsqueeze(1)], dim=1)

            print("cur_len is:",cur_len)
            cur_len = cur_len + 1
        
#         end = datetime.datetime.now()
#         print("runing time is:",end-start)
        
        return_dict = {"generated_tokens":output_ids}
        return return_dict

    
    def  feedback_from_discriminator(self, input_ids, logits_seq, desired_att):
        
        top_logits, top_indices = torch.topk(logits_seq, self.args.ranking_scope) # batch x topk
        
        candidates = []
        for logit_id,  ids  in  zip(top_indices, input_ids):
            data = ids.expand(self.args.ranking_scope, -1)
            new_input_candidates = torch.cat([data, logit_id.unsqueeze(1)], dim=1) # batch x topk x seq+1
            candidates.append(new_input_candidates)
            
        
        candidates = torch.cat(candidates, dim=0)
        
        if candidates.shape[1]<30:
            pad_tensor =torch.empty(candidates.shape[0],30 - candidates.shape[1]).long().fill_(self.tokenizer.eos_token_id).to(self.args.device)
            candidates = torch.cat([pad_tensor,candidates], dim=1)
                    
        pred_scores = []
        for new_input_candidates in torch.split(candidates, 120, dim=0):
            musk =  (new_input_candidates != self.tokenizer.eos_token_id).type(torch.uint8)
            pred_score  = self._predict_scores(new_input_candidates, musk)
            pred_scores.append(pred_score)

        pred_scores = torch.cat(pred_scores, dim=0)
        pred_scores = pred_scores.reshape(input_ids.shape[0], -1)

        res_logits = logits_seq.clone().detach()
        res_logits.scatter_(-1, top_indices, pred_scores)
        return res_logits
    
    
        
    def get_ranked_logtis(self, inputs, logits, desired_att):
        
        return_logits = []
        for i  in range(inputs.shape[1]):
            tmp = self.feedback_from_discriminator(inputs[:, :i+1], logits[:,i,:],  desired_att)
            return_logits.append(tmp)
                        
        return torch.stack(return_logits, dim=1).detach().clone()
    
    
       
    def KL_loss(self, input_x, input_y, attention_mask):
        """
        compute the KL loss
        """
        m = torch.flatten(attention_mask)
        indices = torch.nonzero(m).squeeze(-1)
        
        x = input_x.reshape(-1,input_x.shape[-1])
        x = torch.index_select(x, 0, indices)
            
        y = input_y.reshape(-1,input_y.shape[-1])
        y = torch.index_select(y, 0, indices)
            
        y_ = torch.softmax(y/self.args.temperature, dim = -1)
        loss_ = -(y_ * (x+1e-20).log()).sum() / x.size(0)
        
        _y = torch.softmax(((1-y).mul(y>0.0))/self.args.temperature, dim = -1)
        _loss = -(_y * (1-x+1e-20).log()).sum() / x.size(0)

        return  loss_ + _loss


    
    def contrast_crossEntry_loss(self, logits_prob, x_hs, sentence_labels = None):

        shift_prob = logits_prob[..., :-1, :].contiguous()
        shift_labels = x_hs[..., 1:].contiguous()

        if sentence_labels != None:
            index_negative = (sentence_labels==14774)
            index = torch.nonzero(index_negative).squeeze(1)
            shift_prob[index] = 1-shift_prob[index]

        loss =  F.nll_loss((shift_prob.view(-1, shift_prob.size(-1))+1e-20).log(), shift_labels.view(-1), ignore_index=-100)
        
        return loss
    
    
    def get_candidate_logits(self, x_hs, att_mask):
        
        position_ids = att_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(att_mask == 0, 0)
        
        with torch.no_grad():
            output = self.model(input_ids= x_hs,
                            attention_mask=att_mask,
                            position_ids=position_ids,
                            labels= None)
        return output.logits.detach().clone()
        
    
    def forward(self, x_hs, x_ts, att_mask):
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(x_hs, prompt_tokens)
        # construct label ids
        attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.prompt_encoder.spell_length]).long().to(self.args.device)], dim=1)
        
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
                            labels= None)
        
        output_logits = output.logits
        # ce_loss =  self.contrast_crossEntry_loss(torch.softmax(output_logits, dim = -1), labels, sentence_labels = x_ts)
        
        _queries = queries.view(queries.size(0)*queries.size(1))
        _output_logits = output_logits.view(output_logits.size(0)*output_logits.size(1),-1)
        disc_logits =  _output_logits.index_select(0, torch.nonzero(_queries != self.pseudo_token_id).squeeze(1)).view(output_logits.shape[0],-1, output_logits.shape[2])
        
        logits_candidate = self.get_candidate_logits(x_hs,att_mask)
        logits_candidate = self.top_k_top_p_filtering(logits_candidate.view(logits_candidate.shape[0]*logits_candidate.shape[1], -1), top_k= self.args.ranking_scope , top_p=self.args.top_p, filter_value=BIG_CONST).view(x_hs.shape[0],x_hs.shape[1], -1)

        reank_output = self.get_ranked_logtis(x_hs, logits_candidate.detach().clone(), desired_att=None)
        
        reank_output = (logits_candidate>BIG_CONST+10).mul(reank_output)

        kl_loss = self.KL_loss(torch.softmax(disc_logits, dim=-1), reank_output, att_mask)
        
        loss =  kl_loss

        return loss
        
        

 
