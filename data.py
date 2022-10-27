import itertools
import json
import linecache
import os
import pickle
import re
import socket
import string
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.data import Dataset
import jsonlines

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length,
    ):
        super().__init__()
        
        self.src_file = Path(data_dir)
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        
        
    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n") #+self.tokenizer.bos_token
        source_line = source_line.replace("xxx", '')
        
        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt", truncation=True, padding="max_length")
        return [res_input["input_ids"], res_input["attention_mask"]]
        

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
    

    
class ToxicPrompt(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length,
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        
        self.src_file = data_dir
        
        self.prompts = []
        with open(str(self.src_file), "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    prompt = item["prompt"]["text"]
                    self.prompts.append(prompt)
        
        self.tokenizer = tokenizer
        self.max_lens = max_length
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        index = index  # linecache starts at 1
        source_line = self.prompts[index].rstrip("\n")
        source_line = source_line.replace("xxx", '')
    
        res=self.tokenizer.encode_plus(source_line, max_length=self.max_lens, return_tensors="pt",truncation=True, padding="max_length")
        
        return (res["input_ids"], res["attention_mask"])
        
    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]


  
class SentimentPrompt(Dataset):
    
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length,
        prompt_type="negative",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        
        self.src_file = data_dir + "/" + str(prompt_type) + '_prompts.jsonl'
        
        self.prompts = []
        with open(str(self.src_file), "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    prompt = item["prompt"]["text"]
                    self.prompts.append(prompt)
        
        self.tokenizer = tokenizer
        self.max_lens = max_length
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        index = index  # linecache starts at 1
        source_line = self.prompts[index].rstrip("\n") 
        source_line = source_line.replace("xxx", '')
 
        assert source_line, f"empty source line for index {index}"
    
        res=self.tokenizer.encode_plus(source_line, max_length=self.max_lens, return_tensors="pt",truncation=True, padding="max_length")
        
        return (res["input_ids"], res["attention_mask"])
        
    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
    
    
    
    
    
class DetoxicDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        label_token = {}
    ):
        super().__init__()
        
        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length
        self.max_target_length = max_length
        
        self.label_token = label_token
        
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        
        self.tokenizer = tokenizer
        self.prefix = prefix
        
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"
        
    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n") #+self.tokenizer.bos_token
        source_line = source_line.replace("xxx", '')
        
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        tgt_line =  str(tgt_line)
        if "1" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['positive']))
        else:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['negative']))

        
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        
        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt", truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]
        

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    
    

class Classification_Dataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        label_token = {}
    ):
        super().__init__()
        
        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")


        
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length
        self.max_target_length = max_length
        
        self.label_token = label_token
        
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        
        self.tokenizer = tokenizer
        self.prefix = prefix
        
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"
        
    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n") #+self.tokenizer.bos_token
        
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
         
        if "positive" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['positive']))
        else:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['negative']))
        
        
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        
        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt", truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]
        

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
    
    
    
    
class Sentiment_Suffix(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length,
        task_type="positive",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        label_token = {}
    ):
        super().__init__()
        
        self.src_file = data_dir

        
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length
        
        self.label_token = label_token
        self.task_type = task_type
        
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        
        self.tokenizer = tokenizer
        self.prefix = prefix
        
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"
        
    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n") #+self.tokenizer.bos_token
        
        tgt_line = torch.tensor(self.tokenizer.encode(self.label_token[self.task_type]))
        
        if len(source_line)<2:
            source_line = "Hello world! Today is nice!"
            
        
        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt", truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]
        

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
