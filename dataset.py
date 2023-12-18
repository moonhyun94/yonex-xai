from torch.utils.data import Dataset
from transformers import DonutProcessor

import os
import json
from PIL import Image
from data_utils import *

class DeliveryOrder(Dataset):
    def __init__(self, path, s_tok='<s>', e_tok='</s>', split='train'):
        self.split = split
        self.json_path = os.path.join(path,split,f'meta_{split}.json')
        self.task_start_token = s_tok # start of task token
        self.eos_token = e_tok # eos token of tokenizer
        self.new_special_tokens = [] # new tokens which will be added to the tokenizer
        self.max_length = 512
        self.ignore_id = -100
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        self.dataset = []
        self.processed_dataset = []
        with open(self.json_path) as f:
            do_json = json.load(f)
        for data in do_json["data"]:
            filename = data["filename"]
            text = data["text"]
            self.dataset.append((os.path.join(path, split, filename), text))
        
    def __len__(self):
        return len(self.processed_dataset)
    
    def preprocess_documents_for_donut(self):
        # create Donut-style input
        for img, text in self.dataset:
            d_doc = self.task_start_token + json2token(text, self.new_special_tokens) + self.eos_token
            self.processed_dataset.append({"image":img, "text":d_doc})
        return self
    
    def add_new_special_tokens(self):
        print(f"add new special tokens to tokenizer Bef {len(self.processor.tokenizer)}")
        # add new special tokens to tokenizer
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": self.new_special_tokens \
                                                + [self.task_start_token] + [self.eos_token]})
        print(f"add new special tokens to tokenizer Aft {len(self.processor.tokenizer)}")
        return self

    def __getitem__(self, idx):
        image = Image.open(self.processed_dataset[idx]["image"]).convert('RGB')
        text = self.processed_dataset[idx]["text"]
        # create tensor from image
        try:
            pixel_values = self.processor(
                image, random_padding=self.split == "train", return_tensors="pt"
            ).pixel_values.squeeze()
        except Exception as e:
            print(sample)
            print(f"Error: {e}")
            return {}
        
        input_ids = self.processor.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        return {"pixel_values": pixel_values, "labels": labels, "target_sequence": text}
    