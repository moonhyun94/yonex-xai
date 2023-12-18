# new_special_tokens = [] 
# task_start_token = "<s>"  
# eos_token = "</s>" 

def json2token(obj, new_special_tokens, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = False):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], new_special_tokens, update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj

# def preprocess_documents_for_donut(sample):
    
#     # text = json.loads(sample["text"])
#     text = sample["text"]
#     d_doc = task_start_token + json2token(text) + eos_token
#     # convert all images to RGB
#     image = sample["image"].convert('RGB')
#     return {"image": image, "text": d_doc}

# def add_special_tokens(processor):
#     # add new special tokens to tokenizer
#     processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})
#     return processor

# def transform_and_tokenize(sample, processor, split="train", max_length=512, ignore_id=-100):
#     # create tensor from image
#     try:
#         pixel_values = processor(
#             sample["image"], random_padding=split == "train", return_tensors="pt"
#         ).pixel_values.squeeze()
#     except Exception as e:
#         print(sample)
#         print(f"Error: {e}")
#         return {}

#     # tokenize document
#     input_ids = processor.tokenizer(
#         sample["text"],
#         add_special_tokens=False,
#         max_length=max_length,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt",
#     )["input_ids"].squeeze(0)

#     labels = input_ids.clone()
#     labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
#     return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}