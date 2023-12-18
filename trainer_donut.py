import os
"""
trainer.train()에서 진행도가 멈추는경우 추가해보기
"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DonutProcessor
from data_utils import *
from dataset import DeliveryOrder

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# hyperparameters used for multiple args
hf_repository_id = "donut-base-DO"
filepath = "./"

task_start_token = "<s>"
eos_token = "</s>"

def create_model(processor, max_length):
    # Load model from huggingface
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    # Resize embedding layer to match vocabulary size
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")
    
    # Adjust output sequence lengths
    model.config.decoder.max_length = max_length
    
    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
    return model

def get_training_args():
    # Arguments for training
    training_args = Seq2SeqTrainingArguments(
        output_dir=hf_repository_id,
        num_train_epochs=50,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        save_strategy="epoch",
        predict_with_generate=True,
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=hf_repository_id,
        hub_token=HfFolder.get_token(),
    )
    return training_args
def train():
    # Load and Preprocess dataset
    dataset = DeliveryOrder(filepath, task_start_token, eos_token, split='train')
    processed_dataset = dataset.preprocess_documents_for_donut().add_new_special_tokens()
    processor = processed_dataset.processor
    model = create_model(processor, processed_dataset.max_length)
    
    # is done by Trainer
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    
    training_args = get_training_args()
    
    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save processor and create model card
    processor.save_pretrained(hf_repository_id)
    trainer.create_model_card()
    trainer.push_to_hub()

if __name__ == "__main__":
    train()

