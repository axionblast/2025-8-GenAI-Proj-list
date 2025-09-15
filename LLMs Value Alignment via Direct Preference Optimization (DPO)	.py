!pip install transformers
!pip install bitsandbytes
!pip install datasets
!pip install peft
!pip install trl
!pip install accelerate
!pip install tf-keras
!pip install numpy==1.26.4

import os
import re
import json

import torch
import pandas as pd
from tqdm.auto import tqdm

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl import DPOConfig, DPOTrainer


# 2. Load dataset
#For demonstration purposes
!git clone https://github.com/dataset.git

with open("./dataset/labelled_data.json", 'r') as jsonfile:
    full_data = json.load(jsonfile)

with open("./dataset/test_prompt.json", 'r') as jsonfile:
    test_data = json.load(jsonfile)
full_data[:5], test_data

# 3. Download model using HFD
# Use HFD to speed up Hugging Face model and dataset downloads (pre-installation)

!wget https://hf-mirror.com/hfd/hfd.sh
!chmod a+x hfd.sh

!export HF_ENDPOINT=https://hf-mirror.com
!./hfd.sh 'MediaTek-Research/Breeze-7B-Instruct-v0_1' --tool aria2c -x 16

#Get response from the original model (before fine-tuning)

tokenizer = AutoTokenizer.from_pretrained('MediaTek-Research/Breeze-7B-Instruct-v0_1')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

def data_formulate(data):
    messages = [
        {"role": "system", "content": 'Reply in less than 20 characters'},
        {"role": "user", "content": data['prompt']},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

original_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f"Question {id}:\n{data['prompt']}")

    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
    original_model_response.append(output_text)

    print(f"Response from original model:\n{output_text}\n")

#  Set parameters ### Modify this block

num_epoch = 1 
data_size = 50
support_ratio = 0

# support_ratio preference ratio for "realism support"; reflects your preference
# 0 means completely against realism
# 1 means completely support realism
# 0.1 means 10% support realism

# Prepare training data
# We split the dataset into two parts: support and oppose
# Build a training dataset containing preference pairs.
# With a total of 50 training samples, when support is set to 0.1,
# The first 50*0.1=5 samples are labeled as supporting realism,
# The remaining 45 samples are labeled as opposing realism.

# Select part of the data for training
training_data = full_data[:data_size]

# Define the size of the support dataset
# part of the data is labeled as "support" (chosen), the rest as "oppose" (rejected
support_data_size = int(data_size * support_ratio)

# Prepare the data for the training dataset
prompt_list = [data_formulate(data) for data in training_data]
chosen_list = [data['support'] for data in training_data[:support_data_size]] + [data['oppose'] for data in training_data[support_data_size:]]
rejected_list = [data['oppose'] for data in training_data[:support_data_size]] + [data['support'] for data in training_data[support_data_size:]]
position_list = ['support' for _ in range(support_data_size)] + ['oppose' for _ in range(data_size - support_data_size)]

# Create the training dataset 创建训练数据集
train_dataset = Dataset.from_dict({'prompt': prompt_list, 'position': position_list, 'chosen': chosen_list, 'rejected': rejected_list})
pd.DataFrame(train_dataset).rename(columns={"chosen": "preferred", "rejected": "non-preferred"})

# Training

# Set training parameters
training_args = DPOConfig(
    output_dir='./',
    per_device_train_batch_size=1,
    num_train_epochs=num_epoch,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps = 1,
    warmup_ratio = 0.1,
    beta=0.1,
    report_to = 'none',
    
    # Explicitly declared to avoid errors
    max_length=512,
    max_prompt_length=128,
    remove_unused_columns=False,
)

# Configure PEFT
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Initialize DPO trainer:
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# start DPO training
dpo_trainer.train()

# Get response from the trained model

trained_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f"Question {id}:\n{data['prompt']}")

    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
    trained_model_response.append(output_text)

    print(f"Response from trained model:\n{output_text}\n")

# observe the output 

model_response = []
print(f'num_epoch: {num_epoch}\ndata_size: {data_size}\nsupport_ratio: {support_ratio}')
print()
for data in test_data:
    id = data['id']
    ref_output = original_model_response[id-1]
    output = trained_model_response[id-1]
    print(f'Question {id}:\n'+data['prompt'])
    print('Response from original model:\n'+ref_output)
    print('Response from trained model:\n'+output)
    print()
    model_response.append({'id':data['id'], 'prompt':data['prompt'], 'response_from_original_model':ref_output, 'response_from_trained_model':output})

# get output file
with open(f"epoch-{num_epoch}_size-{data_size}_ratio-{support_ratio}.json", "w", encoding='UTF-8') as outfile:
    json.dump(model_response, outfile, indent=4, ensure_ascii=False)
