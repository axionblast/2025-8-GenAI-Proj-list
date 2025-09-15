# Enable GPU
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# install packages
!pip install bitsandbytes==0.43.0
!pip install datasets==2.10.1
!pip install transformers==4.38.2
!pip install peft==0.9.0
!pip install sentencepiece==0.1.99
!pip install -U accelerate==0.28.0
!pip install colorama==0.4.6
!pip install fsspec==2023.9.2


import os
import sys
import argparse
import json
import warnings
import logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers, datasets
from peft import PeftModel
from colorama import *

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)



# Download  Dataset for Fine-tuning Training dataset
# reference:https:BA
!git clone https://github.com/sample/5.git

# Fix Random Seeds

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Define Some Useful Functions
""" It is recommmended NOT to change codes in this cell """

def generate_training_data(data_point):
    """
    (1) Goal:
        - This function is used to transform a data point (input and output texts) to tokens that our model can read

    (2) Arguments:
        - data_point: dict, with field "instruction", "input", and "output" which are all str

    (3) Returns:
        - a dict with model's input tokens, attention mask that make our model causal, and corresponding output targets

    (3) Example:
        - If you construct a dict, data_point_1, with field "instruction", "input", and "output" which are all str, you can use the function like this:
            formulate_article(data_point_1)

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing marketing Ad. 
<</SYS>>

{data_point["instruction"]}
{data_point["input"]}
[/INST]"""
    # count the number of input tokens
    len_user_prompt_tokens = (
        len(
            tokenizer(
                prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        ) - 1
    )
    # transform input prompt into tokens
    full_tokens = tokenizer(
        prompt + " " + data_point["output"] + "</s>",
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

# Evaluate generated responses
def evaluate(instruction, generation_config, max_len, input="", verbose=True):
    """
    (1) Goal:
        - This function is used to get the model's output given input strings

    (2) Arguments:
        - instruction: str, description of what you want model to do
        - generation_config: transformers.GenerationConfig object, to specify decoding parameters relating to model inference
        - max_len: int, max length of model's output
        - input: str, input string the model needs to solve the instruction, default is "" (no input)
        - verbose: bool, whether to print the mode's output, default is True

    (3) Returns:
        - output: str, the mode's response according to the instruction and the input

    (3) Example:
        - If you the instruction is "ABC" and the input is "DEF" and you want model to give an answer under 128 tokens, you can use the function like this:
            evaluate(instruction="ABC", generation_config=generation_config, max_len=128, input="DEF")

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Market Ad. 
<</SYS>>

{instruction}
{input}
[/INST]"""
    # Convert prompt text into the numeric representation required by the model
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    # Generate responses using the model
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )
    # Decode and print the generated responses
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split("[/INST]")[1].replace("</s>", "").replace("<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
        if (verbose):
            print(output)

    return output

# Download model and inference before fine-tuning
""" You may want (but not necessarily need) to change the LLM model """

model_name = "/content/TAIDE-LX-7B-Chat"                            #
#model_name = "MediaTek-Research/Breeze-7B-Instruct-v0_1"   
!wget -O taide_7b.zip "https://www.dropbox.com/scl/fi/harnetdwx2ttq1xt94rin/TAIDE-LX-7B-Chat.zip?rlkey=yzyf5nxztw6farpwyyildx5s3&st=s22mz5ao&dl=0"

!unzip taide_7b.zip

# Inference before Fine-tuning

""" It is recommmended NOT to change codes in this cell """

cache_dir = "./cache"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the pretrained language model from the specified model name or path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage = True
)

# Create tokenizer and set the end-of-sequence token (eos_token)
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token

# Set decoding parameters needed during model inference
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=2,
)


""" It is recommmended NOT to change codes in this cell """

# demo examples
test_tang_list = ['It is hard to meet, yet parting is just as hard; the east wind grows weak, and the blossoms fade away.']

# get the model output for each examples
demo_before_finetune = []
for tang in test_tang_list:
  demo_before_finetune.append(f'model input :\n"The following is the first line of a marketing Ad. Please use your knowledge to determine and complete the entire ad.
  {tang}\n\nmodel output:\n'+evaluate('"The following is the first line of a marketing Ad. Please use your knowledge to determine and complete the entire ad.', generation_config, max_len, , verbose = False))

# print and store the output to text file
for idx in range(len(demo_before_finetune)):
  print(f"Example {idx + 1}:")
  print(demo_before_finetune[idx])
  print("-" * 80)


# Set Hyperarameters for Fine-tuning

""" It is highly recommended you try to play around this hyperparameter """

num_train_data = 500
                

output_dir = "/content/drive/MyDrive" 
ckpt_dir = "./exp1" 
num_epoch = 1  
LEARNING_RATE = 3e-4  


""" It is recommmended NOT to change codes in this cell """

cache_dir = "./cache"  
from_ckpt = False  
ckpt_name = None  
dataset_dir = "./GenAI-Hw5/_training_data.json"  
logging_steps = 20 
save_steps = 65  
save_total_limit = 3  # checkpoint
report_to = None  
MICRO_BATCH_SIZE = 4  
BATCH_SIZE = 16  
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE 
CUTOFF_LEN = 256  
LORA_R = 8  # LORA（Layer-wise Random Attention）
LORA_ALPHA = 16  
LORA_DROPOUT = 0.05  
VAL_SET_SIZE = 0  
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"] 
device_map = "auto"  
world_size = int(os.environ.get("WORLD_SIZE", 1))  
ddp = world_size != 1  
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

#  Start Fine-tuning

""" It is recommmended NOT to change codes in this cell """

# create the output directory you specify
os.makedirs(output_dir, exist_ok = True)
os.makedirs(ckpt_dir, exist_ok = True)

# Load model weights from checkpoint based on the from_ckpt flag
if from_ckpt:
    model = PeftModel.from_pretrained(model, ckpt_name)

# Prepare the model for INT8 training
model = prepare_model_for_int8_training(model)

# Configure LORA model using LoraConfig
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Set the tokenizer's padding token to 0
tokenizer.pad_token_id = 0

# Load and process training data
with open(dataset_dir, "r", encoding = "utf-8") as f:
    data_json = json.load(f)
with open("tmp_dataset.json", "w", encoding = "utf-8") as f:
    json.dump(data_json[:num_train_data], f, indent = 2, ensure_ascii = False)

data = load_dataset('json', data_files="tmp_dataset.json", download_mode="force_redownload")

# Split the training data into training and validation sets (if VAL_SET_SIZE > 0)
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else:
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None

# Train the model using Transformers Trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=True,  
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,  # Whether to use DDP (Distributed Data Parallel) to control gradient update strategy
        report_to=report_to,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Disable the model’s cache functionality
model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

# Start model training
trainer.train()

# Save the trained model to the specified directory
model.save_pretrained(ckpt_dir)

# Print warnings about possible missing weights during training
print("\n If there's a warning about missing keys above, please disregard :)")

# Testing
""" It is recommmended NOT to change codes in this cell """

# find all available checkpoints
ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if (ckpt.startswith("checkpoint-")):
        ckpts.append(ckpt)

# list all the checkpoints
ckpts = sorted(ckpts, key = lambda ckpt: int(ckpt.split("-")[-1]))
print("all available checkpoints:")
print(" id: checkpoint name")
for (i, ckpt) in enumerate(ckpts):
    print(f"{i:>3}: {ckpt}")

""" You may want (but not necessarily need) to change the check point """

id_of_ckpt_to_use = -1  # The checkpoint ID to be used for inference (corresponds to the output of the previous cell)


ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])

""" You may want (but not necessarily need) to change decoding parameters """
max_len = 128   # Maximum length of generated responses
temperature = 0.1  # Set randomness of generation; lower values produce more stable outputs
top_p = 0.3  # Probability threshold for Top-p (nucleus) sampling, controls diversity of responses
# top_k = 5 # Adjust Top-k to increase response diversity and reduce repeated words

""" It is recommmended NOT to change codes in this cell """

test_data_path = "GenAI-Hw5/_testing_data.json"
output_path = os.path.join(output_dir, "results.txt")

cache_dir = "./cache"  # Set cache directory path
seed = 42  # Set random seed for reproducibility
no_repeat_ngram_size = 3  #Set size of no-repeat N-gram to prevent repeated segments

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Use tokenizer to convert model name into model-readable numeric representation
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)

# Load pretrained model and set it as an INT8 model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},  # Set the device to use, here specified as GPU 0
    cache_dir=cache_dir
)

# Load model weights from the specified checkpoint
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

""" It is recommmended NOT to change codes in this cell """

results = []

# Set generation configuration, including randomness, beam search, etc.
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=2
)

# read testing data
with open(test_data_path, "r", encoding = "utf-8") as f:
    test_datas = json.load(f)

# For each test sample, run prediction and save the results
with open(output_path, "w", encoding = "utf-8") as f:
  for (i, test_data) in enumerate(test_datas):
      predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose = False)
      f.write(f"{i+1}. "+test_data["input"]+predict+"\n")
      print(f"{i+1}. "+test_data["input"]+predict)

# prompt template
# The following is a product or scenario description. Please generate an engaging advertisement copy based on the description.

# Input:
# {product_desc}

# Output:

# Example 1

# Input:
#"New zero-sugar sparkling water, highlighting health, freshness, and no burden."

# Output:
# "Refreshing with every sip, zero sugar with no burden! Stay light and energized with our healthy sparkling water—drink freely, live freely."

# 🔹Example 2

# Input:
# "New smartwatch with all-day heart rate monitoring and fitness tracking features."

# Output:
# "Keep track of your heartbeat and challenge every step. The all-new smartwatch—where health meets performance."

# 🔹Example 3

# Input:
# "Premium coffee brand emphasizing farm-to-cup sourcing and handcrafted roasting."

# Output:
# "A great cup of coffee starts at the source. Hand-roasted to perfection, farm-direct freshness—crafted for true coffee lovers."

# using the same demo examples as before

test__list = ['New zero-sugar sparkling water, highlighting health, freshness, and no burden.']

# inference our fine-tuned model
demo_after_finetune = []
for  in test__list:
  demo_after_finetune.append(f'model input:\n"The following is the first line of Ad. Please use your knowledge to determine and complete the entire Ad.
  {}\n\model output:\n'+evaluate('"The following is the first line ofAd. Please use your knowledge to determine and complete the entire Ad.', generation_config, max_len, , verbose = False))

# print and store the output to text file
for idx in range(len(demo_after_finetune)):
  print(f"Example {idx + 1}:")
  print(demo_after_finetune[idx])
  print("-" * 80)
