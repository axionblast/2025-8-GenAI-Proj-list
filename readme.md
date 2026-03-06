# LLM Value Alignment via Direct Preference Optimization (DPO)

This project demonstrates a small-scale experiment on aligning a large language model using **Direct Preference Optimization (DPO)** with **LoRA parameter-efficient fine-tuning**.

The goal is to explore how preference-labelled responses can shift a model's output behaviour while keeping training computationally lightweight.

## Core Training Script

The main implementation of the DPO alignment pipeline is located in:

[LLMs Value Alignment via Direct Preference Optimization.py](https://github.com/axionblast/2025-8-GenAI-Proj-list/blob/main/LLMs%20Value%20Alignment%20via%20Direct%20Preference%20Optimization%20(DPO).py)

This script includes:
- dataset preprocessing
- preference pair construction
- LoRA configuration
- DPO training with TRL
- comparison of model outputs before and after alignment

---

## Overview

This experiment:

- Uses the **Breeze-7B-Instruct** model
- Applies **LoRA-based fine-tuning** via the `TRL` DPO trainer
- Trains on a **preference-labelled dataset** containing:
  - `support` responses (preferred)
  - `oppose` responses (rejected)

The training objective encourages the model to increase the probability of preferred responses over rejected ones.

---

## Method

The workflow consists of the following steps:

1. Load the preference-labelled dataset
2. Convert raw JSON samples into DPO tuples:

(prompt, chosen, rejected)

3. Configure LoRA adapters for parameter-efficient training
4. Train using the **DPO loss objective**
5. Compare model responses **before and after alignment**

---

## Dataset

Dataset files:

dataset/labelled_data.json  
dataset/test_prompt.json  

Each sample contains:

- `prompt`
- `support` (preferred answer)
- `oppose` (rejected answer)

These are converted into preference pairs required by the DPO trainer.

---

## Training Setup

Key hyperparameters:

num_epoch = 1  
data_size = 50  
support_ratio = 0  
learning_rate = 2e-4  
LoRA rank = 64  
beta (DPO) = 0.1  

### Debug vs Experiment Runs

The repository contains **debug-scale experiments** (`data_size = 50`) used to verify the training pipeline.

Larger experiments used:

data_size = 500  
num_epoch = 2  

---

## Environment

Tested with:

Python 3.10  
PyTorch (CUDA enabled)  
CUDA 12.x  

Key libraries:

transformers  
trl  
peft  
datasets  
bitsandbytes  
accelerate  
numpy==1.26.4  

Install dependencies:

pip install -r requirements.txt

---

## Running the Experiment

Example command:

python dpo_alignment.py --num_epoch 1 --data_size 50 --support_ratio 0

After training, model responses before and after alignment are saved to:

epoch-{num_epoch}_size-{data_size}_ratio-{support_ratio}.json

---

## Example Output

The script prints responses from:

- the **original model**
- the **aligned model**

This allows quick qualitative inspection of how preference alignment changes generation behaviour.

---

## Project Structure

.
├── dataset  
│   ├── labelled_data.json  
│   └── test_prompt.json  
├── dpo_alignment.py  
├── requirements.txt  
└── README.md  

---

## Notes

This repository focuses on demonstrating the **alignment pipeline and training setup**, rather than large-scale training.

Small dataset sizes are intentionally used to make experiments reproducible on a single GPU.

---

## License

For research and educational purposes.
