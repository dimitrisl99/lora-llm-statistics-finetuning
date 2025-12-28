# LoRA Fine-Tuning of a Small LLM for Statistics and Probability Questions


## Overview

This project is a **learning-oriented implementation** of fine-tuning a small language model using **LoRA (Low-Rank Adaptation)**.

The goal of the project is **not performance**, but to understand:
- how instruction-style datasets are used,
- how fine-tuning works for causal language models,
- and how training and inference are separated in practice.

The project was built as a beginner-level but complete fine-tuning pipeline.

## Base Model
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Type: Causal Language Model
- Size: ~135M parameters

The model was chosen because it is small enough to be fine-tuned on a CPU.

## Dataset
The dataset is a **small custom JSONL file** with questions related to:
- probability
- statistics
- basic data analysis concepts

Each example has the form:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```
The dataset is intentionally small and manually written, in order to focus on understanding the fine-tuning process.

## Prompt Format

Each example is converted into the following prompt structure: 
```
### Instruction:
...

### Question:
...

### Answer:
```
This format is used consistently during training and inference. 

## Tokenization and Label Masking

The full text (prompt + answer) is tokenized.

During training:

- tokens belonging to the prompt are masked using label value -100
- only tokens from the answer contribute to the loss

This ensures that the model learns how to generate answers given a fixed prompt.

## Fine-Tuning Method

The model is fine-tuned using LoRA (Parameter-Efficient Fine-Tuning).

Key points:

- base model weights are frozen
- small LoRA adapters are added to linear layers
- only LoRA parameters are updated during training

This allows training with limited compute resources.

## Training Setup

Training is implemented using Hugging Face Trainer.

**Main settings:**

- Batch size: 1
- Gradient accumulation: 8
- Learning rate: 2e-4
- Epochs: 1
- Training device: CPU

The purpose is to verify that the training pipeline works end-to-end.

## Inference 

Inference is implemented in a separate script.

The inference script:

- loads the base model
- loads the trained LoRA adapters
- generates answers for the same prompt format
- allows comparison between base and LoRA-adapted outputs

No further training happens during inference.

## Results

Due to the small dataset and limited training:

- improvements are modest
- the LoRA-adapted model shows a slight shift toward more structured statistical language

The results are meant for demonstration and learning, not benchmarking.

## Limitations 

- Very small dataset
- Single training epoch 
- No automatic evaluation metrics 
- CPU-only training 

These limitations are intentional for learning purposes. 

## How to Run 

### Train 

python train_lora.py

### Inference 

python infer_lora.py

## What This Project Demonstrates

- Basic understanding of LLM fine-tuning
- Use of LoRA for parameter-efficient training
- Correct handling of prompt formatting and label masking
- Clear separation between training and inference
- Practical use of Hugging Face and PyTorch tools

