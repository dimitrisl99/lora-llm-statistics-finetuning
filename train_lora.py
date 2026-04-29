import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model  # PEFT library (LoRA)

#===============================
# Config
#===============================

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "outputs/smollm2-mathstats-lora"

MAX_LENGTH = 256
SEED = 42

# -----------------------
# LoRA hyperparams
# -----------------------

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

# -----------------------
# Training hyperparams (CPU-friendly)
# -----------------------

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 2e-4
EPOCHS = 1


def format_example(example):
    """
    Μετατρέπουμε (instruction, input, output) σε ένα training text.
    Το μοντέλο θα μάθει ΜΟΝΟ το answer (mask στο prompt).
    """
    instruction = (example.get("instruction") or "").strip()
    user_input = (example.get("input") or "").strip()
    answer = (example.get("output") or "").strip()

    prompt = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Question:\n"
        f"{user_input}\n\n"
        "### Answer:\n"
    )

    # Το πλήρες κείμενο που θα δει το μοντέλο (prompt + answer)
    full_text = prompt + answer

    return {
        "prompt": prompt,
        "full_text": full_text,
    }


def tokenize_and_mask(example, tokenizer):
    """
    Tokenize το full_text.
    Labels = input_ids, αλλά κάνουμε mask (-100) στα tokens του prompt,
    ώστε το loss να υπολογίζεται μόνο στο answer.
    """

    # Tokenize μόνο το prompt (για να ξέρουμε πόσα tokens να αγνοήσουμε)
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    # Tokenize όλο το κείμενο (prompt + answer)
    full = tokenizer(
        example["full_text"],
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Labels: ίδια με input_ids,
    # αλλά βάζουμε -100 στα prompt tokens για να αγνοηθούν στο loss
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len  # ignore prompt tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    # -----------------------
    # Setup
    # -----------------------

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # αν δεν υπάρχει ο φάκελος, δημιουργείται

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # -----------------------
    # Load tokenizer & model
    # -----------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Για causal LMs συνήθως κάνουμε pad_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)  # μεταφέρουμε όλα τα tensors στη σωστή συσκευή

    # -----------------------
    # LoRA config
    # -----------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",            # δεν εκπαιδεύουμε bias parameters
        task_type="CAUSAL_LM",  # causal language model
        target_modules="all-linear",  # ασφαλές default
    )

    # Παίρνουμε το base model, παγώνουμε τα weights,
    # και προσθέτουμε trainable LoRA layers
    model = get_peft_model(model, lora_config)

    # Sanity check: πόσες παράμετροι είναι trainable
    model.print_trainable_parameters()

    # -----------------------
    # Load dataset
    # -----------------------

    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(format_example)

    # -----------------------
    # Tokenize + mask (χωρίς lambda)
    # -----------------------

    def tokenize_map_fn(example):
        return tokenize_and_mask(example, tokenizer)

    ds = ds.map(tokenize_map_fn, remove_columns=ds.column_names)

    # -----------------------
    # Training arguments
    # -----------------------

    fp16 = torch.cuda.is_available()  # μόνο αν υπάρχει GPU

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,                     # πού σώζονται checkpoints
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # για μικρή μνήμη
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=10,                          # κάθε 10 steps τύπωσε loss
        save_steps=100,                            # save checkpoint κάθε 100 steps
        save_total_limit=2,
        report_to="none",
        fp16=fp16,
        optim="adamw_torch",                       # optimizer
    )

    # -----------------------
    # Data collator (padding σε batch)
    # -----------------------

    def collate_fn(features):
        return tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

    # -----------------------
    # Trainer
    # -----------------------

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate_fn,
    )

    # -----------------------
    # Train
    # -----------------------

    trainer.train()

    # -----------------------
    # Save LoRA adapters + tokenizer
    # -----------------------

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[DONE] Saved LoRA adapters to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
