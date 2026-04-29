import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LORA_ADAPTER_PATH = "outputs/smollm2-mathstats-lora"

EVAL_FILE = "eval/eval_questions.jsonl"
RESULTS_DIR = "results"

MAX_NEW_TOKENS = 200


def load_questions(path):
    questions = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            questions.append(item["instruction"])

    return questions


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Answer:\n"


def generate_answer(model, tokenizer, instruction, device):
    prompt = format_prompt(instruction)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.5,
            top_p = 0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "### Answer:" in full_output:
        answer = full_output.split("### Answer:")[-1].strip()
    else:
        answer = full_output.strip()

    return answer


def save_outputs(path, outputs):
    with open(path, "w", encoding="utf-8") as file:
        for item in outputs:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluate_base_model(questions, tokenizer, device):
    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base_model.to(device)
    base_model.eval()

    results = []

    for i, question in enumerate(questions, start=1):
        print(f"[Base] Question {i}/{len(questions)}")

        answer = generate_answer(base_model, tokenizer, question, device)

        results.append({
            "instruction": question,
            "answer": answer
        })

    return results


def evaluate_lora_model(questions, tokenizer, device):
    print("Loading LoRA model...")

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    model.to(device)
    model.eval()

    results = []

    for i, question in enumerate(questions, start=1):
        print(f"[LoRA] Question {i}/{len(questions)}")

        answer = generate_answer(model, tokenizer, question, device)

        results.append({
            "instruction": question,
            "answer": answer
        })

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = load_questions(EVAL_FILE)

    base_outputs = evaluate_base_model(questions, tokenizer, device)
    save_outputs(
        os.path.join(RESULTS_DIR, "base_outputs.jsonl"),
        base_outputs
    )

    lora_outputs = evaluate_lora_model(questions, tokenizer, device)
    save_outputs(
        os.path.join(RESULTS_DIR, "lora_outputs.jsonl"),
        lora_outputs
    )

    print("Evaluation completed.")
    print("Saved:")
    print("- results/base_outputs.jsonl")
    print("- results/lora_outputs.jsonl")


if __name__ == "__main__":
    main()