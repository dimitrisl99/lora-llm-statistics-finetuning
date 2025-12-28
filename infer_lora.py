import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_DIR = "outputs/smollm2-mathstats-lora"

def build_prompt(question: str) -> str:
    instruction = "Answer the question clearly and rigorously, using mathematical and statistical reasoning."
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:\n"
    )

def generate(model, tokenizer, prompt: str, max_new_tokens=140):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    question = "Explain the difference between a PDF and a CDF, and give a simple example of each."
    prompt = build_prompt(question)

    # Base model
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    base_model.eval()

    # LoRA model (base + adapters)
    lora_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if lora_tokenizer.pad_token is None:
        lora_tokenizer.pad_token = lora_tokenizer.eos_token
    lora_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, ADAPTER_DIR).to(device)
    lora_model.eval()

    print("\n================ BASE MODEL ================\n")
    print(generate(base_model, base_tokenizer, prompt))

    print("\n================ LoRA MODEL ================\n")
    print(generate(lora_model, lora_tokenizer, prompt))

if __name__ == "__main__":
    main()
