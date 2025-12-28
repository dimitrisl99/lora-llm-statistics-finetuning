from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Explain the difference between accuracy and precision."

inputs = tokenizer(prompt, return_tensors="pt")

print("TOKENS (input_ids):")
print(inputs["input_ids"])

print("\nNUMBER OF TOKENS:")
print(inputs["input_ids"].shape[1])

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100
)

print("\nMODEL OUTPUT:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

