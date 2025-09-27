"""
Simple GPT-2 fine-tuning example on WikiText-2 dataset.
Runs on a single NVIDIA T4 GPU (16GB).
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 1. Load dataset (WikiText-2)
print("Downloading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. Load tokenizer and model
model_name = "gpt2"  # base GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 has no pad_token by default → set it
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Tokenize dataset
block_size = 128  # shorter for demo; increase for better results
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=block_size)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def group_texts(examples):
    # Concatenate all lists
    concatenated_inputs = sum(examples["input_ids"], [])
    concatenated_masks = sum(examples["attention_mask"], [])

    # Cut to a multiple of block_size
    total_length = (len(concatenated_inputs) // block_size) * block_size

    # Split into chunks
    input_ids = [concatenated_inputs[i:i+block_size] for i in range(0, total_length, block_size)]
    attention_mask = [concatenated_masks[i:i+block_size] for i in range(0, total_length, block_size)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),
    }


lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# 4. Data collator (for padding)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    eval_strategy="epoch",

    report_to="none",

    learning_rate=5e-5,
    per_device_train_batch_size=2,   # fits on T4
    per_device_eval_batch_size=2,
    num_train_epochs=1,              # small for demo; try 3+
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="steps",
    fp16=True,                       # use mixed precision on GPU
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 6. Train
print("Starting training...")
trainer.train()

# 7. Save model
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("Model saved to ./gpt2-finetuned")

# 8. Quick inference test
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer=tokenizer)
print(generator("Deep learning is", max_new_tokens=50))
