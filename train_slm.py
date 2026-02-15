import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load dataset
with open("data/alpaca_dataset.json", "r") as f:
    data = json.load(f)

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

formatted_data = [format_example(x) for x in data]
dataset = Dataset.from_list(formatted_data)

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.eos_token_id


def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./models/slm",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=False  # Keep False for CPU
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./models/slm")
tokenizer.save_pretrained("./models/slm")

print("Fine-tuned SLM saved successfully.")
