# Install necessary packages
!pip install datasets
!pip install evaluate
!pip install py7zr

# Import necessary libraries
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import evaluate
import nltk
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
import pandas as pd
from tqdm import tqdm

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Load dataset
dataset = load_dataset("Samsung/samsum")
val_data = dataset["validation"]
test_data = dataset["test"]

# Combine the dataset
combined_data = concatenate_datasets([val_data, test_data])

# Split the dataset into 80% training and 20% testing
train_data, test_data = combined_data.train_test_split(test_size=0.2, shuffle=True).values()

# Delete unused data to save space
del dataset, val_data, combined_data

# Preprocess Dataset
def convert_examples_to_features(example_batch):
  input_encodings = tokenizer(example_batch['dialogue'], max_length = 1024, truncation = True)
  with tokenizer.as_target_tokenizer():
    target_encodings = tokenizer(example_batch['summary'], max_length = 128, truncation = True)
  return{
      'input_ids': input_encodings['input_ids'],
      'attention_mask': input_encodings['attention_mask'],
      'labels': target_encodings['input_ids']
  }

# Tokenize the train and testing dataset
train_data_pt = train_data.map(convert_examples_to_features, batched = True)
test_data_pt = test_data.map(convert_examples_to_features, batched = True)

# Define data collector
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

# Define training arguments
trainer_args = TrainingArguments(
    output_dir = "bart-base-samsum",
    run_name = "bart-base-samsum-run",
    num_train_epochs = 5,
    warmup_steps = 500,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    weight_decay = 0.01,
    logging_steps = 10,
    evaluation_strategy = 'steps',
    eval_steps = 500,
    save_steps = 1e6,
    gradient_accumulation_steps = 16
)

# Trainer setup
trainer = Trainer(
    model = model,
    args = trainer_args,
    tokenizer = tokenizer,
    data_collator = seq2seq_data_collator,
    train_dataset = train_data_pt,
    eval_dataset = test_data_pt
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("bart-base-samsum-model")
tokenizer.save_pretrained("bart-base-samsum-model")


# Prediction function
def summarize(sample_text):
    pipe = pipeline("summarization", model="distilbart-samsum-model", tokenizer=tokenizer)
    gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
    return pipe(sample_text, **gen_kwargs)[0]["summary_text"]

# Example of prediction
sample_text = test_data[1]["dialogue"]
reference = test_data[1]["summary"]

print("Dialogue:")
print(sample_text)
print("\nReference Summary:")
print(reference)
print("\nModel Summary:")
print(summarize(sample_text))