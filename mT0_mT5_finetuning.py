
######## adapted to finetune mT0 and mT5 #################

'''
pip install transformers
pip install evaluate
pip install datasets
pip install transformers[torch]
pip install accelerate -U
pip install numpy
'''

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

model_name = "bigscience/mt0-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

#load Squad-v2
dataset = load_dataset("squad_v2")
train_split = dataset["train"]
dev_split = dataset["validation"]

# Preprocessing function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Mapping start and end positions of answers to token positions
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        # If no answer, set positions to 0
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Get start and end char positions of answer
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Find token positions corresponding to answer positions
        token_start_index = 0
        while token_start_index < len(offset) and offset[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        token_end_index = len(offset) - 1
        while token_end_index >= 0 and offset[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# preproces train and dev splits of dataset
prep_train_data = train_split.map(preprocess_function, batched=True, remove_columns=train_split.column_names)

prep_val_data = dev_split.map(preprocess_function, batched=True, remove_columns=dev_split.column_names)

num_epochs = 3

# Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned/mT0",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,  
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=5, 
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir="./finetuned/mT0/logs", 
    logging_steps=10,  
    load_best_model_at_end=True,
    optim="adamw_torch",
    fp16=True,
)

model.gradient_checkpointing_enable()

trainer = Trainer (
    model=model,
    args=training_args,
    train_dataset=prep_train_data,
    eval_dataset=prep_val_data,
    tokenizer=tokenizer,
)

trainer.train()    

# Save tokenizer
tokenizer.save_pretrained("./finetuned/mT0")