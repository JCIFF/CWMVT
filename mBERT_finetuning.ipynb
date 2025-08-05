!pip install datasets transformers

from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset, load_dataset

# Initialize the mBERT tokenizer and model
model_name = "google-bert/bert-base-multilingual-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Loading SQuAD-v1 dataset
try:
    dataset = load_dataset("squad", "plain_text", split="train")
except FileNotFoundError:
    print(f"Error: file not found ")
    
for param in model.parameters(): param.data = param.data.contiguous()

# Preprocessing dataset
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    # Compute start and end positions
    start_positions = []
    end_positions = []
    for i, offsets in enumerate(inputs["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        # Find start and end token indices
        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
        
        # If answer is not in context, set start and end positions to 0
        if start_char < offsets[context_start][0] or end_char > offsets[context_end][1]:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(next(idx for idx, offset in enumerate(offsets) if offset[0] <= start_char < offset[1]))
            end_positions.append(next(idx for idx, offset in enumerate(offsets) if offset[0] < end_char <= offset[1]))
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# dataset tokonization
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Defining training arguments
training_args = TrainingArguments(
    output_dir="./finetuned/mBERT",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=3,
    optim="adamw_torch",
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

#Save the finetuned model
train_results = trainer.train()
trainer.save_model()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
