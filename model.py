from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch


# Load the Spider dataset from the Hugging Face Hub
dataset = load_dataset("spider")

# Split the dataset into train and validation sets
train_dataset = dataset['train']
eval_dataset = dataset['validation']
# Load T5 tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def preprocess_function(examples):
    # Combine database ID for context (can be useful if needed) and question.
    inputs = ["translate English to SQL: " + question for question in examples['question']]
    targets = examples['query']  # The corresponding SQL query

    # Tokenize the inputs and labels
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    # Add labels to the input dictionary
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply the preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model checkpoints
    evaluation_strategy="epoch",     # Evaluate after each epoch
    learning_rate=3e-4,
    per_device_train_batch_size=8,   # Adjust based on your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,              # Limit the number of saved checkpoints
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=50,
    load_best_model_at_end=True,
    save_strategy="epoch", # Add this line to save at the end of each epoch
    metric_for_best_model="eval_loss", # Add this line for the evaluation metric
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

results = trainer.evaluate()
print("Evaluation results:", results)

def generate_sql(query):
    # Define device within the function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_text = "translate English to SQL: " + query
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device) # Move input_ids to the same device as the model

    output = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Example test
query = "Show me the names of employees in the 'Sales' department."
print("Generated SQL:", generate_sql(query))
