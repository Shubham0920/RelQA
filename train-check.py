from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_dir = './results/checkpoint-8750'

# Load the tokenizer from the original model name (e.g., 't5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load the T5 model from the saved model directory
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate SQL from a natural language query
def generate_sql(query, tokenizer, model):
    
    input_text = "translate English to SQL: " + query
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate output with adjusted generation parameters
    output = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        temperature=0.7,
        top_k=50,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Load the Spider dataset
from datasets import load_dataset
dataset = load_dataset("spider")

# Loop over examples in the training set and compare predictions
print("Checking predictions on training set...")
# Optionally, save the results to a file for further analysis
with open("prediction_results.txt", "w") as f:
    for example in dataset["train"]:
        query = example["question"]
        true_sql = example["query"]

        # Generate predicted SQL
        predicted_sql = generate_sql(query, tokenizer, model)

        # Print the query, actual SQL, and predicted SQL
        print("\nInput Question:", query)
        print("Expected SQL:", true_sql)
        print("Predicted SQL:", predicted_sql)
        f.write(f"Input Question: {query}\n")
        f.write(f"Expected SQL: {true_sql}\n")
        f.write(f"Predicted SQL: {predicted_sql}\n")
        f.write("\n" + "="*50 + "\n")

        
