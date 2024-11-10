from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# Directory where the fine-tuned model is saved
model_dir = './results/checkpoint-8750'

# Load the tokenizer from the original model name (e.g., 't5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load the T5 model from the saved model directory
model = T5ForConditionalGeneration.from_pretrained(model_dir)
# model.eval()
def generate_sql(query):
    # Prepare the input for the model
    input_text = "translate English to SQL: " + query
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the output using the model
    output = model.generate(
        input_ids,
        max_length=128,      # Maximum length of the generated SQL query
        num_beams=5,         # Beam search for better quality outputs
        early_stopping=True  # Stop early when a complete sequence is generated
    )

    # Decode the generated tokens into a human-readable string
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output
query = "your are given a Database where you have"
generated_sql = generate_sql(query)
print("Generated SQL:", generated_sql)