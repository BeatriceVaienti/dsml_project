import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = 'C:\\Users\\berret_c\\Downloads\\flaubert_full\\flaubert_full'

# Check available files
model_files = os.listdir(model_dir)
print("Files in model directory:", model_files)

# Load the tokenizer and the model for sequence classification
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print("Failed to load model and tokenizer:", e)