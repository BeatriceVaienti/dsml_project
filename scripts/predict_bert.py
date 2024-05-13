import sys
sys.path.append('./')
import pandas as pd
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils.label_encoding import get_label_encoder
from transformers import CamembertTokenizer, FlaubertTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils.label_encoding import get_label_encoder
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict CEFR levels using a trained model.')
    parser.add_argument('--model', type=str, choices=['camembert', 'flaubert'], required=True, help='Specify the model to use: camembert or flaubert')
    return parser.parse_args()

def load_model(model_choice):
    if model_choice == 'camembert':
        model_path = './models_saved/camembert_full'
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
    elif model_choice == 'flaubert':
        model_path = './models_saved/flaubert_full'
        tokenizer = FlaubertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def prepare_inference_data(sentences, tokenizer, max_len):
    input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=max_len, truncation=True) for sent in sentences]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

    # Convert to tensors
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    data = TensorDataset(inputs, masks)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=32)
    return dataloader

def predict(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    # Convert logits to class predictions
    predictions = [list(p.argmax(axis=1)) for p in predictions]
    return [pred for sublist in predictions for pred in sublist]  # Flatten list

if __name__ == "__main__":
    args = get_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, tokenizer = load_model(args.model)
    
    inference = pd.read_csv('./test/unlabelled_test_data.csv')
    sentences = inference['sentence'].tolist()
    max_len = 128
    inference_dataloader = prepare_inference_data(sentences, tokenizer, max_len)
    
    predictions = predict(model, inference_dataloader, device)

    # Decode predictions
    encoder = get_label_encoder()
    decoded_predictions = encoder.inverse_transform(predictions)

    # Save or print predictions
    inference['difficulty'] = decoded_predictions
    # Drop the sentence column  
    inference = inference.drop(columns='sentence')
    # Optionally save to CSV
    inference.to_csv(f'./kaggle_submissions/predictions_{args.model}.csv', index=False)