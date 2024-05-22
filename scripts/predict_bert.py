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
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict CEFR levels using a trained model.')
    parser.add_argument('--model', type=str, choices=['camembert', 'camembert-large', 'flaubert'], required=True, help='Specify the model to use: camembert, camembert-large, or flaubert')
    return parser.parse_args()

def load_model(model_choice):
    if model_choice == 'camembert':
        model_path = './models_saved/camembert_full'
        model_path = './ensemble_model/camembert_full'
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
    elif model_choice == 'camembert-large':
        model_path = './models_saved/camembert-large_full'
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
    else:  # flaubert
        model_path = './models_saved/flaubert_full'
        tokenizer = FlaubertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def prepare_inference_data(sentences, tokenizer, max_len, batch_size=32):
    encoded_dict = tokenizer.batch_encode_plus(
        sentences,  # Batch of text to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        padding='max_length',  # Pad all to max_length.
        truncation=True,  # Explicitly truncate to max_length.
        return_attention_mask=True,  # Include attention masks.
        return_tensors='pt'  # Return pytorch tensors.
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    data = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)
    return dataloader

def predict(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)
    return predictions
if __name__ == "__main__":
    args = get_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, tokenizer = load_model(args.model)
    batch_size = 32  # Define your batch size
    inference = pd.read_csv('./test/unlabelled_test_data.csv')
    sentences = inference['sentence'].tolist()
    max_len = 264
    inference_dataloader = prepare_inference_data(sentences, tokenizer, max_len, batch_size=batch_size)
    
    predictions = predict(model, inference_dataloader, device)

    # Decode predictions
    encoder = get_label_encoder()
    decoded_predictions = encoder.inverse_transform(predictions)

    # Save or print predictions
    inference['difficulty'] = decoded_predictions
    # Drop the sentence column if desired
    inference = inference.drop(columns='sentence')
    # Optionally save to CSV
    inference.to_csv(f'./kaggle_submissions/predictions_bert_{args.model}.csv', index=False)