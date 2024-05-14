# predict_combined.py

import sys
sys.path.append('./')
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from models.model_combined import CamembertWithFeatures
from utils.label_encoding import get_label_encoder
import argparse
import joblib  # For loading the scaler

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict CEFR levels using a trained model with additional features.')
    parser.add_argument('--model', type=str, choices=['camembert'], required=True, help='Specify the model to use: camembert')
    return parser.parse_args()

def load_model_with_features(model_choice, feature_dim):
    if model_choice == 'camembert':
        model_path = './models_saved/camembert_full_with_features'
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
        model = CamembertWithFeatures(num_labels=6, feature_dim=feature_dim, model_path=model_path)
    return model, tokenizer

def prepare_inference_data(sentences, features, tokenizer, scaler, max_len, batch_size=32):
    # Scale the additional features
    features = scaler.transform(features)
    
    encoded_dict = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    features = torch.tensor(features, dtype=torch.float32)

    data = TensorDataset(input_ids, attention_masks, features)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)
    return dataloader

def predict(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_features = batch
        with torch.no_grad():
            outputs = model(b_input_ids, b_input_mask, b_features)
            logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)
    return predictions

if __name__ == "__main__":
    args = get_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = 7  # Number of additional features

    model, tokenizer = load_model_with_features(args.model, feature_dim)
    batch_size = 4

    # Load the scaler used during training
    scaler = joblib.load('./models_saved/scaler.pkl')

    inference = pd.read_csv('./test/unlabelled_test_data.csv')
    sentences = inference['sentence'].tolist()
    features = inference[['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']].values
    max_len = 264
    inference_dataloader = prepare_inference_data(sentences, features, tokenizer, scaler, max_len, batch_size=batch_size)

    predictions = predict(model, inference_dataloader, device)

    encoder = get_label_encoder()
    decoded_predictions = encoder.inverse_transform(predictions)

    inference['difficulty'] = decoded_predictions
    inference = inference.drop(columns='sentence')
    inference.to_csv(f'./kaggle_submissions/predictions_{args.model}_with_features.csv', index=False)
