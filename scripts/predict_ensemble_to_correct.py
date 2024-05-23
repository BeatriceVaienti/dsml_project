import os
import sys
sys.path.append('./')
import pandas as pd
import torch
import numpy as np
import joblib
import json
import lightgbm as lgb
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import CamembertTokenizer, FlaubertTokenizer, CamembertForSequenceClassification, FlaubertForSequenceClassification
from sklearn.preprocessing import StandardScaler
from utils.embeddings_generation import generate_embeddings
from utils.data_augmentation import augment_df
from utils.label_encoding import get_encoded_y, get_label_encoder
import torch.nn as nn
import argparse

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MetaNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MetaNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict CEFR levels using a trained ensemble model.')
    parser.add_argument('--meta_model', type=str, choices=['lgb', 'nn'], required=True, help='Meta model to use for prediction (lgb or nn).')
    return parser.parse_args()

def prepare_data(df, tokenizer, scaler, max_len, batch_size=32, use_features=False, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    text = df['sentence'].to_list()

    encoded_dict = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids'].to(device)
    attention_masks = encoded_dict['attention_mask'].to(device)

    if use_features:
        feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
        features = df[feature_columns].values
        features = scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32).to(device)
        data = TensorDataset(input_ids, attention_masks, features)
    else:
        data = TensorDataset(input_ids, attention_masks)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    return dataloader

def prepare_nn_inference_data(df, scaler, embedding_size, batch_size=32):
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    features = df[feature_columns].values
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    embeddings = np.vstack(df['embeddings'].values)
    combined_features = np.hstack((features, embeddings))
    combined_features = torch.tensor(combined_features, dtype=torch.float32)

    data = TensorDataset(combined_features)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)
    return dataloader

def evaluate_model(model, dataloader, device, use_features=False):
    model.eval()
    model.to(device)
    predictions = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        if use_features:
            b_input_ids, b_input_mask, b_features = batch
            with torch.no_grad():
                outputs = model(b_input_ids.long(), attention_mask=b_input_mask.long())
                logits = outputs.logits
        else:
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                outputs = model(b_input_ids.long(), attention_mask=b_input_mask.long())
                logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        torch.cuda.empty_cache()  # Clear cache after each batch
    predictions = [item for sublist in predictions for item in sublist]
    return np.array(predictions)

def evaluate_nn(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            torch.cuda.empty_cache()  # Clear cache after each batch
    return np.array(predictions)

def load_top_pos_tags(filepath):
    with open(filepath, 'r') as file:
        top_tags = json.load(file)
    return top_tags

def load_models_and_data(meta_model_choice):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load tokenizers
    camembert_tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

    # Load models
    camembert_model = CamembertForSequenceClassification.from_pretrained('ensemble_model/camembert_full').to(device)
    flaubert_model = FlaubertForSequenceClassification.from_pretrained('ensemble_model/flaubert_full').to(device)
    embedding_size = 768 
    nn_model = SimpleNN(input_size=7 + embedding_size, hidden_size=64, num_classes=6).to(device)
    nn_model.load_state_dict(torch.load('ensemble_model/simple_nn.pth'))

    # Load scaler
    scaler = joblib.load('ensemble_model/scaler.pkl')

    # Load meta model based on choice
    if meta_model_choice == 'lgb':
        meta_model = lgb.Booster(model_file='ensemble_model/meta_classifier_with_features.txt')
    elif meta_model_choice == 'nn':
        meta_model = MetaNN(input_size=3, hidden_size=64, num_classes=6).to(device)
        meta_model.load_state_dict(torch.load('ensemble_model/meta_nn_with_features.pth'))

    # Load additional files
    top_pos_tags = load_top_pos_tags('ensemble_model/top_pos_tags.json')

    return camembert_tokenizer, flaubert_tokenizer, camembert_model, flaubert_model, nn_model, scaler, meta_model, top_pos_tags, device

def inference(df, camembert_tokenizer, flaubert_tokenizer, camembert_model, flaubert_model, nn_model, scaler, meta_model, top_pos_tags, device, meta_model_choice):
    df = augment_df(df, top_tags=top_pos_tags)
    df, embedding_size = generate_embeddings(df, chosen_tokenizer='camembert')

    camembert_dataloader = prepare_data(df, camembert_tokenizer, scaler, max_len=264, batch_size=32, use_features=True, device=device)
    flaubert_dataloader = prepare_data(df, flaubert_tokenizer, scaler, max_len=264, batch_size=32, use_features=True, device=device)
    nn_dataloader = prepare_nn_inference_data(df, scaler, embedding_size, batch_size=32)

    camembert_predictions = evaluate_model(camembert_model, camembert_dataloader, device, use_features=True)
    flaubert_predictions = evaluate_model(flaubert_model, flaubert_dataloader, device, use_features=True)
    nn_predictions = evaluate_nn(nn_model, nn_dataloader, device)


    test_features = np.hstack([
        np.argmax(camembert_predictions, axis=1).reshape(-1, 1),
        np.argmax(flaubert_predictions, axis=1).reshape(-1, 1),
        nn_predictions.reshape(-1, 1)
    ])

    if meta_model_choice == 'lgb':
        meta_predictions = meta_model.predict(test_features)
        final_predictions = np.argmax(meta_predictions, axis=1)
    elif meta_model_choice == 'nn':
        meta_model.eval()
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = meta_model(test_features_tensor)
            _, final_predictions = torch.max(outputs, 1)
        final_predictions = final_predictions.cpu().numpy()

    return final_predictions


if __name__ == "__main__":
    # Get arguments
    args = get_arguments()

    # Load models, data, and other necessary components
    camembert_tokenizer, flaubert_tokenizer, camembert_model, flaubert_model, nn_model, scaler, meta_model, top_pos_tags, device = load_models_and_data(args.meta_model)

    # Load unlabelled data
    unlabelled_df = pd.read_csv('./test/unlabelled_test_data.csv')

    # Perform inference
    predictions = inference(unlabelled_df, camembert_tokenizer, flaubert_tokenizer, camembert_model, flaubert_model, nn_model, scaler, meta_model, top_pos_tags, device, args.meta_model)

    # Decode predictions
    encoder = get_label_encoder()
    decoded_predictions = encoder.inverse_transform(predictions)
    
    # Add the 'difficulty' column to the DataFrame
    unlabelled_df['difficulty'] = decoded_predictions

    # Save the predictions
    unlabelled_df[['id', 'difficulty']].to_csv(f'./kaggle_submissions/meta_model_predictions.csv', index=False)
