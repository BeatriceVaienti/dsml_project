# train_combined.py

import sys
import os
sys.path.append('./')
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.data_processing import drop_missing_remove_duplicates
from utils.data_augmentation import augment_df
from utils.embeddings_generation import generate_embeddings
from utils.label_encoding import get_encoded_y
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import CamembertTokenizer, CamembertConfig
from torch.optim import AdamW
from models.model_combined import CamembertWithFeatures
import argparse
import torch.nn as nn
import joblib  # For saving the scaler

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a text classifier model on French sentences with additional features.')
    parser.add_argument('--model', type=str, choices=['camembert'], required=True, help='Choose the model to use: camembert')
    return parser.parse_args()

def prepare_full_data(df, tokenizer, scaler, max_len, batch_size=32):
    df = df.copy()
    df = drop_missing_remove_duplicates(df)
    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()

    # Extract and scale additional features
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    features = df[feature_columns].values
    features = scaler.transform(features)

    # Tokenize text data
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    # Convert to tensors
    inputs = input_ids.clone().detach()
    masks = attention_masks.clone().detach()
    labels = torch.tensor(labels)
    features = torch.tensor(features, dtype=torch.float32)

    data = TensorDataset(inputs, masks, features, labels)
    dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)
    return dataloader
import tqdm
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for step, batch in tqdm.tqdm(enumerate(dataloader)):
        
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_features, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, b_input_mask, b_features)
        loss = nn.CrossEntropyLoss()(outputs, b_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss

if __name__ == "__main__":
    args = get_arguments()
    model_choice = args.model
    batch_size = 64
    lr = 5e-5
    feature_dim = 7  # Number of additional features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_choice == 'camembert':
        model_path = './models_saved/camembert_full'
        tokenizer = CamembertTokenizer.from_pretrained(model_path)

    max_len = 264

    augmented_data_path = "./training/training_data_augmented.csv"
    scaler_path = './models_saved/scaler.pkl'
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    if os.path.exists(augmented_data_path) and os.path.exists(scaler_path):
        print("Loading existing augmented dataset and scaler...")
        df_augmented = pd.read_csv(augmented_data_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Creating augmented dataset and fitting scaler...")
        # Load and prepare the augmented dataset
        df = pd.read_csv('./training/training_data.csv')
        df = drop_missing_remove_duplicates(df)
        df_augmented = augment_df(df)
        df_augmented = generate_embeddings(df_augmented, chosen_tokenizer='camembert', batch_size=32)
        df_augmented.to_csv(augmented_data_path, index=False)
        print(f"Augmented dataset with embeddings saved to {augmented_data_path}")
        # Scaling of the selected features
        scaler = StandardScaler()
        scaler = scaler.fit(df_augmented[feature_columns].values)
    
    # Transform the features using the scaler, notice the corrected .values() to .values
    scaled_feature_columns = scaler.transform(df_augmented[feature_columns].values)
    # Convert the scaled features back into a DataFrame
    df_scaled_feature_columns = pd.DataFrame(scaled_feature_columns, columns=feature_columns)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    additional_columns = df_augmented[['sentence', 'difficulty']]
    df_full = pd.concat([df_scaled_feature_columns, additional_columns], axis=1)

    # Prepare the DataLoader
    full_dataloader = prepare_full_data(df_full, tokenizer, scaler, max_len, batch_size)

    model = CamembertWithFeatures(num_labels=6, feature_dim=feature_dim, model_path=model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    epochs = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, full_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")
        
    # Save the model and tokenizer

    model_save_path = f'./models_saved/{model_choice}_full_with_features'
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(model_save_path)
    
    # Save the configuration
    config = CamembertConfig.from_pretrained(model_path)
    config.save_pretrained(model_save_path)
    
    print(f"Model, tokenizer, and config saved to {model_save_path}")
