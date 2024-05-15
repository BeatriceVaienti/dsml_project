# evaluate_combined.py

import sys
import os
sys.path.append('./')
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from utils.data_processing import drop_missing_remove_duplicates
from utils.data_augmentation import augment_df
from utils.embeddings_generation import generate_embeddings
from utils.label_encoding import get_encoded_y
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import CamembertTokenizer, CamembertConfig
from models.model_combined import CamembertWithFeatures
from torch.optim import AdamW
import torch.nn as nn
from itertools import product
from tqdm import tqdm, trange
import argparse
import json
import joblib  # For saving and loading the scaler

def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a trained text classifier model on French sentences with additional features.')
    parser.add_argument('--model', type=str, choices=['camembert'], required=True, help='Specify the model to use: camembert')
    parser.add_argument('--gpu', type=int, default=0, help='Specify the GPU to use')
    return parser.parse_args()

def prepare_data(df, tokenizer, scaler, max_len, batch_size=32):
    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()

    # Extract and scale additional features
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    features = df[feature_columns].values
    features = scaler.transform(features)

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

    inputs = input_ids.clone().detach()
    masks = attention_masks.clone().detach()
    labels = torch.tensor(labels)
    features = torch.tensor(features, dtype=torch.float32)

    data = TensorDataset(inputs, masks, features, labels)
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)
    return dataloader

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions, true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_features, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, b_input_mask, b_features)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    return predictions, true_labels

def train(model, train_dataloader, optimizer, device, epochs):
    model.to(device)
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        total_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_features, b_labels = batch
            optimizer.zero_grad()
            logits = model(b_input_ids, b_input_mask, b_features)
            loss = nn.CrossEntropyLoss()(logits, b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

def perform_hyperparameter_search(train_df, val_df, tokenizer, device, model_choice):
    learning_rates = [1e-5, 5e-5, 1e-4]
    batch_sizes = [16, 32, 64]
    epoch_lengths = [2, 4, 6]
    best_metrics = {'accuracy': 0}
    best_params = {}
    log_file_path = f'./best_hyperparameters_saved/hyperparameter_log_{model_choice}_combined.json'
    all_combinations = list(product(learning_rates, batch_sizes, epoch_lengths))

    # Scaling of the selected features
    scaler = StandardScaler()
    train_features = train_df[['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']]
    scaler.fit(train_features)
    train_df[['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']] = scaler.transform(train_features)
    val_df[['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']] = scaler.transform(val_df[['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']])

    for lr, batch_size, epochs in tqdm(all_combinations, desc='Hyperparameter Search Progress'):
        train_dataloader = prepare_data(train_df, tokenizer, scaler, max_len=264, batch_size=batch_size)
        val_dataloader = prepare_data(val_df, tokenizer, scaler, max_len=264, batch_size=batch_size)
        model = CamembertWithFeatures(num_labels=6, feature_dim=7, model_path='./models_saved/camembert_full')
        optimizer = AdamW(model.parameters(), lr=lr)

        train(model, train_dataloader, optimizer, device, epochs)

        predictions, true_labels = evaluate(model, val_dataloader, device)
        predictions = [np.argmax(pred) for pred in predictions]
        accuracy = accuracy_score(true_labels, predictions)

        # Log current hyperparameters and their accuracy
        log_hyperparameters({
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epochs
        }, accuracy, log_file_path)

        if accuracy > best_metrics['accuracy']:
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
            conf_matrix = confusion_matrix(true_labels, predictions)
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
            }
            best_params = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': epochs
            }
            print(f"New best model found: {best_params} with Accuracy: {accuracy}")

    save_hyperparameters(best_params, best_metrics, f'./best_hyperparameters_saved/best_hyperparameters_{model_choice}_combined.json')
    return best_params, best_metrics

def log_hyperparameters(hyperparameters, accuracy, file_path):
    data = {
        'hyperparameters': hyperparameters,
        'accuracy': accuracy
    }
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')
    print(f"Logged hyperparameters and accuracy to {file_path}")

def save_hyperparameters(hyperparameters, metrics, file_path):
    data = {
        'best_parameters': hyperparameters,
        'metrics': metrics
    }
    with open(file_path, 'w') as file:
        json.dump(data, file)
    print(f"Saved best hyperparameters and evaluation metrics to {file_path}")

if __name__ == "__main__":
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    model_choice = args.model
    print(model_choice)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_choice == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)

    # Load the data
    df = pd.read_csv('./training/training_data.csv')
    df = drop_missing_remove_duplicates(df)

    # Split the data into training, validation, and test sets
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Augment and generate embeddings for the training data
    df_augmented = augment_df(train_df)
    df_augmented = generate_embeddings(df_augmented, chosen_tokenizer='camembert', batch_size=32)
    
    # Save the augmented training data
    augmented_data_path = "./training/training_data_augmented.csv"
    df_augmented.to_csv(augmented_data_path, index=False)
    print(f"Augmented dataset with embeddings saved to {augmented_data_path}")

    # Scaling of the selected features using only the training data
    scaler = StandardScaler()
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    scaler.fit(df_augmented[feature_columns].values)
    scaler_path = './models_saved/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Apply the scaler to both the training and validation data
    df_augmented[feature_columns] = scaler.transform(df_augmented[feature_columns].values)
    val_df[feature_columns] = scaler.transform(val_df[feature_columns].values)
    
    # Tune hyperparameters on the validation set
    best_hyperparameters, best_metrics = perform_hyperparameter_search(df_augmented, val_df, tokenizer, device, model_choice)
    print("Best hyperparameters found:", best_hyperparameters)
    print("Best metrics:", best_metrics)

    # Train the final model on the entire training set with the best hyperparameters
    train_dataloader = prepare_data(train_val_df, tokenizer, scaler, max_len=264, batch_size=best_hyperparameters['batch_size'])
    model = CamembertWithFeatures(num_labels=6, feature_dim=7, model_path='./models_saved/camembert_full').to(device)
    optimizer = AdamW(model.parameters(), lr=best_hyperparameters['learning_rate'])
    train(model, train_dataloader, optimizer, device, best_hyperparameters['epochs'])

    # Evaluate the final model on the test set
    test_dataloader = prepare_data(test_df, tokenizer, scaler, max_len=264, batch_size=best_hyperparameters['batch_size'])
    predictions, true_labels = evaluate(model, test_dataloader, device)
    predictions = [np.argmax(pred) for pred in predictions]
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Save final evaluation metrics
    final_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
    }
    save_hyperparameters(best_hyperparameters, final_metrics, f'./best_hyperparameters_saved/final_metrics_{model_choice}.json')

    print("Final evaluation metrics on the test set:", final_metrics)
