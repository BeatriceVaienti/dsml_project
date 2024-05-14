import sys
sys.path.append('./')
from itertools import product
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import torch
from utils.data_processing import drop_missing_remove_duplicates
from utils.label_encoding import get_encoded_y
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification, FlaubertTokenizer, FlaubertForSequenceClassification

from torch.optim import AdamW
from utils.data_loader import load_and_prepare_data, create_dataloaders
from models.model_camembert import initialize_model, get_optimizer
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from models.model_bert import initialize_model, get_optimizer
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model to predict CEFR levels of French sentences.')
    parser.add_argument('--model', type=str, choices=['camembert', 'flaubert'], required=True, help='Choose the model to use: camembert or flaubert')
    args = parser.parse_args()
    return args

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss

def evaluate(model, validation_dataloader, device):
    model.eval()
    eval_accuracy, nb_eval_steps = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    return eval_accuracy / nb_eval_steps

def train_and_evaluate(model, train_dataloader, validation_dataloader, optimizer, device, epochs):
    model.to(device)
    best_metrics = {}
    #show the progress with tqdm
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds)
            all_labels.extend(label_ids)

        accuracy = accuracy_score(all_labels, all_preds)
        if accuracy > best_metrics.get('accuracy', 0):
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
            conf_matrix = confusion_matrix(all_labels, all_preds)
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
            }

    return best_metrics

def save_hyperparameters(hyperparameters, metrics, file_path):
    data = {
        'best_parameters': hyperparameters,
        'metrics': metrics
    }
    with open(file_path, 'w') as file:
        json.dump(data, file)
    print(f"Saved best hyperparameters and evaluation metrics to {file_path}")

def perform_hyperparameter_search(df, tokenizer, device, model_choice):
    learning_rates = [1e-5, 5e-5, 1e-4]
    batch_sizes = [16, 32, 64]
    epoch_lengths = [2, 4, 8, 16]
    best_metrics = {'accuracy': 0}  # Initialize with low accuracy to ensure replacement
    best_params = {}
    log_file_path = f'./best_hyperparameters_saved/hyperparameter_log_{model_choice}.json'
    all_combinations = list(product(learning_rates, batch_sizes, epoch_lengths))

    for lr, batch_size, epochs in tqdm(all_combinations, desc='Hyperparameter Search Progress'):
        train_dataloader, validation_dataloader = create_dataloaders(*load_and_prepare_data(df, tokenizer, max_len=264), batch_size)
        model = initialize_model(6, device, model_choice)  # Dynamically initialize the model
        optimizer = get_optimizer(model, lr)

        metrics = train_and_evaluate(model, train_dataloader, validation_dataloader, optimizer, device, epochs)

        # Log current hyperparameters and their accuracy
        log_hyperparameters({
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epochs
        }, metrics['accuracy'], log_file_path)

        if metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = metrics
            best_params = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': epochs
            }
            print(f"New best model found: {best_params} with Accuracy: {metrics['accuracy']}")

    save_hyperparameters(best_params, best_metrics, f'./best_hyperparameters_saved/best_hyperparameters_{model_choice}.json')
    return best_params, best_metrics

def log_hyperparameters(hyperparameters, accuracy, file_path):
    # Append hyperparameter performance to the log file
    data = {
        'hyperparameters': hyperparameters,
        'accuracy': accuracy
    }
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')
    print(f"Logged hyperparameters and accuracy to {file_path}")

if __name__ == "__main__":
    args = get_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    else:  # flaubert
        tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', do_lower_case=True)

    df = pd.read_csv('./training/training_data.csv')
    best_hyperparameters = perform_hyperparameter_search(df, tokenizer, device, args.model)
    print("Best hyperparameters found:", best_hyperparameters)

