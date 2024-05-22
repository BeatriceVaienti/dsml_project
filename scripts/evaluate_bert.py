import sys
sys.path.append('./')
from itertools import product
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import torch
from utils.data_processing import drop_missing_remove_duplicates
from utils.label_encoding import get_encoded_y
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification, FlaubertTokenizer, FlaubertForSequenceClassification

from models.model_bert import initialize_model, get_optimizer
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model to predict CEFR levels of French sentences.')
    parser.add_argument('--model', type=str, choices=['camembert', 'camembert-large', 'flaubert'], required=True, help='Choose the model to use: camembert, camembert-large, or flaubert')
    return parser.parse_args()

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(model, train_dataloader, optimizer, device, scaler, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    for step, batch in tqdm(enumerate(train_dataloader), desc="Training", total=len(train_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
        
        scaler.scale(loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss

def evaluate(model, validation_dataloader, device):
    model.eval()
    eval_accuracy, nb_eval_steps = 0, 0
    all_preds, all_labels = [], []
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
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(label_ids)

    avg_accuracy = eval_accuracy / nb_eval_steps
    return avg_accuracy, all_preds, all_labels

def train_and_evaluate(model, train_dataloader, validation_dataloader, optimizer, device, epochs, scaler, gradient_accumulation_steps=1):
    model.to(device)
    best_metrics = {}
    for epoch in trange(epochs, desc="Epoch"):
        train_loss = train(model, train_dataloader, optimizer, device, scaler, gradient_accumulation_steps)
        avg_accuracy, all_preds, all_labels = evaluate(model, validation_dataloader, device)

        if avg_accuracy > best_metrics.get('accuracy', 0):
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
            conf_matrix = confusion_matrix(all_labels, all_preds)
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': avg_accuracy,
                'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
            }

    return best_metrics

def save_hyperparameters(hyperparameters, metrics, file_path):
    data = {
        'best_parameters': hyperparameters,
        'metrics': metrics
    }
    with open(file_path, 'a') as file:  # Changed to 'a' for append mode
        json.dump(data, file)
        file.write('\n')  # Ensure each entry is on a new line
    print(f"Saved best hyperparameters and evaluation metrics to {file_path}")

def log_hyperparameters(hyperparameters, accuracy, file_path):
    # Append hyperparameter performance to the log file
    data = {
        'hyperparameters': hyperparameters,
        'accuracy': accuracy
    }
    with open(file_path, 'a') as file:  # Changed to 'a' for append mode
        file.write(json.dumps(data) + '\n')
    print(f"Logged hyperparameters and accuracy to {file_path}")

def prepare_data(df, tokenizer, max_length=264):
    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()

    encoded_dict = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',  # Automatically pad to the specified max_length
        truncation=True,  # Ensure truncation to the specified max_length
        max_length=max_length,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    inputs = input_ids.clone().detach()
    masks = attention_masks.clone().detach()
    labels = torch.tensor(labels)

    return inputs, masks, labels

def create_dataloaders(train_inputs, validation_inputs, test_inputs, train_masks, validation_masks, test_masks, train_labels, validation_labels, test_labels, batch_size):
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, pin_memory=True)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, pin_memory=True)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, pin_memory=True)

    return train_dataloader, validation_dataloader, test_dataloader

def perform_hyperparameter_search(train_df, val_df, tokenizer, device, model_choice):
    learning_rates = [5e-05, 1e-05]
    batch_sizes = [45, 40, 32, 16]  
    epoch_lengths = [10, 16, 20, 32]
    best_metrics = {'accuracy': 0}  # Initialize with low accuracy to ensure replacement
    best_params = {}
    log_file_path = f'./hyperparameters_log/hyperparameters_log_{model_choice}.json'
    all_combinations = list(product(learning_rates, batch_sizes, epoch_lengths))

    train_inputs, train_masks, train_labels = prepare_data(train_df, tokenizer)
    val_inputs, val_masks, val_labels = prepare_data(val_df, tokenizer)

    for lr, batch_size, epochs in tqdm(all_combinations, desc='Hyperparameter Search Progress'):
        train_dataloader, validation_dataloader, _ = create_dataloaders(train_inputs, val_inputs, val_inputs, train_masks, val_masks, val_masks, train_labels, val_labels, val_labels, batch_size)
        model = initialize_model(6, device, model_choice)  # Dynamically initialize the model
        optimizer = get_optimizer(model, lr)

        # Using gradient accumulation
        gradient_accumulation_steps = max(1, 64 // batch_size)  # Simulate a batch size of 64
        scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler

        metrics = train_and_evaluate(model, train_dataloader, validation_dataloader, optimizer, device, epochs, scaler, gradient_accumulation_steps)

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

    save_hyperparameters(best_params, best_metrics, f'./hyperparameters_log/best_hyperparameters_eval_{model_choice}.json')
    return best_params, best_metrics

def log_hyperparameters(hyperparameters, accuracy, file_path):
    # Append hyperparameter performance to the log file
    data = {
        'hyperparameters': hyperparameters,
        'accuracy': accuracy
    }
    with open(file_path, 'a') as file:  # Changed to 'a' for append mode
        file.write(json.dumps(data) + '\n')
    print(f"Logged hyperparameters and accuracy to {file_path}")

if __name__ == "__main__":
    args = get_arguments()
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()  # Clear CUDA cache
    else:
        device = 'cpu'

    print('USED DEVICE: ', device)
    if args.model == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    elif args.model == 'camembert-large':
        tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-large', do_lower_case=True)
    else:  # flaubert
        tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', do_lower_case=True)

    df = pd.read_csv('./training/training_data.csv')
    df = drop_missing_remove_duplicates(df)

    # Split the data into training, validation, and test sets
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Tune hyperparameters on the validation set
    best_hyperparameters, best_metrics = perform_hyperparameter_search(train_df, val_df, tokenizer, device, args.model)
    print("Best hyperparameters found:", best_hyperparameters)
    print("Best metrics:", best_metrics)

    # Prepare data for final training and testing
    train_inputs, train_masks, train_labels = prepare_data(train_val_df, tokenizer)
    test_inputs, test_masks, test_labels = prepare_data(test_df, tokenizer)
    train_dataloader, _, test_dataloader = create_dataloaders(train_inputs, test_inputs, test_inputs, train_masks, test_masks, test_masks, train_labels, test_labels, test_labels, best_hyperparameters['batch_size'])

    model = initialize_model(6, device, args.model)
    optimizer = get_optimizer(model, best_hyperparameters['learning_rate'])
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    gradient_accumulation_steps = max(1, 64 // best_hyperparameters['batch_size'])  # Simulate a batch size of 64
    # Train the final model on the entire training set with the best hyperparameters

    train(model, train_dataloader, optimizer, device, scaler, gradient_accumulation_steps)
    avg_accuracy, all_preds, all_labels = evaluate(model, test_dataloader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Save final evaluation metrics
    final_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': avg_accuracy,
        'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
    }
    save_hyperparameters(best_hyperparameters, final_metrics, f'./hyperparameters_log/final_metrics_{args.model}.json')

    print("Final evaluation metrics on the test set:", final_metrics)