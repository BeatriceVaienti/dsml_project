import sys
import os
sys.path.append('./')
from itertools import product
from tqdm import trange, tqdm
import joblib
import pandas as pd
import numpy as np
import torch
from utils.data_processing import drop_missing_remove_duplicates
from utils.label_encoding import get_encoded_y
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification, FlaubertTokenizer, FlaubertForSequenceClassification
from models.model_nn import SimpleNN, prepare_nn_data, train_nn, evaluate_nn
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import argparse
from utils.data_augmentation import augment_df, get_top_pos_tags
from sklearn.preprocessing import StandardScaler
from utils.embeddings_generation import generate_embeddings

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model to predict CEFR levels of French sentences.')
    parser.add_argument('--gpu', type=int, default=1, help='Specify the GPU to use')

    return parser.parse_args()

def k_fold_cross_validation(df, scaler, device, hyperparameters, embedding_size, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for train_index, val_index in kfold.split(df):
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]

        train_dataloader = prepare_nn_data(train_df, scaler, batch_size=hyperparameters['batch_size'])
        val_dataloader = prepare_nn_data(val_df, scaler, batch_size=hyperparameters['batch_size'])

        model = SimpleNN(input_size=2 + embedding_size, hidden_size=hyperparameters['hidden_size'], num_classes=6).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'])

        train_nn(model, train_dataloader, optimizer, device, epochs=hyperparameters['epochs'])
        accuracy, _, _ = evaluate_nn(model, val_dataloader, device)
        results.append(accuracy)

    return np.mean(results), np.std(results)

def nn_grid_search(df, scaler, device, embedding_size):
    learning_rates = [1e-4, 5e-5]
    hidden_sizes = [32, 64]
    batch_sizes = [16, 32, 64, 128]
    epochs = [32, 64]

    best_hyperparameters = None
    best_mean_accuracy = 0
    log_file_path = './hyperparameters_log/nn_hyperparameter_log.json'

    for lr, hidden_size, batch_size, epoch in product(learning_rates, hidden_sizes, batch_sizes, epochs):
        hyperparameters = {
            'learning_rate': lr,
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'epochs': epoch
        }
        print(f"Evaluating hyperparameters: {hyperparameters}")
        mean_accuracy, std_accuracy = k_fold_cross_validation(df, scaler, device, hyperparameters, embedding_size)
        print(f"Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}")

        # Log current hyperparameters and their accuracy
        log_hyperparameters(hyperparameters, mean_accuracy, std_accuracy, log_file_path)

        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            best_hyperparameters = hyperparameters

    return best_hyperparameters, best_mean_accuracy


def log_hyperparameters(hyperparameters, mean_accuracy, std_accuracy, file_path):
    data = {
        'hyperparameters': hyperparameters,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }
    with open(file_path, 'a') as file:
        json.dump(data, file)
        file.write('\n')
    print(f"Logged hyperparameters and accuracy to {file_path}")



if __name__ == "__main__":
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('USED DEVICE: ', device)

    # Load and prepare data
    df = pd.read_csv('./training/training_data.csv')
    df = drop_missing_remove_duplicates(df)
    df = augment_df(df)

    # Initialize the scaler and fit on the entire dataset
    scaler = StandardScaler()
    feature_columns = ['n_words', 'avg_word_length']
    scaler.fit(df[feature_columns].values)

    # Apply scaling
    df[feature_columns] = scaler.transform(df[feature_columns].values)
    df, embedding_size = generate_embeddings(df)

    # Perform hyperparameter search
    best_hyperparameters, best_mean_accuracy = nn_grid_search(df, scaler, device, embedding_size)

    # Save the best hyperparameters
    results_path = './hyperparameters_log/nn_best_hyperparameters.json'
    with open(results_path, 'w') as f:
        json.dump({'best_hyperparameters': best_hyperparameters, 'best_mean_accuracy': best_mean_accuracy}, f, indent=4)
    print(f"Best hyperparameters saved to {results_path}")
    print(f"Best Mean Accuracy: {best_mean_accuracy}")