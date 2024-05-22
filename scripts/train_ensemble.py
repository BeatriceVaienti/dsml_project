import sys
import os
sys.path.append('./')
import pandas as pd
import torch
import numpy as np
from utils.data_processing import drop_missing_remove_duplicates
from utils.data_augmentation import augment_df, get_top_pos_tags
from utils.label_encoding import get_encoded_y
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer, FlaubertTokenizer, CamembertForSequenceClassification, FlaubertForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
import joblib
import lightgbm as lgb
from tqdm import tqdm
import json
from utils.embeddings_generation import generate_embeddings
import argparse
from models.model_bert import initialize_model, get_optimizer
from models.model_nn import SimpleNN
from models.model_meta_nn import MetaNN, create_nn_dataloader, train_meta_nn, evaluate_meta_nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_processing import drop_missing_remove_duplicates




def get_arguments():
    parser = argparse.ArgumentParser(description='Train and combine CamemBERT and FlauBERT models with a meta-classifier.')
    parser.add_argument('--gpu', type=int, default=1, help='Specify the GPU to use')
    parser.add_argument('--use_nn', action='store_true', help='Use an additional neural network model with augmented features')
    return parser.parse_args()

def prepare_data(df, tokenizer, scaler, max_len, batch_size=32, use_features=False, shuffle=True, device = 'cpu'):
    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()

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

    # Convert labels to tensor
    labels = torch.tensor(labels).to(device)

    if use_features:
        feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
        features = df[feature_columns].values
        features = scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32).to(device)
        
        # Ensure the dimensions match by expanding dimensions if necessary
        if features.ndim == 1:
            features = features.unsqueeze(1)

        data = TensorDataset(input_ids, attention_masks, features, labels)
    else:
        data = TensorDataset(input_ids, attention_masks, labels)

    sampler = SequentialSampler(data) if not shuffle else RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    return dataloader

def prepare_nn_data_with_embeddings(df, scaler, batch_size=32, shuffle=True, device='cpu'):
    labels = get_encoded_y(df).tolist()
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    features = df[feature_columns].values
    features = scaler.transform(features)
    
    embeddings = np.vstack(df['embeddings'].values)

    # Combine features and embeddings
    combined_features = np.hstack((features, embeddings))
    
    combined_features = torch.tensor(combined_features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels).to(device)

    data = TensorDataset(combined_features, labels)
    sampler = SequentialSampler(data) if not shuffle else RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    return dataloader

def train_model(model, dataloader, optimizer, device, scaler,  num_epochs=1, use_features=False, gradient_accumulation_steps=1):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in tqdm(enumerate(dataloader), desc=f"Training Epoch {epoch+1}", total=len(dataloader)):
            batch = tuple(t.to(device) for t in batch)
            if use_features:
                b_input_ids, b_input_mask, b_features, b_labels = batch
            else:
                b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(b_input_ids.long(), attention_mask=b_input_mask.long(), labels=b_labels)
                loss = outputs.loss

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            torch.cuda.empty_cache()  # Clear cache after each batch
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

def train_nn(model, dataloader, optimizer, device, epochs=1):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in tqdm(dataloader, desc=f"Training NN - Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()  # Clear cache after each batch
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(dataloader)}")

def evaluate_nn(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions, true_labels = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            torch.cuda.empty_cache()  # Clear cache after each batch
    return predictions, true_labels

    # Evaluate models
def evaluate_model(model, dataloader, device, use_features=False):
    model.eval()
    model.to(device)
    predictions, true_labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        if use_features:
            b_input_ids, b_input_mask, b_features, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids.long(), attention_mask=b_input_mask.long())
                logits = outputs.logits
        else:
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids.long(), attention_mask=b_input_mask.long())
                logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
        torch.cuda.empty_cache()  # Clear cache after each batch
    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    return np.array(predictions), np.array(true_labels)


if __name__ == "__main__":
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Load the data
    df = pd.read_csv('./training/training_data.csv')
    df = drop_missing_remove_duplicates(df)

    # Split the data
    #train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the splits
    #train_df.to_csv('ensemble_model/train_data.csv', index=False)
    #test_df.to_csv('ensemble_model/test_data.csv', index=False)
    
    # Load the data
    train_df = pd.read_csv('ensemble_model/train_data.csv')
    test_df = pd.read_csv('ensemble_model/test_data.csv')

    # Augment the training data
    train_df_augmented = augment_df(train_df)
    top_tags = get_top_pos_tags(train_df_augmented)
    #save top_tags
    pos_tags_path = './ensemble_model/top_pos_tags.json'
    with open(pos_tags_path, 'w') as f:
        json.dump(top_tags, f)
    test_df_augmented = augment_df(test_df, top_tags)

    # Scaling
    scaler = StandardScaler()
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    scaler.fit(train_df_augmented[feature_columns].values)
    joblib.dump(scaler, 'ensemble_model/scaler.pkl')

    train_df_augmented[feature_columns] = scaler.transform(train_df_augmented[feature_columns].values)
    test_df_augmented[feature_columns] = scaler.transform(test_df_augmented[feature_columns].values)

    # Generate embeddings
    train_df_augmented, embedding_size = generate_embeddings(train_df_augmented, chosen_tokenizer='camembert')
    test_df_augmented, _ = generate_embeddings(test_df_augmented, chosen_tokenizer='camembert')

    # Prepare data loaders
    camembert_tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

    # Set hyperparameters
    best_hyperparameters_camembert = {
        'learning_rate': 5e-5,
        'batch_size': 40,
        'epochs': 20
    }
    best_hyperparameters_flaubert = {
        'learning_rate': 5e-5,
        'batch_size': 40,
        'epochs': 16
    }

    gradient_accumulation_steps_camembert = max(1, 64 // best_hyperparameters_camembert['batch_size'])
    gradient_accumulation_steps_flaubert = max(1, 64 // best_hyperparameters_flaubert['batch_size'])  

    camembert_dataloader = prepare_data(train_df_augmented, camembert_tokenizer, scaler, max_len=264, batch_size=best_hyperparameters_camembert['batch_size'], use_features=args.use_nn, device=device)
    flaubert_dataloader = prepare_data(train_df_augmented, flaubert_tokenizer, scaler, max_len=264, batch_size=best_hyperparameters_flaubert['batch_size'], use_features=args.use_nn, device=device)

    camembert_model_path = 'ensemble_model/camembert_full'
    flaubert_model_path = 'ensemble_model/flaubert_full'
    nn_model_path = 'ensemble_model/simple_nn.pth'
# Check if models already exist
    if os.path.exists(flaubert_model_path+'/config.json'):
        flaubert_model = FlaubertForSequenceClassification.from_pretrained(flaubert_model_path).to(device)
    else:
        print('Training Flaubert...')
        grad_scaler = torch.cuda.amp.GradScaler()
        flaubert_model = initialize_model(num_labels=6, device=device, model_choice='flaubert')
        flaubert_optimizer = get_optimizer(flaubert_model, learning_rate=best_hyperparameters_flaubert['learning_rate'])

        train_model(flaubert_model, flaubert_dataloader, flaubert_optimizer, device, grad_scaler, num_epochs=best_hyperparameters_flaubert['epochs'], use_features=args.use_nn, gradient_accumulation_steps=gradient_accumulation_steps_flaubert)

        flaubert_model.save_pretrained(flaubert_model_path)

    if os.path.exists(camembert_model_path+'/config.json') :
        camembert_model = CamembertForSequenceClassification.from_pretrained(camembert_model_path).to(device)
    else:
        # Initialize and train models
        print('Training Camembert...')
        grad_scaler = torch.cuda.amp.GradScaler()
        camembert_model = initialize_model(num_labels=6, device=device, model_choice='camembert')
        camembert_optimizer = get_optimizer(camembert_model, learning_rate=best_hyperparameters_camembert['learning_rate'])
        train_model(camembert_model, camembert_dataloader, camembert_optimizer, device, grad_scaler, num_epochs=best_hyperparameters_camembert['epochs'], use_features=args.use_nn, gradient_accumulation_steps=gradient_accumulation_steps_camembert)
        camembert_model.save_pretrained(camembert_model_path)

    
    if not os.path.exists(flaubert_model_path+'/tokenizer_config.json'):
        print('Saving tokenizer flaubert...')
        flaubert_tokenizer.save_pretrained(f'./ensemble_model/full/flaubert_full')

    if not os.path.exists(camembert_model_path+'/tokenizer_config.json'):
        print('Saving tokenizer camembert...')
        camembert_tokenizer.save_pretrained(f'./ensemble_model/full/camembert_full')
    
    # Train SimpleNN   
    nn_hyperparameters = {
            "learning_rate": 0.0001,
            "hidden_size": 64,
            "batch_size": 32,
            "epochs": 64
        }
    if args.use_nn:
        if os.path.exists(nn_model_path):
            nn_model = SimpleNN(input_size=7 + embedding_size, hidden_size=nn_hyperparameters['hidden_size'], num_classes=6).to(device)
            nn_model.load_state_dict(torch.load(nn_model_path))
        else:
            print('Training new nn...')

            nn_model = SimpleNN(input_size=7 + embedding_size, hidden_size=nn_hyperparameters['hidden_size'], num_classes=6).to(device)
            nn_optimizer = AdamW(nn_model.parameters(), lr=nn_hyperparameters['learning_rate'])
            nn_dataloader = prepare_nn_data_with_embeddings(train_df_augmented, scaler, batch_size=nn_hyperparameters['batch_size'], device=device)
            train_nn(nn_model, nn_dataloader, nn_optimizer, device, epochs=nn_hyperparameters['epochs'])

            # Save SimpleNN model
            torch.save(nn_model.state_dict(), nn_model_path)

    # Prepare data loader for evaluation on the entire test set
    camembert_eval_dataloader = prepare_data(test_df_augmented, camembert_tokenizer, scaler, max_len=264, batch_size=32, use_features=args.use_nn, shuffle=False, device=device)
    flaubert_eval_dataloader = prepare_data(test_df_augmented, flaubert_tokenizer, scaler, max_len=264, batch_size=32, use_features=args.use_nn, shuffle=False, device=device)
    
    # Load pre-trained models
    camembert_model = CamembertForSequenceClassification.from_pretrained(camembert_model_path).to(device)
    flaubert_model = FlaubertForSequenceClassification.from_pretrained(flaubert_model_path).to(device)
    
    if args.use_nn:
        nn_model = SimpleNN(input_size=7 + embedding_size, hidden_size = nn_hyperparameters['hidden_size'], num_classes=6).to(device)
        nn_model.load_state_dict(torch.load(nn_model_path))

    camembert_predictions, true_labels = evaluate_model(camembert_model, camembert_eval_dataloader, device, use_features=args.use_nn)
    flaubert_predictions, true_labels = evaluate_model(flaubert_model, flaubert_eval_dataloader, device, use_features=args.use_nn)
    camembert_accuracy = accuracy_score(true_labels, np.argmax(camembert_predictions, axis=1))
    flaubert_accuracy = accuracy_score(true_labels, np.argmax(flaubert_predictions, axis=1))
    print(f"Camembert Model Accuracy: {camembert_accuracy}")
    print(f"Flaubert Model Accuracy: {flaubert_accuracy}")

    if args.use_nn:
        nn_eval_dataloader = prepare_nn_data_with_embeddings(test_df_augmented, scaler, batch_size=32, shuffle=False, device=device)
        nn_predictions, true_labels = evaluate_nn(nn_model, nn_eval_dataloader, device)
        nn_predictions, true_labels = evaluate_nn(nn_model, nn_eval_dataloader, device)
        nn_accuracy = accuracy_score(true_labels, nn_predictions)
        print(f"SimpleNN Model Accuracy: {nn_accuracy}")
        train_features = np.hstack([
            np.argmax(camembert_predictions, axis=1).reshape(-1, 1),
            np.argmax(flaubert_predictions, axis=1).reshape(-1, 1),
            np.array(nn_predictions).reshape(-1, 1)
        ])
    else:
        train_features = np.hstack([
            np.argmax(camembert_predictions, axis=1).reshape(-1, 1),
            np.argmax(flaubert_predictions, axis=1).reshape(-1, 1)
        ])

    lgb_train = lgb.Dataset(train_features, label=true_labels)

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'n_estimators': [50, 100, 200]
    }

    lgb_estimator = lgb.LGBMClassifier(objective='multiclass', num_class=6, metric='multi_logloss', boosting_type='gbdt', verbose=-1)
    grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(train_features, true_labels)

    best_lgb_params = grid_search.best_params_
    print(f"Best LightGBM parameters found: {best_lgb_params}")

    # Train the final LightGBM model with the best parameters
    best_lgb_model = lgb.LGBMClassifier(**best_lgb_params, objective='multiclass', num_class=6, metric='multi_logloss', boosting_type='gbdt', verbose=-1)
    best_lgb_model.fit(train_features, true_labels)

    # Prepare data for MetaNN
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    true_labels_tensor = torch.tensor(true_labels, dtype=torch.long)
    meta_nn_dataloader = create_nn_dataloader(train_features_tensor, true_labels_tensor, batch_size=32)

    meta_nn_param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [32, 64, 128],
        'epochs': [20, 50, 100]
    }

    best_meta_nn_accuracy = 0
    best_meta_nn_params = {}
    best_meta_nn_model = None

    for lr in meta_nn_param_grid['learning_rate']:
        for hidden_size in meta_nn_param_grid['hidden_size']:
            for epochs in meta_nn_param_grid['epochs']:
                meta_nn_model = MetaNN(input_size=train_features.shape[1], hidden_size=hidden_size, num_classes=6).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = AdamW(meta_nn_model.parameters(), lr=lr)
                train_meta_nn(meta_nn_model, meta_nn_dataloader, criterion, optimizer, device, epochs)

                meta_nn_predictions, true_labels = evaluate_meta_nn(meta_nn_model, meta_nn_dataloader, device)
                meta_nn_accuracy = accuracy_score(true_labels, meta_nn_predictions)
                print(f"MetaNN Model Accuracy: {meta_nn_accuracy} with params lr={lr}, hidden_size={hidden_size}, epochs={epochs}")

                if meta_nn_accuracy > best_meta_nn_accuracy:
                    best_meta_nn_accuracy = meta_nn_accuracy
                    best_meta_nn_params = {'learning_rate': lr, 'hidden_size': hidden_size, 'epochs': epochs}
                    best_meta_nn_model = meta_nn_model

    print(f"Best MetaNN parameters found: {best_meta_nn_params}")

    # Compare the best models from LightGBM and MetaNN
    if best_meta_nn_accuracy > grid_search.best_score_:
        print(f"Selecting MetaNN as the final model with accuracy: {best_meta_nn_accuracy}")
        final_model = best_meta_nn_model
        model_suffix = '_with_features' if args.use_nn else ''
        torch.save(final_model.state_dict(), f'ensemble_model/meta_nn{model_suffix}.pth')
    else:
        print(f"Selecting LightGBM as the final model with accuracy: {grid_search.best_score_}")
        final_model = best_lgb_model
        model_suffix = '_with_features' if args.use_nn else ''
        best_lgb_model.booster_.save_model(f'ensemble_model/meta_classifier{model_suffix}.txt')