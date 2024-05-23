import os
import sys
import torch
sys.path.append('./')
import pandas as pd
import numpy as np
import joblib
import json

from transformers import CamembertTokenizer, FlaubertTokenizer, CamembertForSequenceClassification, FlaubertForSequenceClassification
from utils.data_processing import drop_missing_remove_duplicates
from utils.data_augmentation import augment_df, get_top_pos_tags
from utils.label_encoding import get_encoded_y
from utils.embeddings_generation import generate_embeddings
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from models.model_nn import SimpleNN
from models.model_meta_nn import MetaNN, create_nn_dataloader

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def prepare_data(df, tokenizer, scaler, max_len, batch_size=32, use_features=False, shuffle=False, device='cpu'):
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

        if features.ndim == 1:
            features = features.unsqueeze(1)

        data = TensorDataset(input_ids, attention_masks, features)
    else:
        data = TensorDataset(input_ids, attention_masks)

    sampler = SequentialSampler(data) if not shuffle else RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    return dataloader

def prepare_nn_data_with_embeddings(df, scaler, batch_size=32, shuffle=False, device='cpu'):
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    features = df[feature_columns].values
    features = scaler.transform(features)

    embeddings = np.vstack(df['embeddings'].values)

    combined_features = np.hstack((features, embeddings))

    combined_features = torch.tensor(combined_features, dtype=torch.float32).to(device)

    data = TensorDataset(combined_features)
    sampler = SequentialSampler(data) if not shuffle else RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    return dataloader

def load_model(model_path, model_class):
    if model_class == 'camembert':
        model = CamembertForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_class == 'flaubert':
        model = FlaubertForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_class == 'nn':
        model = SimpleNN(input_size=7 + embedding_size, hidden_size=nn_hyperparameters['hidden_size'], num_classes=6).to(device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

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
        for features in dataloader:
            features = features[0].to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            predictions.extend(probs.cpu().numpy())
            torch.cuda.empty_cache()  # Clear cache after each batch
    return np.array(predictions)

def load_meta_model(meta_model_path, meta_model_type, input_size, best_meta_nn_params=None):
    if meta_model_type == 'lgb':
        import lightgbm as lgb
        meta_model = lgb.Booster(model_file=meta_model_path)
    elif meta_model_type == 'nn':
        meta_model = MetaNN(input_size=input_size, hidden_size=best_meta_nn_params['hidden_size'], num_classes=6).to(device)
        if torch.cuda.is_available():
            meta_model.load_state_dict(torch.load(meta_model_path))
        else:
            meta_model.load_state_dict(torch.load(meta_model_path, map_location=torch.device('cpu')))
    return meta_model

if __name__ == "__main__":
    # Load unlabelled data
    unlabelled_df = pd.read_csv('./test/unlabelled_test_data.csv')

    # Load saved models
    camembert_model_path = './ensemble_model/camembert'
    flaubert_model_path = './ensemble_model/flaubert'
    nn_model_path = './ensemble_model/simple_nn/simple_nn.pth'

    camembert_tokenizer = CamembertTokenizer.from_pretrained(camembert_model_path)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(flaubert_model_path)

    # Load scaler
    scaler = joblib.load('./ensemble_model/simple_nn/scaler.pkl')

    # Augment the data and generate embeddings
    pos_tags_path = './ensemble_model/simple_nn/top_pos_tags.json'
    with open(pos_tags_path, 'r') as f:
        top_tags = json.load(f)
    unlabelled_df = augment_df(unlabelled_df, top_tags)
    unlabelled_df, embedding_size = generate_embeddings(unlabelled_df, chosen_tokenizer='camembert')

    # Prepare dataloaders
    camembert_dataloader = prepare_data(unlabelled_df, camembert_tokenizer, scaler, max_len=264, batch_size=32, use_features=True, shuffle=False, device=device)
    flaubert_dataloader = prepare_data(unlabelled_df, flaubert_tokenizer, scaler, max_len=264, batch_size=32, use_features=True, shuffle=False, device=device)
    nn_dataloader = prepare_nn_data_with_embeddings(unlabelled_df, scaler, batch_size=32, shuffle=False, device=device)

    # Evaluate models
    camembert_model = load_model(camembert_model_path, 'camembert')
    flaubert_model = load_model(flaubert_model_path, 'flaubert')
    nn_model = load_model(nn_model_path, 'nn')

    print('Evaluating Camembert model...')
    camembert_predictions = evaluate_model(camembert_model, camembert_dataloader, device, use_features=True)
    print('Evaluating Flaubert model...')
    flaubert_predictions = evaluate_model(flaubert_model, flaubert_dataloader, device, use_features=True)
    print('Evaluating SimpleNN model...')
    nn_predictions = evaluate_nn(nn_model, nn_dataloader, device)

    # Prepare features for meta classifier
    test_features = np.hstack([
        np.argmax(camembert_predictions, axis=1).reshape(-1, 1),
        np.argmax(flaubert_predictions, axis=1).reshape(-1, 1),
        np.argmax(nn_predictions, axis=1).reshape(-1, 1)
    ])

    # Load meta model parameters and hyperparameters
    meta_model_config_path = './ensemble_model/meta_model_config.json'
    with open(meta_model_config_path, 'r') as f:
        meta_model_config = json.load(f)
    meta_model_type = meta_model_config['type']
    best_meta_nn_params = meta_model_config.get('best_meta_nn_params')
    input_size = test_features.shape[1]

    meta_model_path = meta_model_config['path']
    meta_model = load_meta_model(meta_model_path, meta_model_type, input_size, best_meta_nn_params)

    if meta_model_type == 'nn':
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
        meta_nn_dataloader = create_nn_dataloader(test_features_tensor, batch_size=32)
        meta_predictions = evaluate_nn(meta_model, meta_nn_dataloader, device)
    else:
        meta_predictions = meta_model.predict(test_features)

    predicted_labels = np.argmax(meta_predictions, axis=1)

    # Save predictions
    unlabelled_df['predicted_label'] = predicted_labels
    unlabelled_df.to_csv('./test/unlabelled_test_data_predictions.csv', index=False)
    print('Predictions saved to ./test/unlabelled_test_data_predictions.csv')