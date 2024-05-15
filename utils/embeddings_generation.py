import pandas as pd
import numpy as np
import torch
import nltk
from sklearn.preprocessing import LabelEncoder
from transformers import CamembertTokenizer, CamembertForSequenceClassification, FlaubertTokenizer, FlaubertForSequenceClassification, CamembertModel, FlaubertModel
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import tqdm


def get_tokenizer_model(chosen_tokenizer, num_classes):
    if chosen_tokenizer == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        model = CamembertModel.from_pretrained('camembert-base')
        classifier = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=num_classes)
    elif chosen_tokenizer == 'flaubert':
        tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
        model = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased')
        classifier = FlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased', num_labels=num_classes)
    return tokenizer, model, classifier

def batched_embeddings(texts, model, tokenizer, device, batch_size=32):
    all_embeddings = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
        with torch.no_grad():
            output = model(**encoded_input)
        embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def generate_embeddings(df, chosen_tokenizer = 'camembert', batch_size=32):
    # Get the embeddings for each sentence
    df = df.copy()
    num_classes = df['difficulty'].nunique()
    tokenizer, model, _ = get_tokenizer_model(chosen_tokenizer, num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = batched_embeddings(df['sentence'].tolist(), model, tokenizer, device, batch_size=batch_size)
    # Create a DataFrame with the embeddings
    # Concatenate the embeddings with the original DataFrame
    df['embeddings'] = embeddings.tolist()
    return df