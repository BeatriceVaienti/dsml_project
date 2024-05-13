import sys
sys.path.append('./')
import pandas as pd
import torch
from utils.data_processing import drop_missing_remove_duplicates
from utils.label_encoding import get_encoded_y
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification, FlaubertTokenizer, FlaubertForSequenceClassification
from torch.optim import AdamW
from models.model_bert import initialize_model, get_optimizer
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a text classifier model on French sentences.')
    parser.add_argument('--model', type=str, choices=['camembert', 'flaubert'], required=True, help='Choose the model to use: camembert or flaubert')
    return parser.parse_args()

def prepare_full_data(df, tokenizer, max_len, batch_size=32):
    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()
    input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=max_len, truncation=True) for sent in text]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

    # Convert to tensors
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    data = TensorDataset(inputs, masks, labels)
    dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)
    return dataloader

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss


if __name__ == "__main__":
    args = get_arguments()
    model_choice = args.model
    batch_size = 16
    lr = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_choice == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    else:
        tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', do_lower_case=True)
    
    max_len = 128
    df = pd.read_csv('./training/training_data.csv')
    df = drop_missing_remove_duplicates(df)
    full_dataloader = prepare_full_data(df, tokenizer, max_len, batch_size)
    
    model = initialize_model(num_labels=6, device=device, model_choice=model_choice)
    optimizer = get_optimizer(model, lr)
    
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, full_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

    # Save your model based on the selected model
    model.save_pretrained(f'./models_saved/{model_choice}_full')
    tokenizer.save_pretrained(f'./models_saved/{model_choice}_full')