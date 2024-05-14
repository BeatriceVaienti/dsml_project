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
    df = df.copy()
    df = drop_missing_remove_duplicates(df)
    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()
    
    # Use tokenizer's batch_encode_plus to handle tokenization, padding, and attention mask creation
    encoded_dict = tokenizer.batch_encode_plus(
        text,  # Batch of text to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad & truncate all sentences.
        padding='max_length',  # Pad all to max_length.
        truncation=True,  # Explicitly truncate to max_length.
        return_attention_mask=True,  # Include attention masks.
        return_tensors='pt'  # Return pytorch tensors.
    )
    input_ids = encoded_dict['input_ids']  
    attention_masks = encoded_dict['attention_mask']
    # Convert to tensors

    # se funziona quello prima elimina questo:
    inputs = input_ids.clone().detach()
    masks = attention_masks.clone().detach()
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
    batch_size = 8
    lr = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_choice == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=False)
    else:
        tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased', do_lower_case=False)
    
    max_len = 264
    df = pd.read_csv('./training/training_data.csv')
    
    full_dataloader = prepare_full_data(df, tokenizer, max_len, batch_size)
    
    model = initialize_model(num_labels=6, device=device, model_choice=model_choice)
    optimizer = get_optimizer(model, lr)
    
    epochs = 16
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, full_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

    # Save your model based on the selected model
    model.save_pretrained(f'./models_saved/{model_choice}_full')
    tokenizer.save_pretrained(f'./models_saved/{model_choice}_full')