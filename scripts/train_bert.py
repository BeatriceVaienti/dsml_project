import sys
sys.path.append('./')
import pandas as pd
import torch
from utils.data_processing import drop_missing_remove_duplicates
from utils.label_encoding import get_encoded_y
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import CamembertTokenizer, FlaubertTokenizer
from models.model_bert import initialize_model, get_optimizer
import argparse
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a text classifier model on French sentences.')
    parser.add_argument('--model', type=str, choices=['camembert', 'camembert-large', 'flaubert'], required=True, help='Choose the model to use: camembert, camembert-large, or flaubert')
    return parser.parse_args()

def prepare_data(df, tokenizer, max_length=264):
    df = df.copy()
    df = drop_missing_remove_duplicates(df)

    text = df['sentence'].to_list()
    labels = get_encoded_y(df).tolist()

    encoded_dict = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
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

def create_dataloader(inputs, masks, labels, batch_size):
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=True)
    return dataloader

def train(model, dataloader, optimizer, device, scaler, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    for step, batch in tqdm(enumerate(dataloader), desc="Training", total=len(dataloader)):
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
    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss

def main():
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

    # Set hyperparameters based on the best results from evaluation
    best_hyperparameters = {
        'learning_rate': 5e-05,
        'batch_size': 40,
        'epochs': 16
    }

    inputs, masks, labels = prepare_data(df, tokenizer)
    dataloader = create_dataloader(inputs, masks, labels, best_hyperparameters['batch_size'])

    model = initialize_model(6, device, args.model)
    optimizer = get_optimizer(model, best_hyperparameters['learning_rate'])
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    gradient_accumulation_steps = max(1, 64 // best_hyperparameters['batch_size'])  
    # Train the model
    for epoch in range(best_hyperparameters['epochs']):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, dataloader, optimizer, device, scaler, gradient_accumulation_steps=gradient_accumulation_steps)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

    # Save the model
    model.save_pretrained(f'./models_saved/{args.model}_full')
    tokenizer.save_pretrained(f'./models_saved/{args.model}_full')

if __name__ == "__main__":
    main()