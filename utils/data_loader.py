""" 
Contains the functions for data preparation and data loading 
Including tokenization, padding, creating attention masks and dataloaders
"""
import pandas as pd
import torch
from utils.label_encoding import get_encoded_y
from utils.data_processing import drop_missing_remove_duplicates
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



def load_and_prepare_data(df, tokenizer, max_len):
    """
    The function removes missing values and duplicates from the dataset,
    tokenizes the text, pads it, and creates attention masks.
    """
    df = df.copy()
    df = drop_missing_remove_duplicates(df)
    text = df['sentence'].tolist()  # Use .tolist() to convert directly to list
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

    # Extract the inputs and attention masks from the dictionary
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    return train_test_split(input_ids, labels, attention_masks, random_state=42, test_size=0.2)

def create_dataloaders(train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks, batch_size):
    train_inputs, validation_inputs = map(lambda x: x.clone().detach(), (train_inputs, validation_inputs))
    train_labels, validation_labels = map(torch.tensor, (train_labels, validation_labels))
    train_masks, validation_masks = map(lambda x: x.clone().detach(), (train_masks, validation_masks))

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)

    return train_dataloader, validation_dataloader