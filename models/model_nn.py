# models/model_nn.py
import sys
sys.path.append('./')
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from utils.label_encoding import get_encoded_y
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_nn(model, dataloader, optimizer, device, epochs=64):
    model.train()
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
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(dataloader)}")


def evaluate_nn(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions), precision_recall_fscore_support(true_labels, predictions, average='macro'), confusion_matrix(true_labels, predictions)

def prepare_nn_data(df, scaler, batch_size=32, shuffle=True, device='cpu'):
    labels = get_encoded_y(df).tolist()
    feature_columns = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    features = df[feature_columns].values
    features = scaler.transform(features)
    
    #embeddings_camembert = np.vstack(df['embeddings_camembert'].values)
    embeddings_flaubert= np.vstack(df['embeddings_flaubert'].values)
    # Combine features and embeddings
    combined_features = np.hstack((features, embeddings_flaubert)) #add back flaubert here
    
    combined_features = torch.tensor(combined_features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels).to(device)

    data = TensorDataset(combined_features, labels)
    sampler = SequentialSampler(data) if not shuffle else RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, pin_memory=False)
    return dataloader
