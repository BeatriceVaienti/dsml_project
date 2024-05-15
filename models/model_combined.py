# model_combined.py

from transformers import CamembertModel, CamembertForSequenceClassification
import torch.nn as nn
import torch

class CamembertWithFeatures(nn.Module):
    def __init__(self, num_labels, feature_dim, model_path):
        super(CamembertWithFeatures, self).__init__()
        self.camembert = CamembertModel.from_pretrained(model_path)
        self.classifier = CamembertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.feature_layer = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.final_classifier = nn.Linear(self.camembert.config.hidden_size + feature_dim, num_labels)
        
    def forward(self, input_ids, attention_mask, features):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the last hidden state of the first token (CLS token)
        
        feature_output = self.feature_layer(features)
        combined_output = torch.cat((pooled_output, feature_output), dim=1)
        combined_output = self.dropout(combined_output)
        logits = self.final_classifier(combined_output)
        return logits
