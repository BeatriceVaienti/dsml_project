import torch
from transformers import CamembertForSequenceClassification, CamembertConfig, FlaubertForSequenceClassification, FlaubertConfig
from torch.optim import AdamW

def initialize_model(num_labels, device, model_choice):
    """
    Initialize the model based on the user's choice.
    """
    if model_choice == 'camembert':
        config = CamembertConfig.from_pretrained('camembert-base', num_labels=num_labels)
        model = CamembertForSequenceClassification(config)
    elif model_choice == 'flaubert':
        config = FlaubertConfig.from_pretrained('flaubert/flaubert_base_cased', num_labels=num_labels)
        model = FlaubertForSequenceClassification(config)
    else:
        raise ValueError("Invalid model choice. Choose 'camembert' or 'flaubert'.")
    
    model.to(device)
    return model

def get_optimizer(model, learning_rate):
    """
    Get the optimizer for the model. The same optimizer setup can be used for both models.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
