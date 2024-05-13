import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def get_encoded_y(df):
    y = df['difficulty'].values
    labels_ordered = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    encoder = LabelEncoder()
    encoder.fit(labels_ordered)
    y_encoded = encoder.transform(y)
    return y_encoded

def get_label_encoder(labels_ordered=['A1', 'A2', 'B1', 'B2', 'C1', 'C2']):
    try:
        # Try to load an existing LabelEncoder from a file
        with open('models_saved/label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    except FileNotFoundError:
        # If not previously saved, initialize, fit, and save the LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(labels_ordered)
        with open('models_saved/label_encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
    return encoder