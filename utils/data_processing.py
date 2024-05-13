
import pandas as pd
import numpy as np

def drop_missing_remove_duplicates(df):
    # Drop rows with missing 'sentence' or 'difficulty' in training data
    df = df.dropna(subset=['sentence', 'difficulty'])
    # remove duplicates
    df = df.drop_duplicates(subset=['sentence'])
    df.reset_index(drop=True, inplace=True)
    return df