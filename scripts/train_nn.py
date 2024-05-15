import sys
sys.path.append('./')

from utils.data_augmentation import augment_df
from utils.embeddings_generation import generate_embeddings
from utils.label_encoding import get_encoded_y
from utils.data_processing import drop_missing_remove_duplicates
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # data preparation and augmentation

    # Load the dataset
    df = pd.read_csv('./training/training_data.csv')

    df = drop_missing_remove_duplicates(df)

    # Augment the dataset with new features
    print("Augmenting the dataset with new features...")
    df_augmented = augment_df(df)

    # Creating the embeddings
    print("Creating embeddings for the augmented dataset...")
    df_augmented = generate_embeddings(df_augmented, chosen_tokenizer='camembert', batch_size=32)
    # Save the augmented dataset
    
    df_augmented.to_csv("training/training_data_augmented.csv", index=False)
    print("Augmented dataset with embeddings saved to training/training_data_augmented.csv")

    # Encoding the target variable
    y_encoded = get_encoded_y(df)

    #We select the features that we want to use
    selected_features = ['n_words', 'avg_word_length', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4']
    selected_features_columns = df_augmented[selected_features]
    # We isolate the embeddings
    embeddings_array = np.vstack(df['embeddings'].values)

    # Scaling of the selected features
    scaler = StandardScaler()
    selected_features_columns = scaler.fit_transform(selected_features_columns)
    
