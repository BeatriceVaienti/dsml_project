import pandas as pd
import numpy as np
from collections import defaultdict
import spacy

# Load French language model in spaCy
nlp = spacy.load("fr_core_news_sm")

def tokenize_and_extract_features(text):
    doc = nlp(text)
    tokens = []
    pos_counts = defaultdict(int)
    for token in doc:
        if not token.is_stop and not token.is_space:
            tokens.append(token)  # Store the full token object
            pos_counts[token.pos_] += 1
    return tokens, dict(pos_counts)

def add_tokens_and_pos_counts(df):
    df = df.copy()  # Avoid modifying the input DataFrame
    df[['tokens', 'pos_counts']] = df['sentence'].apply(lambda x: pd.Series(tokenize_and_extract_features(x)))
    return df

def add_n_words(df):
    df = df.copy()  # Avoid modifying the input DataFrame
    df['n_words'] = df['tokens'].apply(len)
    return df

def add_avg_word_length_no_stopwords(df):
    df = df.copy()  # Avoid modifying the input DataFrame
    df['avg_word_length'] = df['tokens'].apply(lambda tokens: np.mean([len(token.text) for token in tokens]) if tokens else 0)
    return df

def calculate_pos_frequencies(pos_counts, top_tags):
    total = sum(pos_counts.get(tag, 0) for tag in top_tags)  # Use get for safe access
    return {tag: (pos_counts.get(tag, 0) / total if total > 0 else 0) for tag in top_tags}

def add_pos_frequencies(df, top_tags):
    df = df.copy()  # Avoid modifying the input DataFrame
    df['pos_frequencies'] = df['pos_counts'].apply(lambda x: calculate_pos_frequencies(x, top_tags))
    for tag, i in zip(top_tags, range(len(top_tags))):
        df['tag_' + str(i)] = df['pos_frequencies'].apply(lambda x: x.get(tag, 0))
    return df

def get_top_pos_tags(df, top_n=5):
    tokens_lists = df['tokens']
    total_tag_counts = defaultdict(int)
    for tokens in tokens_lists:
        for token in tokens:
            if token.pos_ not in {'SPACE'}:
                total_tag_counts[token.pos_] += 1
    sorted_tags = sorted(total_tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_tags = [tag for tag, _ in sorted_tags]
    return top_tags

def augment_df(df, top_tags=None):
    df = df.copy()
    print('Adding tokens and pos counts')
    df = add_tokens_and_pos_counts(df)
    print('Adding n words')
    df = add_n_words(df)
    print('Adding avg word length')
    df = add_avg_word_length_no_stopwords(df)
    
    if top_tags is None:
        print('Calculating top POS tags from training dataset')
        top_tags = get_top_pos_tags(df)
    
    print('Adding pos frequencies')
    df = add_pos_frequencies(df, top_tags)
    return df


