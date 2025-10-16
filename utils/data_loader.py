#utils/data_loader.py
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def get_dict_map(data, token_or_tag):
    """Generate token/tag to index and index to token/tag dictionaries."""
    vocab = list(set(data['Word'].to_list())) if token_or_tag == 'token' else list(set(data['Tag'].to_list()))
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

def prepare_data(path='data/ner_dataset.csv'):
    """Load dataset and prepare grouped sequences."""
    data = pd.read_csv(path, encoding='unicode_escape')
    data['Sentence #'] = data['Sentence #'].fillna(method='ffill')

    token2idx, idx2token = get_dict_map(data, 'token')
    tag2idx, idx2tag = get_dict_map(data, 'tag')

    data['Word_idx'] = data['Word'].map(token2idx)
    data['Tag_idx'] = data['Tag'].map(tag2idx)
    data_fillna = data.fillna(method='ffill', axis=0)

    data_group = data_fillna.groupby('Sentence #', as_index=False)[
        ['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx']
    ].agg(lambda x: list(x))

    return data, data_group, token2idx, tag2idx

def get_pad_train_test_val(data_group, data, tag2idx):
    """Pad and split data into train/val/test sets."""
    n_token = len(list(set(data['Word'].to_list())))
    n_tag = len(list(set(data['Tag'].to_list())))

    tokens = data_group['Word_idx'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value=n_token - 1)

    tags = data_group['Tag_idx'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value=tag2idx["O"])
    pad_tags = [to_categorical(i, num_classes=n_tag) for i in pad_tags]

    tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, random_state=42)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, tags_, test_size=0.25, random_state=42)

    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags, maxlen, n_token, n_tag
