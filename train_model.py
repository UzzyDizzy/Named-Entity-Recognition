#train_model.py
import numpy as np
from utils.data_loader import prepare_data, get_pad_train_test_val
from utils.model_builder import get_bilstm_lstm_model
import os

def train_model():
    data, data_group, token2idx, tag2idx = prepare_data()
    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags, maxlen, n_token, n_tag = get_pad_train_test_val(data_group, data, tag2idx)

    input_dim = n_token + 1
    output_dim = 64
    input_length = maxlen

    model = get_bilstm_lstm_model(input_dim, output_dim, input_length, n_tag)

    print("Training model... this may take some time ⏳")
    history = model.fit(
        train_tokens,
        np.array(train_tags),
        validation_data=(val_tokens, np.array(val_tags)),
        batch_size=512,
        epochs=5,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/ner_model.h5")
    print("✅ Model saved to models/ner_model.h5")

if __name__ == "__main__":
    train_model()
