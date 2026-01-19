import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_prepare(
    csv_path: str,
    max_vocab: int = 20000,
    max_len: int = 50,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    df = pd.read_csv(csv_path).dropna()
    texts = df["text_clean"].astype(str).tolist()
    labels = df["sentiment_label"].astype(int).values

    # Train/Val/Test split (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=(test_size + val_size),
        random_state=random_state, stratify=labels
    )

    rel_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - rel_val_size),
        random_state=random_state, stratify=y_tmp
    )

    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def to_padded(x):
        seq = tokenizer.texts_to_sequences(x)
        return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    X_train_pad = to_padded(X_train)
    X_val_pad   = to_padded(X_val)
    X_test_pad  = to_padded(X_test)

    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab)

    return (X_train_pad, y_train, X_val_pad, y_val, X_test_pad, y_test, tokenizer, vocab_size)
