import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "models/sentiment/gru_sentiment_model.keras"
TOKENIZER_PATH = "models/sentiment/tokenizer.pkl"

_model = None
_tokenizer = None

def load_assets():
    global _model, _tokenizer
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
    return _model, _tokenizer

def confidence_band(prob_positive: float) -> str:
    """
    Heuristic confidence bands based on distance from 0.5 decision boundary.
    HIGH:   p >= 0.80 or p <= 0.20
    MEDIUM: 0.65-0.80 or 0.20-0.35
    LOW:    0.35-0.65
    """
    p = prob_positive
    if p >= 0.80 or p <= 0.20:
        return "HIGH"
    if (0.65 <= p < 0.80) or (0.20 < p <= 0.35):
        return "MEDIUM"
    return "LOW"

def predict(text: str, max_len: int = 50, threshold: float = 0.5):
    model, tokenizer = load_assets()
    seq = tokenizer.texts_to_sequences([text])
    X = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    prob = float(model.predict(X, verbose=0)[0][0])
    label = 1 if prob >= threshold else 0
    sentiment = "POSITIVE" if label == 1 else "NEGATIVE"

    return {
        "prob_positive": prob,
        "label": label,
        "sentiment": sentiment,
        "confidence": confidence_band(prob)
    }

def predict_batch(texts, max_len: int = 50, threshold: float = 0.5):
    """
    Predict many texts in one model call.
    Returns list of dicts aligned with input order.
    """
    model, tokenizer = load_assets()

    # Tokenize
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    # Predict probs
    probs = model.predict(X, verbose=0).reshape(-1)

    results = []
    for text, p in zip(texts, probs):
        p = float(p)
        label = 1 if p >= threshold else 0
        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
        results.append({
            "text": text,
            "prob_positive": p,
            "label": label,
            "sentiment": sentiment,
            "confidence": confidence_band(p)
        })
    return results
