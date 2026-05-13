"""
Train sentiment model (BiGRU) on Sentiment140 prepared dataset.

Inputs:
- data/processed/features/sentiment_dl_dataset.csv
  columns: text_clean, sentiment_label  (0/1)

Outputs:
- models/sentiment/gru_sentiment_model.keras
- models/sentiment/tokenizer.pkl
- data/processed/predictions/dl_test_predictions.csv
- data/processed/predictions/dl_test_y_prob.npy
- data/processed/predictions/dl_test_y_true.npy
- data/processed/predictions/dl_history.csv
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dl.dataset import load_and_prepare
from src.dl.model_defs import build_gru_model


# -----------------------
# Config (safe defaults)
# -----------------------
DATASET_CSV = "data/processed/features/sentiment_dl_dataset.csv"

MODEL_DIR = "models/sentiment"
PRED_DIR = "data/processed/predictions"

MODEL_PATH = os.path.join(MODEL_DIR, "gru_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

PRED_CSV_PATH = os.path.join(PRED_DIR, "dl_test_predictions.csv")
YPROB_NPY_PATH = os.path.join(PRED_DIR, "dl_test_y_prob.npy")
YTRUE_NPY_PATH = os.path.join(PRED_DIR, "dl_test_y_true.npy")
HISTORY_CSV_PATH = os.path.join(PRED_DIR, "dl_history.csv")

# Training hyperparams (you can tune later)
MAX_VOCAB = 20000   # keep as-is unless you change it intentionally
MAX_LEN = 50        # keep as-is unless you change it intentionally
EMBED_DIM = 128
EPOCHS = 3
BATCH_SIZE = 128


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    # 1) Load + tokenize + pad + split
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, vocab_size = load_and_prepare(
        DATASET_CSV,
        max_vocab=MAX_VOCAB,
        max_len=MAX_LEN
    )

    print(f"[i] vocab_size={vocab_size}  max_len={MAX_LEN}")
    print(f"[i] train/val/test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # 2) Build model
    model = build_gru_model(vocab_size=vocab_size, max_len=MAX_LEN, embed_dim=EMBED_DIM)
    model.summary()

    # 3) Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save training history for plots later (R / Shiny)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"[✓] Saved training history to: {HISTORY_CSV_PATH}")

    # 4) Predict on test set
    y_prob = model.predict(X_test, batch_size=256).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    # 5) Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    print(f"[✓] Test metrics: {metrics}")

    # 6) Save model + tokenizer
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[✓] Saved model to: {MODEL_PATH}")
    print(f"[✓] Saved tokenizer to: {TOKENIZER_PATH}")

    # 7) Save predictions (CSV + NPY for debugging)
    np.save(YPROB_NPY_PATH, y_prob)
    np.save(YTRUE_NPY_PATH, y_test)

    pred_df = pd.DataFrame({
        "y_true": y_test.astype(int),
        "y_prob": y_prob,
        "y_pred": y_pred
    })
    pred_df.to_csv(PRED_CSV_PATH, index=False)

    print(f"[✓] Saved predictions CSV to: {PRED_CSV_PATH}")
    print(f"[✓] Saved y_prob/y_true to: {YPROB_NPY_PATH}, {YTRUE_NPY_PATH}")


if __name__ == "__main__":
    main()
