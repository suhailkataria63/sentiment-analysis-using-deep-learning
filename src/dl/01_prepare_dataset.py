# src/dl/01_prepare_dataset.py
# Prepare cleaned Spark data for deep learning models

import os
import pandas as pd

# ---------------------------
# Paths
# ---------------------------
INPUT_PARQUET = "data/processed/parquet/sentiment_clean.parquet"
OUTPUT_DIR = "data/processed/features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sentiment_dl_dataset.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_parquet(INPUT_PARQUET)

# ---------------------------
# Select + rename columns
# ---------------------------
df = df[["text", "sentiment"]].dropna()

df = df.rename(columns={
    "text": "text_clean",
    "sentiment": "sentiment_label"
})

# Ensure label is integer
df["sentiment_label"] = df["sentiment_label"].astype(int)

# ---------------------------
# Save for DL pipeline
# ---------------------------
df.to_csv(OUTPUT_FILE, index=False)

print(f"[✓] DL dataset written to: {OUTPUT_FILE}")
print(df.head(3))
