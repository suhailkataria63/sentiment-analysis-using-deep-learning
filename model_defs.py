from tensorflow.keras import layers, models

def build_gru_model(vocab_size: int, max_len: int = 100, embed_dim: int = 128):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),
        layers.Bidirectional(layers.GRU(128)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model