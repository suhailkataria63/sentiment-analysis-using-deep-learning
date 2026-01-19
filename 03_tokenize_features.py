from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
import yaml
import os


def main():
    spark = (
        SparkSession.builder
        .appName("TokenizeStopwordsTFIDF")
        .getOrCreate()
    )

    with open("src/common/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parquet_dir = cfg["paths"]["processed"]["parquet"]
    in_path = os.path.join(parquet_dir, "sentiment_clean.parquet")
    out_path = os.path.join(parquet_dir, "sentiment_tfidf.parquet")

    df = spark.read.parquet(in_path).select("text", "sentiment", "text_len")

    # Tokenize: split on non-word characters, keep tokens length >= 2
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\W+",
        minTokenLength=2
    )

    # Stopword removal
    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="tokens_nostop"
    )

    # TF (hashing trick): scalable sparse vector
    hashing_tf = HashingTF(
        inputCol="tokens_nostop",
        outputCol="tf",
        numFeatures=1 << 18  # 262,144 dims
    )

    # IDF: downweight common terms, upweight rare terms
    idf = IDF(
        inputCol="tf",
        outputCol="tfidf"
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
    model = pipeline.fit(df)
    out = model.transform(df)

    # Keep only what we need downstream
    final_df = out.select(
        "sentiment",
        "text_len",
        "tokens_nostop",
        "tfidf"
    )

    print("\n[+] Tokenized sample:")
    final_df.select("sentiment", "tokens_nostop").show(5, truncate=80)

    final_df.write.mode("overwrite").parquet(out_path)
    print(f"\n[✓] Wrote TF-IDF Parquet to: {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()
