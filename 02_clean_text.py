from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import yaml
import os


def main():
    spark = (
        SparkSession.builder
        .appName("CleanTextSentiment")
        .getOrCreate()
    )

    with open("src/common/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parquet_dir = cfg["paths"]["processed"]["parquet"]

    in_path = os.path.join(parquet_dir, "sentiment_raw.parquet")
    out_path = os.path.join(parquet_dir, "sentiment_clean.parquet")

    df = spark.read.parquet(in_path)

    # ---- 1) Normalize column names to standard schema ----
    # Your data has: sentence, sentiment
    # We standardize to: text, sentiment
    df = df.select(
        F.col("sentence").alias("text"),
        F.col("sentiment").alias("sentiment")
    )

    # ---- 2) Clean text ----
    # Lowercase
    df = df.withColumn("text", F.lower(F.col("text")))

    # Remove URLs
    df = df.withColumn("text", F.regexp_replace(F.col("text"), r"http\S+|www\.\S+", " "))

    # Remove @mentions
    df = df.withColumn("text", F.regexp_replace(F.col("text"), r"@\w+", " "))

    # Remove hashtags symbol (#) but keep the word
    df = df.withColumn("text", F.regexp_replace(F.col("text"), r"#", ""))

    # Remove non-alphanumeric chars except spaces (keeps words/numbers)
    df = df.withColumn("text", F.regexp_replace(F.col("text"), r"[^a-z0-9\s]", " "))

    # Collapse multiple spaces
    df = df.withColumn("text", F.regexp_replace(F.col("text"), r"\s+", " "))

    # Trim
    df = df.withColumn("text", F.trim(F.col("text")))

    # ---- 3) Clean labels ----
    # Ensure integer (some CSVs store as string)
    df = df.withColumn("sentiment", F.col("sentiment").cast("int"))

    # Keep only valid labels (0 or 1)
    df = df.filter(F.col("sentiment").isin([0, 1]))

    # Drop empty text rows
    df = df.filter(F.length(F.col("text")) > 0)

    # ---- 4) Add simple analytics helper columns ----
    df = df.withColumn("text_len", F.length(F.col("text")))
    df = df.withColumn("ingested_at", F.current_timestamp())

    print("\n[+] Clean dataset schema:")
    df.printSchema()

    print("\n[+] Class distribution:")
    df.groupBy("sentiment").count().orderBy("sentiment").show()

    print("\n[+] Sample rows:")
    df.show(5, truncate=80)

    df.write.mode("overwrite").parquet(out_path)
    print(f"\n[✓] Wrote cleaned Parquet to: {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()
