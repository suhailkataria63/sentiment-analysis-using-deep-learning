from pyspark.sql import SparkSession
import yaml
import os


def main():
    spark = (
        SparkSession.builder
        .appName("IngestToParquet")
        .getOrCreate()
    )

    # Load config
    with open("src/common/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    input_dir = cfg["paths"]["raw_data"]["sentiment"]          # e.g., data/raw/sentiment140
    output_dir = cfg["paths"]["processed"]["parquet"]          # e.g., data/processed/parquet

    # Your renamed file should be here:
    # data/raw/sentiment140/sentiment140.csv
    input_file = os.path.join(input_dir, "sentiment140.csv")

    # Decide header based on the first line of the CSV
    # (Kaggle variants often have headers like: text,label)
    # If you’re unsure, we’ll print the first line in terminal below.
    has_header = True

    df = (
        spark.read
        .option("header", "true" if has_header else "false")
        .option("inferSchema", "true")
        .csv(input_file)
    )

    # If NO header, rename columns manually
    if not has_header:
        # Common no-header format: sentiment,label,text or sentiment,text
        # Adjust if needed after checking df.columns
        df = df.toDF("sentiment", "text")

    print("\n[+] Schema:")
    df.printSchema()

    print("\n[+] Sample rows:")
    df.show(5, truncate=80)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sentiment_raw.parquet")

    df.write.mode("overwrite").parquet(out_path)
    print(f"\n[✓] Wrote Parquet to: {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()
