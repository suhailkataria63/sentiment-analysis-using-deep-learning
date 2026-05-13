from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import yaml
import os


def main():
    spark = (
        SparkSession.builder
        .appName("AggregateSentimentSummaries")
        .getOrCreate()
    )

    with open("src/common/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parquet_dir = cfg["paths"]["processed"]["parquet"]
    out_dir = cfg["paths"]["processed"]["predictions"]  # we’ll store small CSV summaries here too
    os.makedirs(out_dir, exist_ok=True)

    clean_path = os.path.join(parquet_dir, "sentiment_clean.parquet")

    df = spark.read.parquet(clean_path).select("text", "sentiment", "text_len")

    # 1) Class distribution
    class_dist = df.groupBy("sentiment").count().orderBy("sentiment")
    print("\n[+] Class distribution:")
    class_dist.show()

    class_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(out_dir, "summary_class_distribution")
    )

    # 2) Avg / median-ish text length by class
    len_stats = df.groupBy("sentiment").agg(
        F.count("*").alias("n"),
        F.avg("text_len").alias("avg_text_len"),
        F.expr("percentile_approx(text_len, 0.5)").alias("median_text_len"),
        F.max("text_len").alias("max_text_len")
    ).orderBy("sentiment")

    print("\n[+] Text length stats by class:")
    len_stats.show(truncate=False)

    len_stats.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(out_dir, "summary_text_length_stats")
    )

    # 3) Top tokens by class (requires tokens from tfidf stage)
    tfidf_path = os.path.join(parquet_dir, "sentiment_tfidf.parquet")
    tf = spark.read.parquet(tfidf_path).select("sentiment", "tokens_nostop")

    exploded = tf.select("sentiment", F.explode("tokens_nostop").alias("token"))
    token_counts = (
        exploded.groupBy("sentiment", "token")
        .count()
        .orderBy(F.col("count").desc())
    )

    # Top 30 per class
    windowed = token_counts.withColumn(
        "rank",
        F.row_number().over(
            __import__("pyspark").sql.window.Window.partitionBy("sentiment").orderBy(F.col("count").desc())
        )
    ).filter(F.col("rank") <= 30).drop("rank")

    print("\n[+] Top tokens by class (top 30):")
    windowed.show(60, truncate=False)

    windowed.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(out_dir, "summary_top_tokens_by_class")
    )

    print(f"\n[✓] Wrote summaries to: {out_dir}/summary_* (CSV folders)")

    spark.stop()


if __name__ == "__main__":
    main()
