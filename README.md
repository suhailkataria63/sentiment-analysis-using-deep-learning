# Sentiment Analysis Using Deep Learning

This repository contains a structured sentiment-analysis project that combines Spark data processing, R analytics, a deep-learning model, a FastAPI prediction service, and a Shiny dashboard.

## Project Structure

```text
.
|-- dashboard/              # R Shiny dashboard and dashboard modules
|-- data/
|   |-- raw/                # Original input data
|   |-- processed/          # Prepared features, parquet outputs, predictions, summaries
|-- models/
|   |-- sentiment/          # Trained GRU model and tokenizer
|-- reports/
|   |-- figures/            # Generated charts
|   |-- tables/             # Metrics and report tables
|-- requirements/           # Python and R dependency lists
|-- scripts/                # Pipeline runner scripts
|-- src/
|   |-- api/                # FastAPI app and web template
|   |-- common/             # Shared configuration
|   |-- dl/                 # Deep-learning dataset, training, prediction, metrics
|   |-- r_analytics/        # R EDA and topic-modeling scripts
|   |-- spark/              # Spark ingest, cleaning, feature, and trend jobs
```

## Main Workflows

1. Run the Spark pipeline to ingest, clean, featurize, and aggregate text data:

```bash
bash scripts/run_spark_pipeline.sh
```

2. Train and evaluate the deep-learning sentiment model:

```bash
bash scripts/run_dl_pipeline.sh
```

3. Generate R analytics outputs:

```bash
bash scripts/run_r_analytics.sh
```

4. Serve predictions with FastAPI:

```bash
uvicorn src.api.app:app --reload --port 8001
```

5. Open the R Shiny dashboard from the `dashboard/` directory.

## Repository Hygiene

The repository keeps the project in its original structured folders and excludes local/generated machine files such as virtual environments, `.DS_Store`, Python caches, R session files, backup files, IDE settings, and Spark checksum sidecars. This avoids duplicate flat-file uploads while preserving the actual project code, data artifacts, trained model artifacts, reports, and scripts.
