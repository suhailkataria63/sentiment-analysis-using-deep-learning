library(arrow)
library(dplyr)
library(ggplot2)
library(stringr)

# Paths
parquet_path <- "data/processed/parquet/sentiment_clean.parquet"
fig_dir <- "reports/figures"
tab_dir <- "reports/tables"

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
df <- open_dataset(parquet_path) |> collect()

# Map sentiment to readable labels
df <- df %>%
  mutate(sentiment_label = ifelse(sentiment == 1, "Positive", "Negative"))

# -----------------------------
# 1) Class distribution
# -----------------------------
p1 <- ggplot(df, aes(x = sentiment_label)) +
  geom_bar() +
  labs(
    title = "Sentiment Class Distribution",
    x = "Sentiment",
    y = "Count"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(fig_dir, "01_class_distribution.png"),
  plot = p1,
  width = 6,
  height = 4,
  dpi = 300
)

# -----------------------------
# 2) Text length distribution
# -----------------------------
# Clip x-axis at 99th percentile to avoid long-tail stretching the plot
xmax <- quantile(df$text_len, 0.99, na.rm = TRUE)

p2 <- ggplot(df, aes(x = text_len, fill = sentiment_label)) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  coord_cartesian(xlim = c(0, xmax)) +
  labs(
    title = "Text Length Distribution by Sentiment",
    x = "Text length (characters)",
    y = "Frequency",
    fill = "Sentiment"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(fig_dir, "02_text_length_distribution.png"),
  plot = p2,
  width = 7,
  height = 4,
  dpi = 300
)

# -----------------------------
# 3) Summary table for report
# -----------------------------
summary_tbl <- df %>%
  group_by(sentiment_label) %>%
  summarise(
    n = n(),
    avg_text_len = mean(text_len, na.rm = TRUE),
    median_text_len = median(text_len, na.rm = TRUE),
    max_text_len = max(text_len, na.rm = TRUE)
  )

print(summary_tbl)

write.csv(
  summary_tbl,
  file = file.path(tab_dir, "01_summary_text_length_by_sentiment.csv"),
  row.names = FALSE
)
