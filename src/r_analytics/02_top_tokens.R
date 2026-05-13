library(readr)
library(dplyr)
library(ggplot2)
library(stringr)
library(tidytext)

# Paths
token_dir <- "data/processed/predictions/summary_top_tokens_by_class"
fig_dir <- "reports/figures"
tab_dir <- "reports/tables"

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

# Spark writes CSVs as folders with part-*.csv
csv_file <- list.files(token_dir, pattern = "^part-.*\\.csv$", full.names = TRUE)[1]

df <- read_csv(csv_file, show_col_types = FALSE)

# Map sentiment labels
df <- df %>%
  mutate(
    sentiment_label = ifelse(sentiment == 1, "Positive", "Negative")
  )

# Save table for report
write.csv(
  df,
  file = file.path(tab_dir, "_top_tokens_by_sentiment.csv"),
  row.names = FALSE
)

# -----------------------------
# Plot: Top tokens per sentiment
# -----------------------------
top_n <- 15

plot_df <- df %>%
  group_by(sentiment_label) %>%
  slice_max(order_by = count, n = top_n) %>%
  ungroup() %>%
  mutate(token = reorder_within(token, count, sentiment_label))

p <- ggplot(plot_df, aes(x = token, y = count, fill = sentiment_label)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ sentiment_label, scales = "free_y") +
  scale_x_reordered() +
  labs(
    title = "Top Tokens by Sentiment",
    x = "Token",
    y = "Frequency"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(fig_dir, "03_top_tokens_by_sentiment.png"),
  plot = p,
  width = 8,
  height = 5,
  dpi = 300
)
