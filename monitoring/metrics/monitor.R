library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(gridExtra)
library(reshape2)
library(zoo)
library(scales)

# Load the performance metrics CSV or other data sources
metrics_data <- read.csv("/metrics.csv")

# Ensure the date column is in date format
metrics_data$date <- as.Date(metrics_data$date, format = "%Y-%m-%d")

# Remove rows with missing or invalid values
metrics_data <- metrics_data %>%
  filter(!is.na(accuracy) & !is.na(precision) & !is.na(recall) & !is.na(f1_score))

# Smooth metrics using a rolling average (7 days window)
metrics_data <- metrics_data %>%
  arrange(date) %>%
  mutate(
    smoothed_accuracy = rollmean(accuracy, 7, fill = NA),
    smoothed_precision = rollmean(precision, 7, fill = NA),
    smoothed_recall = rollmean(recall, 7, fill = NA),
    smoothed_f1 = rollmean(f1_score, 7, fill = NA)
  )

# Aggregate metrics by month for better visualization
metrics_summary <- metrics_data %>%
  group_by(month = floor_date(date, "month")) %>%
  summarize(
    avg_accuracy = mean(accuracy, na.rm = TRUE),
    avg_precision = mean(precision, na.rm = TRUE),
    avg_recall = mean(recall, na.rm = TRUE),
    avg_f1 = mean(f1_score, na.rm = TRUE)
  )

# Plot Accuracy Over Time
plot_accuracy <- ggplot(metrics_summary, aes(x = month)) +
  geom_line(aes(y = avg_accuracy, color = "Accuracy")) +
  labs(title = "Model Accuracy Over Time", x = "Month", y = "Average Accuracy") +
  theme_minimal()

# Plot Smoothed Accuracy Over Time
plot_smoothed_accuracy <- ggplot(metrics_data, aes(x = date)) +
  geom_line(aes(y = smoothed_accuracy, color = "Smoothed Accuracy")) +
  labs(title = "Smoothed Accuracy Over Time (7-day rolling average)", x = "Date", y = "Smoothed Accuracy") +
  theme_minimal()

# Plot Precision, Recall, and F1-Score Over Time
metrics_long <- metrics_summary %>%
  pivot_longer(cols = c(avg_precision, avg_recall, avg_f1), names_to = "metric", values_to = "value")

plot_metrics <- ggplot(metrics_long, aes(x = month, y = value, color = metric)) +
  geom_line() +
  labs(title = "Precision, Recall, and F1-Score Over Time", x = "Month", y = "Metric Value") +
  theme_minimal()

# Boxplot of Metrics Distribution
plot_boxplot <- ggplot(metrics_data, aes(x = "", y = accuracy)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red") +
  labs(title = "Accuracy Distribution", x = "", y = "Accuracy") +
  theme_minimal()

# Density plots for all metrics
plot_density <- ggplot(metrics_data, aes(x = accuracy)) +
  geom_density(fill = "lightgreen", alpha = 0.7) +
  labs(title = "Density Plot of Accuracy", x = "Accuracy", y = "Density") +
  theme_minimal()

# Scatterplot of Precision vs. Recall
plot_scatter <- ggplot(metrics_data, aes(x = precision, y = recall)) +
  geom_point(color = "darkblue", alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Precision vs Recall", x = "Precision", y = "Recall") +
  theme_minimal()

# Outlier Detection: Identifying potential outliers in accuracy
accuracy_outliers <- metrics_data %>%
  filter(accuracy < quantile(accuracy, 0.05) | accuracy > quantile(accuracy, 0.95))

# Visualizing the outliers
plot_outliers <- ggplot(metrics_data, aes(x = date, y = accuracy)) +
  geom_line(color = "gray") +
  geom_point(data = accuracy_outliers, aes(x = date, y = accuracy), color = "red") +
  labs(title = "Outlier Detection in Accuracy", x = "Date", y = "Accuracy") +
  theme_minimal()

# Calculate the correlation between Precision and Recall
correlation_precision_recall <- cor(metrics_data$precision, metrics_data$recall, use = "complete.obs")

# Summary Statistics for each metric
metrics_summary_stats <- metrics_data %>%
  summarize(
    min_accuracy = min(accuracy),
    max_accuracy = max(accuracy),
    mean_accuracy = mean(accuracy),
    sd_accuracy = sd(accuracy),
    min_precision = min(precision),
    max_precision = max(precision),
    mean_precision = mean(precision),
    sd_precision = sd(precision),
    min_recall = min(recall),
    max_recall = max(recall),
    mean_recall = mean(recall),
    sd_recall = sd(recall),
    min_f1 = min(f1_score),
    max_f1 = max(f1_score),
    mean_f1 = mean(f1_score),
    sd_f1 = sd(f1_score)
  )

# Perform a statistical test to compare precision and recall distributions
precision_recall_t_test <- t.test(metrics_data$precision, metrics_data$recall)

# Generate a correlation matrix for all metrics
correlation_matrix <- cor(metrics_data %>% select(accuracy, precision, recall, f1_score), use = "complete.obs")

# Visualize the correlation matrix
correlation_melt <- melt(correlation_matrix)
plot_correlation_matrix <- ggplot(data = correlation_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +
  labs(title = "Correlation Matrix of Metrics", x = "Metric", y = "Metric") +
  theme_minimal()

# Arrange the plots into a grid
grid.arrange(plot_accuracy, plot_smoothed_accuracy, plot_metrics, ncol = 2)
grid.arrange(plot_boxplot, plot_density, plot_scatter, plot_outliers, ncol = 2)
grid.arrange(plot_correlation_matrix, ncol = 1)

# Save the plots to files
ggsave("accuracy_over_time.png", plot = plot_accuracy, width = 8, height = 5)
ggsave("smoothed_accuracy_over_time.png", plot = plot_smoothed_accuracy, width = 8, height = 5)
ggsave("metrics_over_time.png", plot = plot_metrics, width = 8, height = 5)
ggsave("accuracy_distribution.png", plot = plot_boxplot, width = 8, height = 5)
ggsave("accuracy_density.png", plot = plot_density, width = 8, height = 5)
ggsave("precision_vs_recall.png", plot = plot_scatter, width = 8, height = 5)
ggsave("accuracy_outliers.png", plot = plot_outliers, width = 8, height = 5)
ggsave("correlation_matrix.png", plot = plot_correlation_matrix, width = 8, height = 5)

# Print summary statistics
print("Summary Statistics:")
print(metrics_summary_stats)

# Print correlation between precision and recall
print(paste("Correlation between Precision and Recall: ", round(correlation_precision_recall, 3)))

# Print the results of the t-test
print("T-test between Precision and Recall:")
print(precision_recall_t_test)

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)