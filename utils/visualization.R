library(ggplot2)
library(ggpubr)
library(dplyr)
library(tidyr)
library(viridis)
library(reshape2)

# Function to visualize confusion matrix
plot_confusion_matrix <- function(conf_matrix, class_labels) {
  conf_matrix_melt <- melt(conf_matrix)
  colnames(conf_matrix_melt) <- c("Predicted", "Actual", "Count")
  
  ggplot(conf_matrix_melt, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile(color = "white") +
    scale_fill_viridis(option = "C") +
    geom_text(aes(label = sprintf("%d", Count)), vjust = 1, color = "white") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      title = "Confusion Matrix",
      x = "Predicted Class",
      y = "Actual Class",
      fill = "Count"
    )
}

# Function to plot ROC curve
plot_roc_curve <- function(roc_data) {
  ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
    geom_line(size = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_minimal() +
    labs(
      title = "ROC Curve",
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = "Model"
    ) +
    geom_abline(linetype = "dashed", color = "gray")
}

# Function to plot precision-recall curve
plot_pr_curve <- function(pr_data) {
  ggplot(pr_data, aes(x = Recall, y = Precision, color = Model)) +
    geom_line(size = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_minimal() +
    labs(
      title = "Precision-Recall Curve",
      x = "Recall",
      y = "Precision",
      color = "Model"
    )
}

# Function to plot accuracy and loss over epochs
plot_training_metrics <- function(metrics_data) {
  metrics_data_long <- gather(metrics_data, key = "Metric", value = "Value", -Epoch)
  
  ggplot(metrics_data_long, aes(x = Epoch, y = Value, color = Metric)) +
    geom_line(size = 1) +
    scale_color_viridis(discrete = TRUE) +
    theme_minimal() +
    labs(
      title = "Model Training Metrics",
      x = "Epoch",
      y = "Value",
      color = "Metric"
    )
}

# Function to visualize sentiment distribution
plot_sentiment_distribution <- function(sentiment_data) {
  ggplot(sentiment_data, aes(x = Sentiment, fill = Sentiment)) +
    geom_bar() +
    scale_fill_viridis(discrete = TRUE) +
    theme_minimal() +
    labs(
      title = "Sentiment Distribution",
      x = "Sentiment",
      y = "Count",
      fill = "Sentiment"
    )
}

# Function to visualize word cloud from text data
plot_wordcloud <- function(word_freq_data) {
  wordcloud2::wordcloud2(word_freq_data, color = "random-light", backgroundColor = "white")
}

# Function call
# plot_confusion_matrix(conf_matrix, class_labels)
# plot_roc_curve(roc_data)
# plot_pr_curve(pr_data)
# plot_training_metrics(metrics_data)
# plot_sentiment_distribution(sentiment_data)
# plot_wordcloud(word_freq_data)