library(caret)
library(e1071)
library(MLmetrics)

# Function to calculate confusion matrix
calculate_confusion_matrix <- function(actual, predicted) {
  confusionMatrix(as.factor(predicted), as.factor(actual))
}

# Function to calculate accuracy
calculate_accuracy <- function(actual, predicted) {
  Accuracy(y_pred = predicted, y_true = actual)
}

# Function to calculate precision
calculate_precision <- function(actual, predicted) {
  Precision(y_pred = predicted, y_true = actual)
}

# Function to calculate recall
calculate_recall <- function(actual, predicted) {
  Recall(y_pred = predicted, y_true = actual)
}

# Function to calculate F1 Score
calculate_f1 <- function(actual, predicted) {
  F1_Score(y_pred = predicted, y_true = actual)
}

# Function to calculate ROC AUC score
calculate_auc <- function(actual, predicted_prob) {
  roc_curve <- roc(actual, predicted_prob)
  auc(roc_curve)
}

# Function to generate a full evaluation report
generate_metrics_report <- function(actual, predicted, predicted_prob = NULL) {
  report <- list()
  
  report$confusion_matrix <- calculate_confusion_matrix(actual, predicted)
  report$accuracy <- calculate_accuracy(actual, predicted)
  report$precision <- calculate_precision(actual, predicted)
  report$recall <- calculate_recall(actual, predicted)
  report$f1_score <- calculate_f1(actual, predicted)
  
  if (!is.null(predicted_prob)) {
    report$auc <- calculate_auc(actual, predicted_prob)
  }
  
  return(report)
}