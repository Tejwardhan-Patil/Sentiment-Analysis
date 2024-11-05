library(ggplot2)
library(caret)
library(pROC)
library(PRROC)
library(e1071)
library(reshape2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Load the model predictions and true labels
predictions <- read.csv("data/predictions.csv")
true_labels <- read.csv("data/true_labels.csv")

# Combine the data into a single data frame
evaluation_data <- data.frame(
  Prediction = predictions$pred,
  TrueLabel = true_labels$label
)

# Convert predictions and true labels into factors
evaluation_data$Prediction <- factor(evaluation_data$Prediction)
evaluation_data$TrueLabel <- factor(evaluation_data$TrueLabel)

# Confusion matrix and classification metrics
conf_matrix <- confusionMatrix(
  evaluation_data$Prediction,
  evaluation_data$TrueLabel,
  positive = "positive"
)

# Print overall statistics
print(conf_matrix)

# Extract and store specific metrics
overall_accuracy <- conf_matrix$overall['Accuracy']
sensitivity <- conf_matrix$byClass['Sensitivity']
specificity <- conf_matrix$byClass['Specificity']
precision <- conf_matrix$byClass['Pos Pred Value']
f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))

# Display key performance metrics
cat("Overall Accuracy:", overall_accuracy, "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")
cat("Precision:", precision, "\n")
cat("F1 Score:", f1_score, "\n")

# Generate ROC Curve
roc_curve <- roc(evaluation_data$TrueLabel, as.numeric(evaluation_data$Prediction))
plot(roc_curve, main = "ROC Curve")

# Calculate and print AUC
auc_value <- auc(roc_curve)
cat("AUC Value:", auc_value, "\n")

# Precision-Recall Curve
pr_curve <- pr.curve(
  scores.class0 = as.numeric(evaluation_data$Prediction),
  weights.class0 = evaluation_data$TrueLabel,
  curve = TRUE
)
plot(pr_curve, main = "Precision-Recall Curve")

# Calculate Cohen's Kappa
kappa_value <- kappa2(evaluation_data[, c("Prediction", "TrueLabel")])
cat("Cohen's Kappa:", kappa_value$value, "\n")

# Visualizing the confusion matrix as a heatmap
conf_matrix_table <- as.data.frame(table(evaluation_data$TrueLabel, evaluation_data$Prediction))
colnames(conf_matrix_table) <- c("TrueLabel", "Prediction", "Freq")

ggplot(conf_matrix_table, aes(x = TrueLabel, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "True Labels", y = "Predicted Labels") +
  theme_minimal()

# Barplot of True vs Predicted labels proportions
ggplot(evaluation_data, aes(x = TrueLabel, fill = Prediction)) +
  geom_bar(position = "fill") +
  labs(title = "True vs Predicted Labels Proportions", y = "Proportion", x = "True Labels") +
  theme_minimal()

# Plot the class distribution for both true labels and predictions
class_distribution <- evaluation_data %>%
  gather(Type, Label, c(TrueLabel, Prediction)) %>%
  count(Type, Label)

ggplot(class_distribution, aes(x = Label, y = n, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Class Distribution: True vs Predicted", y = "Count", x = "Labels") +
  theme_minimal()

# Receiver Operating Characteristic (ROC) Curve
roc_curve <- roc(evaluation_data$TrueLabel, as.numeric(evaluation_data$Prediction))
plot(roc_curve, col = "blue", main = "ROC Curve")
auc_value <- auc(roc_curve)
cat("Area Under the Curve (AUC):", auc_value, "\n")

# Density plot for model predictions
ggplot(evaluation_data, aes(x = as.numeric(Prediction), fill = TrueLabel)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Predictions by True Labels", x = "Prediction Scores", y = "Density") +
  theme_minimal()

# Save confusion matrix as a CSV file
write.csv(conf_matrix$table, "confusion_matrix.csv", row.names = FALSE)

# Advanced pairwise correlations of prediction probabilities
prediction_probs <- as.numeric(evaluation_data$Prediction)
cor_matrix <- cor(cbind(evaluation_data$TrueLabel, prediction_probs))
cor_melt <- melt(cor_matrix)

ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", high = "green", mid = "yellow", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal() +
  labs(title = "Correlation Matrix", x = "", y = "")

# Pairwise correlation heatmap
ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Heatmap of Pairwise Correlations") +
  theme_minimal()

# F1-Score Visualization
ggplot(evaluation_data, aes(x = TrueLabel, y = Prediction)) +
  geom_jitter(width = 0.2, height = 0.2, color = "blue") +
  labs(title = "Jitter Plot of Predictions", x = "True Labels", y = "Predictions") +
  theme_minimal()

# Boxplot of prediction probabilities by true label
ggplot(evaluation_data, aes(x = TrueLabel, y = as.numeric(Prediction), fill = TrueLabel)) +
  geom_boxplot() +
  labs(title = "Boxplot of Prediction Probabilities by True Label", x = "True Label", y = "Prediction Probabilities") +
  theme_minimal()

# Visualizing the class distribution in both predicted and true labels
label_distribution <- evaluation_data %>%
  group_by(TrueLabel) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = TrueLabel, y = n, fill = TrueLabel)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Class Distribution in True Labels", x = "True Label", y = "Count")

# Displaying multiple plots side by side
grid.arrange(
  ggplot(evaluation_data, aes(x = TrueLabel, y = Prediction)) +
    geom_point(alpha = 0.5) +
    labs(title = "Scatter Plot of Predictions", x = "True Label", y = "Prediction") +
    theme_minimal(),
  
  ggplot(evaluation_data, aes(x = Prediction, fill = TrueLabel)) +
    geom_density(alpha = 0.7) +
    labs(title = "Density Plot of Predictions", x = "Prediction", y = "Density") +
    theme_minimal(),
  
  ncol = 2
)

# Generating performance summary report
performance_summary <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score", "AUC"),
  Value = c(overall_accuracy, sensitivity, specificity, precision, f1_score, auc_value)
)
write.csv(performance_summary, "performance_summary.csv", row.names = FALSE)

# Save visualizations
ggsave("confusion_matrix_heatmap.png", width = 8, height = 6)
ggsave("roc_curve_plot.png", width = 8, height = 6)
ggsave("pr_curve_plot.png", width = 8, height = 6)
ggsave("class_distribution_plot.png", width = 8, height = 6)

# Statistical analysis: t-test for model performance across different classes
t_test_result <- t.test(
  as.numeric(evaluation_data$Prediction) ~ evaluation_data$TrueLabel
)
cat("T-test results:\n", t_test_result, "\n")

# Compute and visualize residuals (difference between predicted and true labels)
evaluation_data$residuals <- as.numeric(evaluation_data$TrueLabel) - as.numeric(evaluation_data$Prediction)

ggplot(evaluation_data, aes(x = residuals)) +
  geom_histogram(binwidth = 0.1, fill = "blue", color = "white") +
  labs(title = "Residuals Distribution", x = "Residuals", y = "Count") +
  theme_minimal()

# Save residuals data
write.csv(evaluation_data$residuals, file = "residuals.csv", row.names = FALSE)

# Final summary output
cat("Evaluation completed. All visualizations and metrics have been saved.\n")