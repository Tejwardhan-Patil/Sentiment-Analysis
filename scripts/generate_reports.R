library(ggplot2)
library(dplyr)
library(knitr)
library(gridExtra)
library(caret)
library(randomForest)

# Define paths for the processed data and model evaluation results
processed_data_path <- "data/processed/"
evaluation_results_path <- "models/evaluation_results.csv"

# Load evaluation results
eval_results <- read.csv(evaluation_results_path)

# Summary statistics of evaluation metrics
eval_summary <- eval_results %>%
  summarise(
    Accuracy = mean(Accuracy, na.rm = TRUE),
    Precision = mean(Precision, na.rm = TRUE),
    Recall = mean(Recall, na.rm = TRUE),
    F1_Score = mean(F1_Score, na.rm = TRUE)
  )

# Save summary statistics to a CSV
write.csv(eval_summary, "reports/eval_summary.csv", row.names = FALSE)

# Visualization 1: Confusion Matrix Heatmap
confusion_matrix <- table(eval_results$True_Label, eval_results$Predicted_Label)
confusion_heatmap <- ggplot(as.data.frame(confusion_matrix), aes(Var1, Var2)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(x = "Predicted Label", y = "True Label", fill = "Count") +
  ggtitle("Confusion Matrix Heatmap")

# Visualization 2: Precision-Recall Curve
precision_recall_plot <- ggplot(eval_results, aes(Recall, Precision)) +
  geom_line(color = "blue") +
  geom_point() +
  labs(x = "Recall", y = "Precision", title = "Precision-Recall Curve")

# Visualization 3: ROC Curve
roc_plot <- ggplot(eval_results, aes(FPR, TPR)) +
  geom_line(color = "red") +
  labs(x = "False Positive Rate", y = "True Positive Rate", title = "ROC Curve") +
  geom_abline(linetype = "dashed")

# Combine all visualizations into a single report
pdf("reports/evaluation_report.pdf", height = 10, width = 10)
grid.arrange(confusion_heatmap, precision_recall_plot, roc_plot, ncol = 2)
dev.off()

# Generate HTML Report using knitr
rmarkdown::render(input = "scripts/report_template.Rmd", 
                  output_file = "../reports/model_evaluation_report.html",
                  params = list(summary = eval_summary))