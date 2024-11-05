library(caret)
library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(mlr3pipelines)
library(paradox)
library(mlr3misc)
library(data.table)
library(ggplot2)
library(yaml)

# Set random seed for reproducibility
set.seed(123)

# Load configuration (e.g., hyperparameter ranges, paths)
config <- yaml::read_yaml('experiments/configs/config.yaml')

# Load and prepare the dataset
data <- fread('data/processed/dataset.csv')

# Display the structure of the dataset
print(str(data))

# Check for missing values
missing_values <- colSums(is.na(data))
print("Missing values in each column:")
print(missing_values)

# Handle missing values (e.g., impute or remove)
data[is.na(data)] <- 0  # Simple imputation by replacing NA with 0

# Normalize the numerical columns
num_cols <- sapply(data, is.numeric)
data[, (names(data)[num_cols]) := lapply(.SD, scale), .SDcols = num_cols]

# Convert categorical variables to factors
cat_cols <- sapply(data, is.character)
data[, (names(data)[cat_cols]) := lapply(.SD, as.factor), .SDcols = cat_cols]

# Split data into training and testing sets
trainIndex <- createDataPartition(data$sentiment, p = .8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Create mlr3 task
task <- TaskClassif$new(id = "sentiment", backend = trainData, target = "sentiment")

# Create learner (random forest)
learner <- lrn("classif.ranger", predict_type = "prob")

# Define the resampling strategy (cross-validation with stratified folds)
resampling <- rsmp("cv", folds = 5)

# Setup hyperparameter search space
param_set <- ParamSet$new(params = list(
  ParamDbl$new("mtry", lower = 1, upper = ncol(trainData) - 1),
  ParamInt$new("min.node.size", lower = 1, upper = 10),
  ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
  ParamInt$new("num.trees", lower = 100, upper = 1000)
))

# Define the tuner using grid search
tuner <- tnr("grid_search", resolution = 10)

# Logging function to track progress
log_progress <- function(message) {
  cat(Sys.time(), "-", message, "\n")
}

log_progress("Starting hyperparameter tuning...")

# Create tuning instance
instance <- TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = resampling,
  measure = msr("classif.auc"),
  search_space = param_set,
  terminator = trm("evals", n_evals = 50)
)

# Start tuning process
tuner$optimize(instance)

log_progress("Hyperparameter tuning completed.")

# Extract best hyperparameters and re-train the model on full training set
best_params <- instance$result_learner_param_vals
learner$param_set$values <- best_params

log_progress("Training final model with best hyperparameters...")

learner$train(task)

# Save the best model for future use
saveRDS(learner, "models/tuned_ranger_model.rds")
log_progress("Best model saved.")

# Prepare test data for evaluation
task_test <- TaskClassif$new(id = "sentiment_test", backend = testData, target = "sentiment")

# Generate predictions on test set
prediction <- learner$predict(task_test)

# Evaluation metrics: AUC, accuracy, precision, recall, F1-score
metrics <- list(
  auc = msr("classif.auc"),
  accuracy = msr("classif.acc"),
  precision = msr("classif.precision"),
  recall = msr("classif.recall"),
  f1 = msr("classif.f1")
)

# Calculate metrics on the test set
results <- lapply(metrics, function(m) prediction$score(m))
names(results) <- names(metrics)

log_progress("Evaluation completed on test set.")
print("Test set performance metrics:")
print(results)

# Confusion matrix
conf_matrix <- table(prediction$truth(), prediction$response())
print("Confusion Matrix:")
print(conf_matrix)

# Visualize ROC curve
roc_data <- as.data.frame(prediction$score(msr("classif.roc")))
ggplot(roc_data, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(linetype = "dashed") +
  labs(title = "ROC Curve", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()

# Save evaluation results to CSV
eval_results <- data.table(
  auc = results$auc,
  accuracy = results$accuracy,
  precision = results$precision,
  recall = results$recall,
  f1 = results$f1
)
fwrite(eval_results, "experiments/results/evaluation_metrics.csv")

log_progress("Evaluation metrics saved.")

# Save the final prediction on test data
predictions_df <- data.table(truth = prediction$truth(), response = prediction$response(), prob = prediction$prob[, 2])
fwrite(predictions_df, "experiments/results/test_predictions.csv")

log_progress("Predictions on test data saved.")

# Variable importance plot
importance <- learner$importance()
importance_df <- data.frame(
  Variable = names(importance),
  Importance = as.numeric(importance)
)

ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Feature", y = "Importance") +
  theme_minimal()

log_progress("Variable importance plot generated.")

# Advanced error analysis: Misclassified samples
misclassified <- testData[prediction$response() != prediction$truth()]
log_progress("Number of misclassified samples:")
print(nrow(misclassified))

# Save misclassified samples for further review
fwrite(misclassified, "experiments/results/misclassified_samples.csv")

# Advanced error analysis: Class-wise accuracy
class_acc <- prediction$confusion$calculate(msr("classif.acc", positive = "Positive"))
log_progress("Class-wise accuracy:")
print(class_acc)

# Generate detailed report of the entire experiment
log_progress("Generating final report...")

report_content <- paste(
  "Experiment Summary:\n",
  "Best Hyperparameters:\n", print(best_params), "\n",
  "Test Set Metrics:\n", print(eval_results), "\n",
  "Confusion Matrix:\n", print(conf_matrix), "\n",
  "Class-wise Accuracy:\n", print(class_acc), "\n"
)

writeLines(report_content, con = "experiments/results/final_report.txt")

log_progress("Final report generated.")

# End of script
log_progress("Experiment script completed successfully.")