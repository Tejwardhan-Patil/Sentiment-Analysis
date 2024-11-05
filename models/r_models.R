library(glmnet)
library(randomForest)
library(caret)
library(tidyverse)
library(text2vec)
library(Matrix)
library(e1071)
library(pROC)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)

# Load preprocessed data
data_train <- read.csv("data/processed/train.csv")
data_test <- read.csv("data/processed/test.csv")

# Data inspection
str(data_train)
summary(data_train)
table(data_train$label)

# Visualize word clouds for positive and negative sentiments
positive_text <- data_train %>% filter(label == "positive") %>% pull(text)
negative_text <- data_train %>% filter(label == "negative") %>% pull(text)

wordcloud(positive_text, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))
wordcloud(negative_text, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Reds"))

# Preprocessing for model input (converting text data to document-term matrix)
prep_tokenizer <- function(text_column) {
  itoken(text_column, preprocess_function = tolower, tokenizer = word_tokenizer)
}

# Tokenization
train_tokens <- prep_tokenizer(data_train$text)
test_tokens <- prep_tokenizer(data_test$text)

# Create vocabulary and document-term matrices
vocabulary <- create_vocabulary(train_tokens)
vectorizer <- vocab_vectorizer(vocabulary)
dtm_train <- create_dtm(train_tokens, vectorizer)
dtm_test <- create_dtm(test_tokens, vectorizer)

# Feature scaling (term frequency-inverse document frequency)
tfidf <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- transform(dtm_test, tfidf)

# Convert labels to factor for classification models
y_train <- as.factor(data_train$label)
y_test <- as.factor(data_test$label)

# Model 1: Regularized Logistic Regression (GLMNET)
set.seed(123)
cv_glmnet <- cv.glmnet(x = dtm_train_tfidf, y = y_train, family = "binomial", alpha = 1, nfolds = 5)
glmnet_pred <- predict(cv_glmnet, newx = dtm_test_tfidf, type = "class")

# Model 2: Random Forest
set.seed(123)
rf_model <- randomForest(x = dtm_train_tfidf, y = y_train, ntree = 500, mtry = 30)
rf_pred <- predict(rf_model, newdata = dtm_test_tfidf)

# Model 3: Support Vector Machines (SVM)
set.seed(123)
svm_model <- train(dtm_train_tfidf, y_train, method = "svmRadial", 
                   trControl = trainControl(method = "cv", number = 5))
svm_pred <- predict(svm_model, newdata = dtm_test_tfidf)

# Model 4: Naive Bayes
nb_model <- naiveBayes(dtm_train_tfidf, y_train)
nb_pred <- predict(nb_model, dtm_test_tfidf)

# Model 5: XGBoost
xgb_train <- xgb.DMatrix(data = as.matrix(dtm_train_tfidf), label = as.numeric(y_train) - 1)
xgb_test <- xgb.DMatrix(data = as.matrix(dtm_test_tfidf))

params <- list(objective = "binary:logistic", eval_metric = "auc", max_depth = 6, eta = 0.1)
xgb_model <- xgboost(params = params, data = xgb_train, nrounds = 100)
xgb_pred <- predict(xgb_model, xgb_test)
xgb_pred_class <- ifelse(xgb_pred > 0.5, "positive", "negative")

# Evaluation metrics for individual models
glmnet_conf_matrix <- confusionMatrix(glmnet_pred, y_test)
rf_conf_matrix <- confusionMatrix(rf_pred, y_test)
svm_conf_matrix <- confusionMatrix(svm_pred, y_test)
nb_conf_matrix <- confusionMatrix(nb_pred, y_test)

# AUC for GLMNET
roc_glmnet <- roc(response = y_test, predictor = as.numeric(glmnet_pred), levels = rev(levels(y_test)))
auc_glmnet <- auc(roc_glmnet)

# AUC for Random Forest
roc_rf <- roc(response = y_test, predictor = as.numeric(rf_pred), levels = rev(levels(y_test)))
auc_rf <- auc(roc_rf)

# AUC for SVM
roc_svm <- roc(response = y_test, predictor = as.numeric(svm_pred), levels = rev(levels(y_test)))
auc_svm <- auc(roc_svm)

# AUC for Naive Bayes
roc_nb <- roc(response = y_test, predictor = as.numeric(nb_pred), levels = rev(levels(y_test)))
auc_nb <- auc(roc_nb)

# AUC for XGBoost
roc_xgb <- roc(response = y_test, predictor = as.numeric(xgb_pred_class), levels = rev(levels(y_test)))
auc_xgb <- auc(roc_xgb)

# Print evaluation metrics
print(glmnet_conf_matrix)
print(rf_conf_matrix)
print(svm_conf_matrix)
print(nb_conf_matrix)
print(paste("AUC for GLMNET:", auc_glmnet))
print(paste("AUC for Random Forest:", auc_rf))
print(paste("AUC for SVM:", auc_svm))
print(paste("AUC for Naive Bayes:", auc_nb))
print(paste("AUC for XGBoost:", auc_xgb))

# Ensemble method (Voting)
predictions <- data.frame(glmnet_pred, rf_pred, svm_pred, nb_pred, xgb_pred_class)
ensemble_pred <- apply(predictions, 1, function(x) names(sort(table(x), decreasing = TRUE))[1])

# Final evaluation for ensemble method
ensemble_conf_matrix <- confusionMatrix(as.factor(ensemble_pred), y_test)
ensemble_roc <- roc(response = y_test, predictor = as.numeric(ensemble_pred), levels = rev(levels(y_test)))
ensemble_auc <- auc(ensemble_roc)

# Print ensemble results
print(ensemble_conf_matrix)
print(paste("AUC for Ensemble:", ensemble_auc))

# Plot AUC for all models
roc_plot <- ggroc(list(GLMNET = roc_glmnet, RandomForest = roc_rf, SVM = roc_svm, NaiveBayes = roc_nb, XGBoost = roc_xgb, Ensemble = ensemble_roc)) +
  ggtitle("ROC Curves for All Models") +
  labs(color = "Models") +
  theme_minimal()

print(roc_plot)

# Model Hyperparameter Tuning for Random Forest
tuned_rf_model <- train(dtm_train_tfidf, y_train, method = "rf", tuneLength = 10, 
                        trControl = trainControl(method = "cv", number = 5))

rf_tuned_pred <- predict(tuned_rf_model, newdata = dtm_test_tfidf)
rf_tuned_conf_matrix <- confusionMatrix(rf_tuned_pred, y_test)
print(rf_tuned_conf_matrix)

# Feature Importance from Random Forest
rf_importance <- varImp(tuned_rf_model, scale = TRUE)
plot(rf_importance, top = 20)

# Feature Engineering: Adding more features
data_train$char_length <- nchar(data_train$text)
data_test$char_length <- nchar(data_test$text)

data_train$word_count <- sapply(strsplit(data_train$text, "\\s+"), length)
data_test$word_count <- sapply(strsplit(data_test$text, "\\s+"), length)

# Rebuild the document-term matrices with the new features included
extra_features_train <- data.matrix(data_train[, c("char_length", "word_count")])
extra_features_test <- data.matrix(data_test[, c("char_length", "word_count")])

# Bind the additional features to the DTM
dtm_train_full <- cBind(dtm_train_tfidf, extra_features_train)
dtm_test_full <- cBind(dtm_test_tfidf, extra_features_test)

# Train models again on the extended feature set
tuned_rf_model_full <- train(dtm_train_full, y_train, method = "rf", tuneLength = 10, 
                             trControl = trainControl(method = "cv", number = 5))
rf_full_pred <- predict(tuned_rf_model_full, newdata = dtm_test_full)
rf_full_conf_matrix <- confusionMatrix(rf_full_pred, y_test)

print(rf_full_conf_matrix)

# Save the models for future inference
saveRDS(cv_glmnet, file = "models/glmnet_model.rds")
saveRDS(rf_model, file = "models/rf_model.rds")
saveRDS(svm_model, file = "models/svm_model.rds")
saveRDS(nb_model, file = "models/nb_model.rds")
saveRDS(xgb_model, file = "models/xgb_model.rds")
saveRDS(tuned_rf_model_full, file = "models/tuned_rf_full_model.rds")

# Save predictions
write.csv(data.frame(ensemble_pred = ensemble_pred), file = "results/ensemble_predictions.csv")
write.csv(data.frame(glmnet_pred = glmnet_pred), file = "results/glmnet_predictions.csv")
write.csv(data.frame(rf_pred = rf_pred), file = "results/rf_predictions.csv")