library(testthat)
library(glmnet)  
library(randomForest) 

# Source the R models script to test the implemented models
source("models/r_models.R")

# Test Suite for Generalized Linear Models (GLM)
test_that("GLM model works as expected", {
  
  # Mock input data for GLM
  x_train <- as.matrix(data.frame(
    feature1 = rnorm(100),
    feature2 = rnorm(100),
    feature3 = rnorm(100)
  ))
  
  y_train <- rbinom(100, 1, 0.5)
  
  # Test that the model runs without errors
  glm_model <- train_glm(x_train, y_train)
  expect_true(is(glm_model, "cv.glmnet"))
  
  # Ensure predictions return expected number of results
  predictions <- predict(glm_model, newx = x_train, s = "lambda.min")
  expect_equal(length(predictions), nrow(x_train))
  
  # Test that prediction output is numeric
  expect_true(is.numeric(predictions))
  
  # Check that lambda value is selected correctly
  expect_true(is.numeric(glm_model$lambda.min))
})

# Edge case testing for GLM
test_that("GLM model handles edge cases", {
  
  # Empty input data
  x_empty <- matrix(, nrow = 0, ncol = 3)
  y_empty <- numeric(0)
  
  expect_error(train_glm(x_empty, y_empty), "error")
  
  # Single data point case
  x_single <- as.matrix(data.frame(
    feature1 = rnorm(1),
    feature2 = rnorm(1),
    feature3 = rnorm(1)
  ))
  
  y_single <- rbinom(1, 1, 0.5)
  
  glm_model_single <- train_glm(x_single, y_single)
  expect_true(is(glm_model_single, "cv.glmnet"))
})

# Test Suite for Random Forest Models
test_that("Random Forest model works as expected", {
  
  # Mock input data for Random Forest
  x_train <- data.frame(
    feature1 = rnorm(100),
    feature2 = rnorm(100),
    feature3 = rnorm(100)
  )
  
  y_train <- as.factor(rbinom(100, 1, 0.5))
  
  # Train the random forest model
  rf_model <- train_random_forest(x_train, y_train)
  expect_true(is(rf_model, "randomForest"))
  
  # Ensure predictions return expected number of results
  predictions <- predict(rf_model, newdata = x_train)
  expect_equal(length(predictions), nrow(x_train))
  
  # Test that prediction output is a factor
  expect_true(is.factor(predictions))
  
  # Check importance of features
  importance <- importance(rf_model)
  expect_true(is.numeric(importance))
})

# Test for Random Forest model on small dataset
test_that("Random Forest handles small dataset", {
  
  # Create a small dataset
  x_small <- data.frame(
    feature1 = rnorm(5),
    feature2 = rnorm(5),
    feature3 = rnorm(5)
  )
  
  y_small <- as.factor(rbinom(5, 1, 0.5))
  
  # Train the random forest model on small dataset
  rf_model_small <- train_random_forest(x_small, y_small)
  expect_true(is(rf_model_small, "randomForest"))
})

# Test Suite for Model Performance Metrics
test_that("Model performance metrics are calculated correctly", {
  
  # Mock predictions and actual values
  predictions <- rbinom(100, 1, 0.5)
  actual <- rbinom(100, 1, 0.5)
  
  # Test accuracy function
  accuracy <- calculate_accuracy(predictions, actual)
  expect_true(is.numeric(accuracy))
  expect_gte(accuracy, 0)
  expect_lte(accuracy, 1)
  
  # Test precision function
  precision <- calculate_precision(predictions, actual)
  expect_true(is.numeric(precision))
  expect_gte(precision, 0)
  expect_lte(precision, 1)
  
  # Test recall function
  recall <- calculate_recall(predictions, actual)
  expect_true(is.numeric(recall))
  expect_gte(recall, 0)
  expect_lte(recall, 1)
  
  # Test F1-score function
  f1_score <- calculate_f1_score(predictions, actual)
  expect_true(is.numeric(f1_score))
  expect_gte(f1_score, 0)
  expect_lte(f1_score, 1)
})

# Test Suite for Data Preprocessing
test_that("Data preprocessing functions handle data correctly", {
  
  # Mock raw text data
  raw_text <- c("I love this product!", "This is the worst service ever.", "Absolutely fantastic!")
  
  # Test the text cleaning function
  cleaned_text <- preprocess_text(raw_text)
  expect_true(is.character(cleaned_text))
  expect_equal(length(cleaned_text), length(raw_text))
  
  # Check that stopwords are removed correctly
  expect_false(any(grepl("\\bthe\\b", cleaned_text)))
})

# Test text preprocessing for empty input
test_that("Text preprocessing handles empty input", {
  
  empty_text <- character(0)
  cleaned_empty <- preprocess_text(empty_text)
  
  # Ensure the output is still a character vector
  expect_true(is.character(cleaned_empty))
  expect_equal(length(cleaned_empty), 0)
})

# Test text preprocessing for edge cases
test_that("Text preprocessing handles special characters", {
  
  special_text <- c("Hello!!!", "How's it going?", "Text with @mentions and #hashtags.")
  
  cleaned_special <- preprocess_text(special_text)
  
  # Check if punctuation is removed
  expect_false(any(grepl("[!@#]", cleaned_special)))
})

# Additional tests for edge case in model performance
test_that("Model performance metrics handle edge cases", {
  
  # All predictions correct
  all_correct_predictions <- rep(1, 100)
  actual_all_correct <- rep(1, 100)
  
  accuracy <- calculate_accuracy(all_correct_predictions, actual_all_correct)
  expect_equal(accuracy, 1)
  
  # All predictions wrong
  all_wrong_predictions <- rep(1, 100)
  actual_all_wrong <- rep(0, 100)
  
  accuracy <- calculate_accuracy(all_wrong_predictions, actual_all_wrong)
  expect_equal(accuracy, 0)
  
  # Test for edge cases in F1-score
  f1_all_correct <- calculate_f1_score(all_correct_predictions, actual_all_correct)
  expect_equal(f1_all_correct, 1)
  
  f1_all_wrong <- calculate_f1_score(all_wrong_predictions, actual_all_wrong)
  expect_equal(f1_all_wrong, 0)
})

# Test Suite for Model Saving and Loading
test_that("Model saving and loading works", {
  
  # Create a mock model (using random forest for this test)
  x_train <- data.frame(
    feature1 = rnorm(100),
    feature2 = rnorm(100),
    feature3 = rnorm(100)
  )
  y_train <- as.factor(rbinom(100, 1, 0.5))
  
  rf_model <- train_random_forest(x_train, y_train)
  
  # Save the model
  save_model(rf_model, "rf_model_test.rds")
  expect_true(file.exists("rf_model_test.rds"))
  
  # Load the model
  loaded_rf_model <- load_model("rf_model_test.rds")
  expect_true(is(loaded_rf_model, "randomForest"))
  
  # Clean up
  file.remove("rf_model_test.rds")
})

# Test Suite for Model Training Time
test_that("Model training time is within expected range", {
  
  # Mock input data
  x_train <- data.frame(
    feature1 = rnorm(1000),
    feature2 = rnorm(1000),
    feature3 = rnorm(1000)
  )
  
  y_train <- as.factor(rbinom(1000, 1, 0.5))
  
  # Measure the time for training a random forest model
  start_time <- Sys.time()
  rf_model <- train_random_forest(x_train, y_train)
  end_time <- Sys.time()
  
  training_time <- end_time - start_time
  expect_lt(as.numeric(training_time), 60)  # Training should take less than 60 seconds
})

# Test for model inference
test_that("Model inference works correctly", {
  
  # Mock input data
  x_test <- data.frame(
    feature1 = rnorm(10),
    feature2 = rnorm(10),
    feature3 = rnorm(10)
  )
  
  y_test <- as.factor(rbinom(10, 1, 0.5))
  
  # Load a pre-trained model
  loaded_rf_model <- load_model("pretrained_rf_model.rds")
  
  # Test inference
  predictions <- predict(loaded_rf_model, newdata = x_test)
  expect_true(is.factor(predictions))
  expect_equal(length(predictions), nrow(x_test))
})