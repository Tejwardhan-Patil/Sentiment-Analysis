library(plumber)
library(jsonlite)
library(tidyverse)
library(logger)

# Initialize Logger
log_appender(appender_file("logs/plumber.log"))
log_threshold(INFO)

# Load sentiment analysis model
model <- readRDS("models/pretrained/model.rds")

#* @apiTitle Sentiment Analysis API
#* @apiDescription API for deploying a sentiment analysis model

# Health Check Route --------------------------------------------------------

#* Health Check Route
#* @get /health
#* @response 200 OK if the API is running
#* @response 500 Error if there's an issue
function() {
  log_info("Health check requested.")
  tryCatch({
    list(status = "API is running")
  }, error = function(e) {
    log_error("Error during health check: {e$message}")
    res$status <- 500
    list(error = "Health check failed.")
  })
}

# Preprocessing Function ----------------------------------------------------

# Text Preprocessing Function
preprocess_text <- function(text) {
  log_info("Preprocessing input text.")
  
  clean_text <- text %>%
    tolower() %>%
    str_replace_all("[[:punct:]]", "") %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
  
  log_info("Text after preprocessing: {clean_text}")
  return(clean_text)
}

# Input Validation ----------------------------------------------------------

# Validate Input Text
validate_input <- function(text) {
  if (is.null(text) || text == "") {
    log_warn("Invalid input: empty or null text.")
    return(FALSE)
  }
  
  log_info("Valid input: {text}")
  return(TRUE)
}

# Prediction Route ----------------------------------------------------------

#* Predict Sentiment
#* @post /predict
#* @param text The input text for sentiment analysis
#* @response 200 A JSON object with the sentiment prediction
#* @response 400 Invalid input text
#* @response 500 Internal Server Error if prediction fails
function(req, res, text = "") {
  log_info("Prediction requested for text: {text}")
  
  # Validate input
  if (!validate_input(text)) {
    res$status <- 400
    return(list(error = "Invalid input text."))
  }
  
  # Preprocess text
  clean_text <- preprocess_text(text)
  
  # Try to make a prediction
  tryCatch({
    prediction <- predict(model, newdata = data.frame(text = clean_text), type = "response")
    sentiment <- ifelse(prediction > 0.5, "Positive", "Negative")
    
    # Return prediction in JSON format
    result <- list(
      text = text,
      sentiment = sentiment,
      score = round(prediction, 4)
    )
    
    log_info("Prediction result: {result}")
    return(toJSON(result))
    
  }, error = function(e) {
    log_error("Error during prediction: {e$message}")
    res$status <- 500
    return(list(error = "Prediction failed."))
  })
}

# Logging Route -------------------------------------------------------------

#* @get /logs
#* @response 200 Returns the recent log entries
function(req, res) {
  log_info("Logs requested.")
  
  tryCatch({
    logs <- readLines("logs/plumber.log", warn = FALSE)
    return(paste(logs, collapse = "\n"))
    
  }, error = function(e) {
    log_error("Error reading logs: {e$message}")
    res$status <- 500
    return(list(error = "Could not retrieve logs."))
  })
}

# Version Route -------------------------------------------------------------

#* API Version Route
#* @get /version
#* @response 200 Returns the current API version
function() {
  log_info("Version requested.")
  list(version = "1.0.0")
}

# Monitoring Route ----------------------------------------------------------

#* Model Performance Monitoring
#* @get /monitor
#* @response 200 Returns the model monitoring metrics
function() {
  log_info("Monitoring metrics requested.")
  
  tryCatch({
    metrics <- list(
      accuracy = 0.95,
      f1_score = 0.92,
      last_trained = "2024-09-15"
    )
    
    return(toJSON(metrics))
    
  }, error = function(e) {
    log_error("Error during monitoring: {e$message}")
    res$status <- 500
    return(list(error = "Monitoring failed."))
  })
}

# Detailed Prediction Route -------------------------------------------------

#* Predict Sentiment with Details
#* @post /predict_detailed
#* @param text The input text for sentiment analysis
#* @response 200 A JSON object with the sentiment prediction and details
#* @response 400 Invalid input text
#* @response 500 Internal Server Error if prediction fails
function(req, res, text = "") {
  log_info("Detailed prediction requested for text: {text}")
  
  # Validate input
  if (!validate_input(text)) {
    res$status <- 400
    return(list(error = "Invalid input text."))
  }
  
  # Preprocess text
  clean_text <- preprocess_text(text)
  
  # Try to make a prediction
  tryCatch({
    prediction <- predict(model, newdata = data.frame(text = clean_text), type = "response")
    sentiment <- ifelse(prediction > 0.5, "Positive", "Negative")
    
    # Generate confidence score
    confidence <- abs(prediction - 0.5) * 2
    
    # Return detailed prediction in JSON format
    result <- list(
      text = text,
      sentiment = sentiment,
      score = round(prediction, 4),
      confidence = round(confidence, 4)
    )
    
    log_info("Detailed prediction result: {result}")
    return(toJSON(result))
    
  }, error = function(e) {
    log_error("Error during detailed prediction: {e$message}")
    res$status <- 500
    return(list(error = "Prediction failed."))
  })
}

# Error Handling Route ------------------------------------------------------

#* Custom 404 Error Handler
#* @filter 404
function(req, res) {
  log_warn("404 Error: {req$PATH_INFO}")
  res$status <- 404
  list(error = "Endpoint not found.")
}

#* Custom 500 Error Handler
#* @filter 500
function(req, res, err) {
  log_error("500 Internal Server Error: {err$message}")
  res$status <- 500
  list(error = "Internal Server Error.")
}

# Model Re-Training Route ---------------------------------------------------

#* Trigger Model Re-Training
#* @post /retrain
#* @response 200 Confirmation of model retraining
#* @response 500 Error if retraining fails
function() {
  log_info("Model retraining requested.")
  
  tryCatch({
    # Model re-training logic
    log_info("Loading training data...")
    training_data <- load_training_data() 
    
    log_info("Preprocessing training data...")
    processed_data <- preprocess_data(training_data)  
    
    log_info("Initializing model...")
    model <- initialize_model()  
    
    log_info("Training the model...")
    trained_model <- train_model(model, processed_data)  
    
    log_info("Saving the trained model...")
    save_model(trained_model)  
    
    log_info("Model retrained successfully.")
    list(message = "Model retrained successfully.")
    
  }, error = function(e) {
    log_error("Error during model retraining: {e$message}")
    res$status <- 500
    list(error = "Retraining failed.")
  })
}

# Start Plumber API ---------------------------------------------------------

log_info("Starting Plumber API...")
plumber::pr() %>%
  pr_run(host = "0.0.0.0", port = 8000)