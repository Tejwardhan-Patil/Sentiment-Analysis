library(logging)
library(jsonlite)

# Define logging configuration
basicConfig()

# Set log file path
log_file <- "logs/model_log.log"

# Function to initialize the logger
init_logger <- function(log_file) {
  addHandler(writeToFile, file = log_file, level = 'INFO')
}

# Function to log model predictions
log_prediction <- function(input_data, prediction, model_name) {
  log_info <- list(
    timestamp = Sys.time(),
    model = model_name,
    input = input_data,
    prediction = prediction
  )
  
  log_info_json <- toJSON(log_info, auto_unbox = TRUE)
  loginfo(paste("Prediction Log:", log_info_json))
}

# Function to log errors
log_error <- function(error_message, error_details = NULL) {
  error_log <- list(
    timestamp = Sys.time(),
    error_message = error_message,
    error_details = error_details
  )
  
  error_log_json <- toJSON(error_log, auto_unbox = TRUE)
  logerror(paste("Error Log:", error_log_json))
}

# Initializing the logger
init_logger(log_file)

# Logging a prediction
log_prediction("sample input data", "positive sentiment", "Sentiment_Model_1")

# Logging an error
log_error("Model inference failed", "Details about the error can be captured here")