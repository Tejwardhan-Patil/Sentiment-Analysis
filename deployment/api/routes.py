from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, ValidationError
from inference import run_inference 
from typing import Optional
import logging
import time

router = APIRouter()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_analysis")

# Input data model with extended validation
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Input text to analyze.")

# Output data model with additional metadata
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    processed_text: Optional[str] = None
    timestamp: Optional[str] = None

# Utility to preprocess text
def preprocess_text(text: str) -> str:
    logger.info("Starting text preprocessing.")
    # Perform text preprocessing (strip, lowercasing, etc.)
    processed_text = text.strip().lower()
    logger.info(f"Text after preprocessing: {processed_text}")
    return processed_text

# Utility to log request metadata
def log_request_metadata(request: Request, sentiment_request: SentimentRequest):
    logger.info(f"Request metadata - client host: {request.client.host}")
    logger.info(f"User-agent: {request.headers.get('user-agent')}")
    logger.info(f"Request payload: {sentiment_request}")

# Utility to get current timestamp
def get_current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# Advanced error handling
class InferenceError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: Request, sentiment_request: SentimentRequest):
    """
    API route for analyzing sentiment of the input text.
    """
    try:
        # Log request metadata
        log_request_metadata(request, sentiment_request)
        
        # Validate input length
        if len(sentiment_request.text) > 1000:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text length exceeds maximum limit.")
        
        # Preprocess the text before inference
        processed_text = preprocess_text(sentiment_request.text)
        
        # Log before calling inference
        logger.info("Calling inference engine...")
        
        # Run sentiment analysis (this function returns sentiment and confidence score)
        sentiment, confidence = run_inference(processed_text)
        
        if not sentiment or confidence is None:
            raise InferenceError("Inference returned invalid results.")

        # Build response object
        response = SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            processed_text=processed_text,
            timestamp=get_current_timestamp()
        )

        logger.info(f"Sentiment analysis result: {response}")
        
        return response

    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid input format.")

    except InferenceError as ie:
        logger.error(f"Inference error: {str(ie)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ie))

    except Exception as e:
        logger.error(f"Unexpected error during sentiment analysis: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sentiment analysis failed due to an unexpected error.")

# Route to get API status
@router.get("/status")
async def get_status():
    """
    Simple API route to check the status of the API.
    """
    logger.info("Status check request received.")
    return {"status": "API is running", "timestamp": get_current_timestamp()}

# Route to handle a simple health check
@router.get("/healthcheck")
async def healthcheck():
    """
    Health check route to ensure the API is live.
    """
    logger.info("Health check request received.")
    return {"status": "Healthy", "timestamp": get_current_timestamp()}

# Detailed logging for input validation errors
def handle_input_validation_error(error: ValidationError):
    logger.error(f"Input validation error: {error}")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input.")

# Advanced logging setup for monitoring
def setup_advanced_logging():
    file_handler = logging.FileHandler("api_logs.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Advanced logging setup complete.")

# Additional route for debugging request headers
@router.get("/debug")
async def debug_headers(request: Request):
    """
    Debug route to view request headers and metadata.
    """
    headers = dict(request.headers)
    logger.debug(f"Request headers: {headers}")
    return {"headers": headers}

# Route for testing with random text (for internal use)
@router.get("/test")
async def test_analysis():
    """
    Test route to analyze a random text for sentiment.
    """
    test_text = "This is a great day!"
    logger.info(f"Running test analysis for text: {test_text}")
    
    processed_text = preprocess_text(test_text)
    sentiment, confidence = run_inference(processed_text)

    response = SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        processed_text=processed_text,
        timestamp=get_current_timestamp()
    )
    
    logger.info(f"Test analysis result: {response}")
    return response

# Route to log recent analysis history
analysis_history = []

@router.get("/history")
async def get_analysis_history():
    """
    Retrieve history of recent sentiment analysis requests.
    """
    logger.info("Retrieving analysis history.")
    return {"history": analysis_history}

@router.post("/analyze_v2", response_model=SentimentResponse)
async def analyze_sentiment_v2(request: Request, sentiment_request: SentimentRequest):
    """
    API route for analyzing sentiment of the input text (alternative version).
    """
    try:
        log_request_metadata(request, sentiment_request)

        processed_text = preprocess_text(sentiment_request.text)
        sentiment, confidence = run_inference(processed_text)

        response = SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            processed_text=processed_text,
            timestamp=get_current_timestamp()
        )

        # Append to history
        analysis_history.append({"text": sentiment_request.text, "sentiment": sentiment, "timestamp": response.timestamp})

        logger.info(f"Sentiment analysis (v2) result: {response}")
        return response

    except Exception as e:
        logger.error(f"Error in analyze_sentiment_v2: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during sentiment analysis (v2)")

# Route to reset analysis history
@router.delete("/history/reset")
async def reset_history():
    """
    Route to reset the sentiment analysis history.
    """
    global analysis_history
    analysis_history = []
    logger.info("Analysis history has been reset.")
    return {"message": "Analysis history reset successfully."}

# Start advanced logging on import
setup_advanced_logging()