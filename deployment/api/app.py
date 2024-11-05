from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from inference import run_inference, load_model
from models.architectures.lstm import LSTMModel
from models.architectures.bert import BERTModel
from models.architectures.transformer import TransformerModel

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Define models to be loaded at startup
lstm_model = None
bert_model = None
transformer_model = None

# Model selection and configuration
class ModelChoice(BaseModel):
    model_name: str  # "lstm", "bert", "transformer"
    temperature: Optional[float] = 1.0  # For adjusting output sensitivity

# Request schema
class SentimentRequest(BaseModel):
    text: str
    model_choice: ModelChoice

# Response schema
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    model_used: str
    tokens: Optional[List[str]] = None
    attention_scores: Optional[List[float]] = None
    time_taken: Optional[float] = None

# Load models at startup
@app.on_event("startup")
async def load_models():
    global lstm_model, bert_model, transformer_model
    lstm_model = LSTMModel()
    bert_model = BERTModel()
    transformer_model = TransformerModel()
    logger.info("Models loaded successfully")

# Get active model based on the user's choice
def get_model(model_name: str):
    if model_name == "lstm":
        return lstm_model
    elif model_name == "bert":
        return bert_model
    elif model_name == "transformer":
        return transformer_model
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

# Error handling middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Main API route for sentiment analysis
@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    logger.info(f"Received request for sentiment analysis: {request.text}")
    
    # Select the model
    model = get_model(request.model_choice.model_name)
    
    # Run inference
    try:
        sentiment, confidence, tokens, attention_scores, time_taken = run_inference(
            text=request.text, model=model, temperature=request.model_choice.temperature
        )
        response = SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            model_used=request.model_choice.model_name,
            tokens=tokens,
            attention_scores=attention_scores,
            time_taken=time_taken
        )
        logger.info(f"Sentiment analysis completed: {response.sentiment}")
        return response
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing sentiment analysis")

# Route to get available models
@app.get("/models", response_model=List[str])
async def get_models():
    return ["lstm", "bert", "transformer"]

# Route for health check
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}

# Additional utility functions
async def log_request_data(request: SentimentRequest):
    logger.info(f"Logging request data: {request.text}, Model: {request.model_choice.model_name}")

# Error handling
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return {"error": exc.detail}

# Main function to run the application
if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")