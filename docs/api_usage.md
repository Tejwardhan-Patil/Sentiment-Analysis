# API Usage

This document provides guidelines on how to interact with the deployed sentiment analysis API. The API accepts text data and returns a sentiment classification in a structured format.

## 1. API Endpoints

### 1.1 `/predict`

- **Method**: POST
- **Description**: Submits a text string for sentiment analysis.
- **Request**:
  - `text`: A string containing the input text, such as "I love this product!"
- **Response**:
  - `sentiment`: The predicted sentiment, such as "positive."
  - `confidence`: The model's confidence level, for example, 0.95.

### 1.2 `/batch_predict`

- **Method**: POST
- **Description**: Submits multiple text strings for batch sentiment analysis.
- **Request**:
  - `texts`: A list of strings containing input text, such as:
    - "This is amazing!"
    - "I'm not happy with the service."
- **Response**:
  - `results`: A list of sentiment predictions and corresponding confidence levels, such as:
    - Sentiment: "positive", Confidence: 0.98
    - Sentiment: "negative", Confidence: 0.85

## 2. Authentication

- If authentication is required, include a token in the request header:
  - `Authorization: Bearer <token>`

## 3. Error Handling

- If an error occurs, the API will return a standardized error response:
  - `error`: Describes the error, such as "Invalid input."
  - `message`: Additional details, such as "Text field cannot be empty."
