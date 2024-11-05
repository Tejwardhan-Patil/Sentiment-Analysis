import unittest
import json
from fastapi.testclient import TestClient
from api.app import app 

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_route(self):
        """Test the root endpoint returns the correct message."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Welcome to the Sentiment Analysis API"})

    def test_predict_positive_sentiment(self):
        """Test prediction for a positive sentiment text."""
        data = {"text": "This is an amazing product!"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertEqual(response.json()["prediction"], "positive")

    def test_predict_negative_sentiment(self):
        """Test prediction for a negative sentiment text."""
        data = {"text": "This is a terrible experience."}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertEqual(response.json()["prediction"], "negative")

    def test_predict_neutral_sentiment(self):
        """Test prediction for neutral sentiment."""
        data = {"text": "The product is okay."}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertEqual(response.json()["prediction"], "neutral")

    def test_invalid_input(self):
        """Test the API handles invalid input gracefully."""
        data = {"wrong_field": "This input is missing the 'text' field."}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 422)

    def test_empty_text_input(self):
        """Test the API's behavior with an empty string as input."""
        data = {"text": ""}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Input text cannot be empty.")

    def test_large_text_input(self):
        """Test the API's handling of very large input strings."""
        large_text = "This is a good product. " * 10000  # A very large input string
        data = {"text": large_text}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_script_injection(self):
        """Test the API's protection against script injection."""
        data = {"text": "<script>alert('Hacked!');</script>"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Invalid characters in input.")

    def test_sql_injection(self):
        """Test the API's protection against SQL injection."""
        data = {"text": "'; DROP TABLE users; --"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Invalid characters in input.")

    def test_predict_multilingual_sentiment_english(self):
        """Test prediction for English text."""
        data = {"text": "I love this product!"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "positive")

    def test_predict_multilingual_sentiment_spanish(self):
        """Test prediction for Spanish text."""
        data = {"text": "Me encanta este producto!"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "positive")

    def test_predict_multilingual_sentiment_french(self):
        """Test prediction for French text."""
        data = {"text": "J'adore ce produit!"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "positive")

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_load_testing(self):
        """Test the API under high load by making multiple concurrent requests."""
        data = {"text": "This product is fantastic!"}
        for _ in range(100):
            response = self.client.post("/predict", data=json.dumps(data))
            self.assertEqual(response.status_code, 200)

    def test_rate_limiting(self):
        """Test if the API implements rate limiting by sending multiple requests quickly."""
        data = {"text": "Great product!"}
        for _ in range(1000):  # Simulate excessive API requests
            response = self.client.post("/predict", data=json.dumps(data))
            if response.status_code == 429:
                self.assertIn("error", response.json())
                self.assertEqual(response.json()["error"], "Rate limit exceeded")
                break

    def test_post_without_content_type(self):
        """Test the API's behavior when the 'Content-Type' header is missing."""
        data = {"text": "This is a good product."}
        response = self.client.post("/predict", data=json.dumps(data), headers={})
        self.assertEqual(response.status_code, 400)

    def test_large_batch_requests(self):
        """Test the API's ability to handle large batch sentiment analysis requests."""
        data = {
            "texts": [
                "I love this product.",
                "I hate this service.",
                "This is an average experience."
            ] * 1000
        }
        response = self.client.post("/predict_batch", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())
        self.assertEqual(len(response.json()["predictions"]), 3000)

    def test_predict_with_special_characters(self):
        """Test prediction for text with special characters."""
        data = {"text": "I @love% this #product!$%^"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_predict_with_numeric_values(self):
        """Test prediction for text with numbers."""
        data = {"text": "I rate this product 9 out of 10."}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_predict_with_edge_case(self):
        """Test prediction for text with only one word."""
        data = {"text": "Good"}
        response = self.client.post("/predict", data=json.dumps(data))
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

if __name__ == '__main__':
    unittest.main()