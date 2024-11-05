import torch
from transformers import BertTokenizer, BertForSequenceClassification
from architectures.lstm import LSTMModel
from architectures.bert import BERTModel
from architectures.transformer import TransformerModel
from custom.custom_lstm import CustomLSTMModel
from custom.custom_transformer import CustomTransformerModel
from custom.custom_bert import CustomBERTModel
from custom.hybrid_model import HybridModel
from custom.attention_lstm import AttentionLSTMModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Inference:
    def __init__(self, model_name, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_name = model_name.lower()

        # Initialize model
        self.model = self._load_model(model_name)
        self._load_weights(model_path)

        # Initialize tokenizer for transformer-based models
        if 'bert' in self.model_name or 'transformer' in self.model_name:
            logger.info("Initializing tokenizer for BERT or Transformer model.")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model {self.model_name} loaded successfully.")

    def _load_model(self, model_name):
        """Load model based on the model name."""
        if model_name == 'lstm':
            logger.info("Loading LSTM model.")
            return LSTMModel()
        elif model_name == 'bert':
            logger.info("Loading BERT model.")
            return BERTModel()
        elif model_name == 'transformer':
            logger.info("Loading Transformer model.")
            return TransformerModel()
        elif model_name == 'custom_lstm':
            logger.info("Loading Custom LSTM model.")
            return CustomLSTMModel()
        elif model_name == 'custom_transformer':
            logger.info("Loading Custom Transformer model.")
            return CustomTransformerModel()
        elif model_name == 'custom_bert':
            logger.info("Loading Custom BERT model.")
            return CustomBERTModel()
        elif model_name == 'hybrid':
            logger.info("Loading Hybrid model (LSTM + Transformer).")
            return HybridModel()
        elif model_name == 'attention_lstm':
            logger.info("Loading Attention LSTM model.")
            return AttentionLSTMModel()
        else:
            logger.error(f"Model {model_name} not recognized.")
            raise ValueError(f"Model {model_name} not recognized.")

    def _load_weights(self, model_path):
        """Load the model weights from the specified path."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model weights loaded from {model_path}.")
        except Exception as e:
            logger.error(f"Error loading model weights from {model_path}: {e}")
            raise

    def preprocess(self, text):
        """Preprocess text input for model inference."""
        if isinstance(text, str):
            logger.info("Preprocessing input text.")
            if 'bert' in self.model_name or 'transformer' in self.model_name:
                inputs = self.tokenizer(
                    text, return_tensors='pt', padding=True, truncation=True, max_length=128
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                return inputs
            else:
                logger.info("Running custom preprocessing for non-transformer model.")
                tokens = self.custom_preprocess(text)
                inputs = torch.tensor(tokens).unsqueeze(0).to(self.device)
                return inputs
        else:
            logger.error("Input is not a valid string.")
            raise ValueError("Input must be a string.")

    def custom_preprocess(self, text):
        """Custom preprocessing for non-transformer models."""
        logger.info("Tokenizing text for LSTM-based models.")
        tokens = [word for word in text.lower().split()]
        # Padding for simplicity
        max_length = 50
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += ['<PAD>'] * (max_length - len(tokens))
        logger.info(f"Tokenized input: {tokens}")
        return tokens

    def predict(self, text):
        """Run inference on the input text."""
        inputs = self.preprocess(text)

        try:
            with torch.no_grad():
                logger.info("Running inference.")
                if 'bert' in self.model_name or 'transformer' in self.model_name:
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                else:
                    outputs = self.model(inputs)
                    logits = outputs

            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()

            logger.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
            return prediction, probabilities
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def batch_predict(self, texts):
        """Run inference on a batch of texts."""
        if not isinstance(texts, list):
            logger.error("Input must be a list of strings.")
            raise ValueError("Input must be a list of strings.")

        logger.info(f"Processing batch of {len(texts)} texts.")
        predictions = []
        probabilities_list = []

        for text in texts:
            try:
                prediction, probabilities = self.predict(text)
                predictions.append(prediction)
                probabilities_list.append(probabilities)
            except Exception as e:
                logger.error(f"Failed to process text: {text}. Error: {e}")
                predictions.append(None)
                probabilities_list.append(None)

        return predictions, probabilities_list

    def explain_prediction(self, text):
        """Explain the prediction for the given text."""
        logger.info("Generating explanation for the prediction.")
        prediction, probabilities = self.predict(text)
        explanation = {
            'text': text,
            'prediction': prediction,
            'probability_distribution': probabilities.cpu().numpy().tolist(),
        }
        logger.info(f"Explanation: {explanation}")
        return explanation

    def save_predictions(self, texts, predictions, file_path):
        """Save the predictions to a file."""
        logger.info(f"Saving predictions to {file_path}.")
        try:
            with open(file_path, 'w') as f:
                for text, prediction in zip(texts, predictions):
                    f.write(f"Text: {text}\nPrediction: {prediction}\n\n")
            logger.info("Predictions saved successfully.")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise

    def load_model(self, model_path):
        """Reload the model with new weights."""
        logger.info(f"Reloading model from {model_path}.")
        self._load_weights(model_path)
        logger.info("Model reloaded successfully.")


# Usage:
# inference = Inference(model_name='bert', model_path='trained_model.pt')
# sentiment, confidence = inference.predict("This is a great product!")
# predictions, probabilities = inference.batch_predict(["I love this!", "I hate it."])
# explanation = inference.explain_prediction("This is a fantastic experience.")
# inference.save_predictions(["Text 1", "Text 2"], predictions, "predictions.txt")