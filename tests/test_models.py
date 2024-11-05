import unittest
import torch
import torch.nn as nn
from models.architectures.lstm import LSTMModel
from models.architectures.bert import BERTModel
from models.architectures.transformer import TransformerModel
from models.custom.custom_lstm import CustomLSTMModel
from models.custom.custom_transformer import CustomTransformerModel
from models.custom.custom_bert import CustomBERTModel

class TestModels(unittest.TestCase):

    def setUp(self):
        # Set up input data for the tests (batch of 32 samples, sequence length of 50, embedding size of 300)
        self.input_data = torch.randn(32, 50, 300)
        self.labels = torch.randint(0, 2, (32,))  # Binary classification

    def test_lstm_model(self):
        # Test LSTM model with normal configuration
        model = LSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "LSTM output shape mismatch")

    def test_lstm_model_different_config(self):
        # Test LSTM model with different hidden size
        model = LSTMModel(input_size=300, hidden_size=256, output_size=2, num_layers=3)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "LSTM output shape mismatch with different hidden size")

    def test_lstm_training_multiple_steps(self):
        # Test LSTM model training over multiple steps
        model = LSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for step in range(5):  # Train for 5 steps
            outputs = model(self.input_data)
            loss = criterion(outputs, self.labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertLess(loss.item(), 2.0, f"Training loss at step {step} should be reasonable")

    def test_lstm_edge_case_small_batch(self):
        # Test LSTM model with a small batch size
        small_input_data = torch.randn(2, 50, 300)
        small_labels = torch.randint(0, 2, (2,))
        model = LSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2)
        output = model(small_input_data)
        self.assertEqual(output.shape, (2, 2), "LSTM output shape mismatch for small batch size")

    def test_bert_model(self):
        # Test BERT model with normal configuration
        model = BERTModel(pretrained_model_name='bert-base-uncased', num_labels=2)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "BERT output shape mismatch")

    def test_bert_model_longer_sequence(self):
        # Test BERT model with longer sequence length
        longer_input_data = torch.randn(32, 100, 300)
        model = BERTModel(pretrained_model_name='bert-base-uncased', num_labels=2)
        output = model(longer_input_data)
        self.assertEqual(output.shape, (32, 2), "BERT output shape mismatch for longer sequence length")

    def test_bert_training_multiple_steps(self):
        # Test BERT model training over multiple steps
        model = BERTModel(pretrained_model_name='bert-base-uncased', num_labels=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for step in range(3):  # Train for 3 steps
            outputs = model(self.input_data)
            loss = criterion(outputs, self.labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertLess(loss.item(), 1.5, f"BERT training loss at step {step} should be reasonable")

    def test_transformer_model(self):
        # Test Transformer model with normal configuration
        model = TransformerModel(input_size=300, num_heads=8, num_layers=6, output_size=2)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "Transformer output shape mismatch")

    def test_transformer_different_heads(self):
        # Test Transformer model with different number of heads
        model = TransformerModel(input_size=300, num_heads=4, num_layers=4, output_size=2)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "Transformer output shape mismatch with different heads")

    def test_transformer_edge_case_short_sequence(self):
        # Test Transformer model with short sequence length
        short_input_data = torch.randn(32, 10, 300)
        model = TransformerModel(input_size=300, num_heads=8, num_layers=6, output_size=2)
        output = model(short_input_data)
        self.assertEqual(output.shape, (32, 2), "Transformer output shape mismatch for short sequence length")

    def test_custom_lstm_model(self):
        # Test custom LSTM model with attention mechanism
        model = CustomLSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2, attention=True)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "Custom LSTM output shape mismatch with attention")

    def test_custom_lstm_training(self):
        # Test custom LSTM model training over multiple steps
        model = CustomLSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2, attention=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for step in range(5):  # Train for 5 steps
            outputs = model(self.input_data)
            loss = criterion(outputs, self.labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertLess(loss.item(), 2.0, f"Custom LSTM training loss at step {step} should be reasonable")

    def test_custom_transformer_model(self):
        # Test custom Transformer model with dropout
        model = CustomTransformerModel(input_size=300, num_heads=8, num_layers=6, output_size=2, dropout=0.1)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "Custom Transformer output shape mismatch with dropout")

    def test_custom_bert_model(self):
        # Test custom BERT model with attention
        model = CustomBERTModel(pretrained_model_name='bert-base-uncased', num_labels=2, attention=True)
        output = model(self.input_data)
        self.assertEqual(output.shape, (32, 2), "Custom BERT output shape mismatch with attention")

    def test_failure_case(self):
        # Test a failure case for incorrect input dimensions
        model = LSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2)
        incorrect_input_data = torch.randn(32, 100)  # Incorrect shape (should be 32, 50, 300)
        with self.assertRaises(RuntimeError, msg="Expected failure due to incorrect input shape"):
            model(incorrect_input_data)

    def test_custom_transformer_short_sequence(self):
        # Test custom Transformer model with short sequence length
        short_input_data = torch.randn(32, 10, 300)
        model = CustomTransformerModel(input_size=300, num_heads=8, num_layers=6, output_size=2, dropout=0.1)
        output = model(short_input_data)
        self.assertEqual(output.shape, (32, 2), "Custom Transformer output shape mismatch for short sequence")

    def test_custom_lstm_edge_case_single_sample(self):
        # Test custom LSTM model with a single sample in the batch
        single_input_data = torch.randn(1, 50, 300)
        model = CustomLSTMModel(input_size=300, hidden_size=128, output_size=2, num_layers=2, attention=True)
        output = model(single_input_data)
        self.assertEqual(output.shape, (1, 2), "Custom LSTM output shape mismatch for single sample")

if __name__ == '__main__':
    unittest.main()