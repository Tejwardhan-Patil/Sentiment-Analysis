import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define Dataset class for loading data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define the LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_preds = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_preds += torch.sum(preds == labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_preds.double() / len(dataloader.dataset)
    return avg_loss, accuracy

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_preds += torch.sum(preds == labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_preds.double() / len(dataloader.dataset)
    return avg_loss, accuracy

# Define utility function to tokenize and split data
def prepare_data(texts, labels, tokenizer, max_len, test_size=0.2):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)
    
    return train_dataset, test_dataset

# Main execution logic
def main(train_texts, train_labels, test_texts, test_labels, tokenizer, max_len=128, batch_size=32, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare datasets and dataloaders
    train_dataset, test_dataset = prepare_data(train_texts, train_labels, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define model parameters
    vocab_size = tokenizer.vocab_size
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 2
    num_layers = 2
    dropout = 0.3

    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}')
        print(f'Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc:.3f}')
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}')
        print(f'Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc:.3f}')
        
        # Early stopping mechanism
        if epoch == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print("Model saved!")

    # Loading the best model for final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f'Final Test Loss: {final_test_loss:.3f}, Final Test Accuracy: {final_test_acc:.3f}')

# Additional utility functions
def load_data(file_path):
    """
    Load text and labels from a given file.
    Assumes the file has two columns: text and label.
    """
    import pandas as pd
    data = pd.read_csv(file_path)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    return texts, labels

def preprocess_text(text):
    """
    Simple text preprocessing including:
    - Lowercasing
    - Removing special characters
    - Removing extra spaces
    """
    import re
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def tokenize_data(texts, tokenizer, max_len):
    """
    Tokenizes the text data using the provided tokenizer.
    """
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

# Model Evaluation: Function to print confusion matrix and classification report
def evaluate_performance(model, dataloader, device):
    from sklearn.metrics import confusion_matrix, classification_report
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    cm = confusion_matrix(true_labels, pred_labels)
    print(f"Confusion Matrix:\n{cm}")

    cr = classification_report(true_labels, pred_labels)
    print(f"Classification Report:\n{cr}")

# Hyperparameter tuning functionality
def tune_hyperparameters(train_texts, train_labels, test_texts, test_labels, tokenizer, max_len, learning_rates, hidden_dims, batch_sizes, num_epochs=5):
    best_accuracy = 0.0
    best_params = {}

    for lr in learning_rates:
        for hidden_dim in hidden_dims:
            for batch_size in batch_sizes:
                print(f"Training with lr={lr}, hidden_dim={hidden_dim}, batch_size={batch_size}")

                # Re-initialize the model with different hyperparameters
                model = SentimentLSTM(tokenizer.vocab_size, 128, hidden_dim, 2, 2, 0.3)
                model = model.to(device)

                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
                test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)

                for epoch in range(num_epochs):
                    train_model(model, train_loader, optimizer, criterion, device)
                    val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        best_params = {
                            'learning_rate': lr,
                            'hidden_dim': hidden_dim,
                            'batch_size': batch_size
                        }
                        torch.save(model.state_dict(), 'best_tuned_model.pt')

    print(f"Best accuracy: {best_accuracy}")
    print(f"Best parameters: {best_params}")

    return best_params

# Load and preprocess data
def prepare_and_load_data(file_path, tokenizer, max_len):
    texts, labels = load_data(file_path)
    processed_texts = [preprocess_text(text) for text in texts]
    return tokenize_data(processed_texts, tokenizer, max_len), labels

# Batch inference for new unseen data
def batch_inference(model, new_texts, tokenizer, max_len, batch_size=32):
    model.eval()
    all_preds = []

    dataset = TextDataset(new_texts, [0]*len(new_texts), tokenizer, max_len) 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds

# Function to test batch inference
def test_inference():
    test_texts = [
        "This is a great movie!",
        "I hated the plot and the acting was bad.",
        "Absolutely wonderful experience!",
        "Not worth the time, very disappointing."
    ]
    tokenizer = ... 
    max_len = 128

    model = SentimentLSTM(tokenizer.vocab_size, 128, 256, 2, 2, 0.3)
    model.load_state_dict(torch.load('best_model.pt'))
    model = model.to(device)

    predictions = batch_inference(model, test_texts, tokenizer, max_len)
    print(f"Predictions: {predictions}")

# Visualization of training and validation metrics
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Function for model deployment with a simple REST API using FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Define a Pydantic model for input validation
class SentimentRequest(BaseModel):
    text: str

# Initialize model and tokenizer outside the request scope
def init_model_and_tokenizer(tokenizer_path, model_path, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=2, num_layers=2, dropout=0.3):
    # Load tokenizer and model
    tokenizer = ...  # Loading from tokenizer_path
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, tokenizer

# Load the model and tokenizer at the start
model, tokenizer = init_model_and_tokenizer("tokenizer_path", "best_model.pt", vocab_size=30522)  # Vocab size

# Set the device: Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the API route for sentiment prediction
@app.post("/predict/")
async def predict_sentiment(request: SentimentRequest):
    try:
        text = request.text
        preprocessed_text = preprocess_text(text)
        inputs = tokenize_data([preprocessed_text], tokenizer, max_len=128)
        
        # Inference
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            _, prediction = torch.max(outputs, dim=1)

        sentiment = "positive" if prediction.item() == 1 else "negative"
        return {"sentiment": sentiment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to run the FastAPI server
def run_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main execution point for deployment
if __name__ == "__main__":
    run_server()

# Function to log predictions
import logging

def setup_logging():
    logging.basicConfig(filename='sentiment_analysis.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_prediction(text, sentiment):
    logging.info(f"Text: {text}, Predicted Sentiment: {sentiment}")

# Advanced Evaluation with ROC Curve and AUC
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(true_labels, pred_probs):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Function to generate prediction probabilities
def generate_pred_probs(model, dataloader, device):
    model.eval()
    pred_probs = []
    true_labels = []
    import numpy as np
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_probs.append(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    pred_probs = np.vstack(pred_probs)
    return np.array(true_labels), pred_probs

# Function to train and plot learning curves dynamically during training
def train_with_dynamic_plot(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training step
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation step
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Plotting dynamically
        clear_output(wait=True)
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, val_losses, label="Val Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, label="Train Accuracy")
        plt.plot(epochs_range, val_accs, label="Val Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.show()

# Function for saving the trained model and tokenizer for reuse
def save_model_and_tokenizer(model, tokenizer, model_save_path, tokenizer_save_path):
    # Save model state
    torch.save(model.state_dict(), model_save_path)

    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_save_path)

    print(f"Model saved to {model_save_path}")
    print(f"Tokenizer saved to {tokenizer_save_path}")

# Usage of saving a model and tokenizer
if __name__ == "__main__":
    model_save_path = "sentiment_lstm_model.pt"
    tokenizer_save_path = "tokenizer/"

    model, tokenizer = init_model_and_tokenizer("tokenizer_path", "best_model.pt", vocab_size=30522)
    save_model_and_tokenizer(model, tokenizer, model_save_path, tokenizer_save_path)