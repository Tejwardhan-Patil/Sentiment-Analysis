import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from models.architectures.lstm import LSTMModel
from models.architectures.transformer import TransformerModel
from data.scripts.preprocess import preprocess_data
from utils.metrics import calculate_metrics, compute_confusion_matrix
from utils.visualization import plot_training_curves
from utils.logger import setup_logger
from utils.text_preprocessing import tokenize_text

# Configuration
MODEL_TYPE = 'bert'  # Options: 'bert', 'lstm', 'transformer'
EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3
SAVE_PATH = './checkpoints'
LOG_PATH = './logs/train.log'
BEST_MODEL_PATH = './checkpoints/best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logger
logger = setup_logger(LOG_PATH)

# Preprocess and load data
logger.info("Preprocessing data...")
train_data, val_data, test_data = preprocess_data('./data/processed/')
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=RandomSampler(train_data))
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, sampler=SequentialSampler(val_data))

# Model selection
if MODEL_TYPE == 'bert':
    logger.info("Initializing BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
elif MODEL_TYPE == 'lstm':
    logger.info("Initializing LSTM model...")
    model = LSTMModel(input_size=768, hidden_size=256, output_size=2)
elif MODEL_TYPE == 'transformer':
    logger.info("Initializing Transformer model...")
    model = TransformerModel(num_labels=2)

model = model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Early stopping and best model saving
best_val_loss = float('inf')
early_stopping_counter = 0

# Training loop
def train_model():
    global best_val_loss, early_stopping_counter
    logger.info("Starting training...")
    model.train()
    total_loss, total_accuracy = 0, 0
    training_stats = []
    for epoch in range(EPOCHS):
        epoch_loss, epoch_accuracy = 0, 0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch['input_ids'].to(DEVICE), batch['labels'].to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += (logits.argmax(dim=1) == labels).float().mean().item()

            if step % 10 == 0:
                logger.info(f"Epoch: {epoch + 1}/{EPOCHS} | Step: {step}/{len(train_loader)} | Loss: {loss.item()}")

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')
        
        # Validation
        val_loss, val_accuracy, val_metrics = evaluate_model()
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        
        # Early Stopping and Best Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info(f"Best model saved at epoch {epoch + 1}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_PATH, f'{MODEL_TYPE}_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved at: {checkpoint_path}")

# Evaluation
def evaluate_model():
    logger.info("Evaluating model...")
    model.eval()
    total_loss, total_accuracy = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['input_ids'].to(DEVICE), batch['labels'].to(DEVICE)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_accuracy += (logits.argmax(dim=1) == labels).float().mean().item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    confusion_matrix = compute_confusion_matrix(all_labels, all_preds)

    logger.info(f"Confusion Matrix: \n{confusion_matrix}")
    logger.info(f"Validation Metrics: {metrics}")

    return avg_loss, avg_accuracy, metrics

# Test model
def test_model():
    logger.info("Testing model on test dataset...")
    model.eval()
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, sampler=SequentialSampler(test_data))
    total_loss, total_accuracy = 0, 0
    all_preds, all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input_ids'].to(DEVICE), batch['labels'].to(DEVICE)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_accuracy += (logits.argmax(dim=1) == labels).float().mean().item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    metrics = calculate_metrics(all_labels, all_preds)

    logger.info(f"Test Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Test Metrics: {metrics}")

if __name__ == '__main__':
    logger.info("Starting training process...")
    train_model()
    test_model()
    
    # Plot the training curves
    plot_training_curves(SAVE_PATH)
    logger.info("Training completed.")