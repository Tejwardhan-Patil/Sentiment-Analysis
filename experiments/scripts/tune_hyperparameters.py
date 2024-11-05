import os
import torch
import optuna
import logging
from torch import nn, optim
from datetime import datetime
from models import lstm, bert, transformer 
from data_loader import get_dataloaders
from train import train_model, evaluate_model 

# Setup logging
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"hyperparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Hyperparameter tuning objective function
def objective(trial):
    logging.info("Starting a new trial")

    # Suggest model type
    model_type = trial.suggest_categorical('model_type', ['lstm', 'bert', 'transformer'])
    logging.info(f"Model type: {model_type}")

    # Common hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 3, 10)
    logging.info(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Num Epochs: {num_epochs}")

    # Model-specific hyperparameters
    if model_type == 'lstm':
        hidden_size = trial.suggest_int('hidden_size', 64, 256)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
        model = lstm.LSTMModel(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        logging.info(f"LSTM - Hidden Size: {hidden_size}, Num Layers: {num_layers}, Dropout: {dropout}")
    
    elif model_type == 'bert':
        # BERT fine-tuning with predefined configurations
        model = bert.BERTModel()
        logging.info("BERT model selected with default parameters")
    
    elif model_type == 'transformer':
        num_heads = trial.suggest_int('num_heads', 4, 8)
        num_layers = trial.suggest_int('num_layers', 1, 6)
        model = transformer.TransformerModel(num_heads=num_heads, num_layers=num_layers)
        logging.info(f"Transformer - Num Heads: {num_heads}, Num Layers: {num_layers}")
    
    # Move model to device (GPU/CPU)
    model.to(device)
    logging.info("Model moved to device")

    # Load data
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    logging.info("Data loaded")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logging.info("Loss function and optimizer initialized")

    # Training loop
    logging.info("Starting model training")
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_model(model, train_loader, criterion, optimizer, device)
        
        # Evaluate the model after each epoch
        val_accuracy = evaluate_model(model, val_loader, criterion, device)
        logging.info(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy:.4f}")
        
        # Save the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = f"best_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved at {best_model_path}")

    return best_accuracy

# Main function for hyperparameter optimization
def run_hyperparameter_tuning():
    logging.info("Starting hyperparameter tuning")
    
    # Optuna study for maximizing validation accuracy
    study = optuna.create_study(direction='maximize')
    
    # Optimize over multiple trials
    study.optimize(objective, n_trials=50)

    # Log the best trial
    best_trial = study.best_trial
    logging.info(f"Best trial - Accuracy: {best_trial.value}")
    logging.info(f"Best hyperparameters: {best_trial.params}")

    # Save study results
    study_path = f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(study_path, "wb") as f:
        optuna.logging.dump(study, f)
    logging.info(f"Study saved to {study_path}")

if __name__ == "__main__":
    run_hyperparameter_tuning()