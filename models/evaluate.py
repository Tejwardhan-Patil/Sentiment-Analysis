import os
import torch
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from utils.metrics import custom_accuracy, custom_f1_score, custom_precision, custom_recall 
from utils.visualization import plot_confusion_matrix, plot_classification_report, plot_loss_curve 
from utils.logger import setup_logger
from custom_lstm import CustomLSTM
from custom_bert import CustomBERTModel
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Setup logger
logger = setup_logger('model_evaluation', 'logs/evaluate.log')

def load_model(model_path, model_class, device='cpu'):
    """
    Load the model from the specified path.
    """
    logger.info(f"Loading model from {model_path}")
    try:
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def evaluate_model(model, dataloader, criterion, device='cpu'):
    """
    Evaluate the model on the provided dataloader and criterion.
    """
    logger.info("Starting evaluation of the model.")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            logger.info(f"Batch {i + 1}: Loss = {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Average loss: {avg_loss}")
    return avg_loss, all_preds, all_labels

def calculate_metrics(labels, preds):
    """
    Calculate evaluation metrics from true labels and predicted labels.
    """
    logger.info("Calculating evaluation metrics.")
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    conf_matrix = confusion_matrix(labels, preds)
    report = classification_report(labels, preds)

    custom_acc = custom_accuracy(labels, preds)
    custom_prec = custom_precision(labels, preds)
    custom_rec = custom_recall(labels, preds)
    custom_f1 = custom_f1_score(labels, preds)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Custom Accuracy: {custom_acc}")
    logger.info(f"Custom Precision: {custom_prec}")
    logger.info(f"Custom Recall: {custom_rec}")
    logger.info(f"Custom F1 Score: {custom_f1}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "custom_accuracy": custom_acc,
        "custom_precision": custom_prec,
        "custom_recall": custom_rec,
        "custom_f1_score": custom_f1,
        "conf_matrix": conf_matrix,
        "report": report
    }

def evaluate_on_dataset(model_path, model_class, dataloader, criterion, device='cpu'):
    """
    Load a model, evaluate it on the dataset, and calculate metrics.
    """
    model = load_model(model_path, model_class, device)
    loss, preds, labels = evaluate_model(model, dataloader, criterion, device)
    
    metrics = calculate_metrics(labels, preds)

    logger.info(f"Model evaluation completed for {model_path}. Loss: {loss}")
    
    # Visualizations
    logger.info("Generating visualizations.")
    plot_confusion_matrix(metrics["conf_matrix"], labels=set(labels))
    plot_classification_report(metrics["report"])

    return metrics

def log_results(model_path, results):
    """
    Log the evaluation results for the model.
    """
    logger.info(f"Evaluation results for {model_path}:")
    logger.info(f"Accuracy: {results['accuracy']}")
    logger.info(f"Precision: {results['precision']}")
    logger.info(f"Recall: {results['recall']}")
    logger.info(f"F1 Score: {results['f1_score']}")
    logger.info(f"Custom Accuracy: {results['custom_accuracy']}")
    logger.info(f"Custom Precision: {results['custom_precision']}")
    logger.info(f"Custom Recall: {results['custom_recall']}")
    logger.info(f"Custom F1 Score: {results['custom_f1_score']}")

def main(model_paths, model_classes, dataloaders, criterion, device='cpu'):
    """
    Evaluate multiple models and log the results.
    """
    results = {}
    for model_path, model_class, dataloader in zip(model_paths, model_classes, dataloaders):
        logger.info(f"Evaluating model: {model_path}")
        metrics = evaluate_on_dataset(model_path, model_class, dataloader, criterion, device)
        log_results(model_path, metrics)
        results[model_path] = metrics

    return results

if __name__ == "__main__":
    # Load split datasets from processed files (CSV files)
    validation_data = pd.read_csv('data/processed/validation_data.csv')  # Path to validation data
    test_data = pd.read_csv('data/processed/test_data.csv')  # Path to test data

    # Convert data to tensors
    validation_inputs = torch.tensor(validation_data['X'].values, dtype=torch.float32)  # Features column 'X'
    validation_labels = torch.tensor(validation_data['y'].values, dtype=torch.long)  # Labels column 'y'
    
    test_inputs = torch.tensor(test_data['X'].values, dtype=torch.float32)  # Features column 'X'
    test_labels = torch.tensor(test_data['y'].values, dtype=torch.long)  # Labels column 'y'

    # Create PyTorch datasets
    validation_dataset = TensorDataset(validation_inputs, validation_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    # Create PyTorch dataloaders
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Usage
    model_paths = ['/lstm_model.pth', '/bert_model.pth']
    model_classes = [CustomLSTM, CustomBERTModel]
    dataloaders = [validation_dataloader, test_dataloader]
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Starting model evaluation process.")
    results = main(model_paths, model_classes, dataloaders, criterion, device)
    logger.info("Model evaluation process completed.")

    print("Evaluation complete.")
    print(results)