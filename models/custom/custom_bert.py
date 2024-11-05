import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class CustomBERTModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', dropout=0.3, output_size=2):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = bert_output[1]  # Use the [CLS] token representation
        dropped_output = self.dropout(pooled_output)
        activated_output = self.relu(dropped_output)
        output = self.fc(activated_output)
        return output

def load_tokenizer(pretrained_model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    return tokenizer

def prepare_input(text_list, tokenizer, max_length=128):
    encoding = tokenizer(text_list, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids']

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids, attention_mask, token_type_ids = prepare_input([text], self.tokenizer, self.max_length)
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'token_type_ids': token_type_ids.squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

        accuracy = correct_predictions.double() / total_predictions
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss, accuracy

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        accuracy = correct_predictions.double() / total_predictions
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, accuracy

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

def create_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer

def create_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

def train_custom_bert_model(model, train_data, val_data, tokenizer, device, num_epochs=3, batch_size=32, max_length=128):
    train_dataset = SentimentDataset(train_data['texts'], train_data['labels'], tokenizer, max_length)
    val_dataset = SentimentDataset(val_data['texts'], val_data['labels'], tokenizer, max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = create_criterion()
    optimizer = create_optimizer(model)

    trainer = SentimentTrainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs)

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = CustomBERTModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_data = {
        'texts': ["I love this!", "This is terrible..."],
        'labels': [1, 0]
    }
    val_data = {
        'texts': ["Amazing product!", "Worst purchase ever."],
        'labels': [1, 0]
    }

    train_custom_bert_model(model, train_data, val_data, tokenizer, device)

import os
import logging
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in self.data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = correct_predictions.double() / total_predictions
        avg_loss = total_loss / len(self.data_loader)

        return avg_loss, accuracy, all_labels, all_preds

    def generate_classification_report(self, all_labels, all_preds, target_names):
        report = classification_report(all_labels, all_preds, target_names=target_names)
        print(report)
        return report

    def generate_confusion_matrix(self, all_labels, all_preds):
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
        return cm

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def create_logging(log_dir="logs", log_file="training.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=os.path.join(log_dir, log_file), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging is set up.")

def log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc):
    logging.info(f'Epoch {epoch + 1}')
    logging.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    logging.info(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

class AdvancedSentimentTrainer(SentimentTrainer):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, target_names):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device)
        self.target_names = target_names
        self.evaluator = ModelEvaluator(model, val_loader, criterion, device)

    def train(self, num_epochs, log_every=1, save_model_path=None):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, all_labels, all_preds = self.evaluator.evaluate()

            if epoch % log_every == 0:
                log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc)
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

            # Classification report and confusion matrix at each log step
            self.evaluator.generate_classification_report(all_labels, all_preds, self.target_names)
            self.evaluator.generate_confusion_matrix(all_labels, all_preds)

        if save_model_path:
            save_model(self.model, save_model_path)

def prepare_data_splits(data, split_ratio=0.8):
    split_idx = int(len(data['texts']) * split_ratio)
    train_texts = data['texts'][:split_idx]
    train_labels = data['labels'][:split_idx]
    val_texts = data['texts'][split_idx:]
    val_labels = data['labels'][split_idx:]

    return {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels}
    }

def augment_data(texts, augment_factor=2):
    augmented_texts = texts * augment_factor
    return augmented_texts

def create_advanced_optimizer(model, learning_rate=3e-5, weight_decay=1e-2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def fine_tune_model(model, train_data, val_data, tokenizer, device, num_epochs=5, batch_size=32, max_length=128):
    train_dataset = SentimentDataset(train_data['texts'], train_data['labels'], tokenizer, max_length)
    val_dataset = SentimentDataset(val_data['texts'], val_data['labels'], tokenizer, max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = create_criterion()
    optimizer = create_advanced_optimizer(model)

    trainer = AdvancedSentimentTrainer(model, train_loader, val_loader, criterion, optimizer, device, target_names=["Negative", "Positive"])
    trainer.train(num_epochs, save_model_path="fine_tuned_model.pth")

if __name__ == "__main__":
    create_logging()

    tokenizer = load_tokenizer()
    model = CustomBERTModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Data preparation
    data = {
        'texts': ["I absolutely love this product!", "The service was terrible and slow.", "Great quality!", "Not satisfied at all.", "Superb experience!", "Would not recommend."],
        'labels': [1, 0, 1, 0, 1, 0]
    }

    augmented_texts = augment_data(data['texts'], augment_factor=2)
    data_splits = prepare_data_splits({'texts': augmented_texts, 'labels': data['labels'] * 2})

    fine_tune_model(model, data_splits['train'], data_splits['val'], tokenizer, device)

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class AdvancedModelEvaluator(ModelEvaluator):
    def __init__(self, model, data_loader, criterion, device):
        super().__init__(model, data_loader, criterion, device)

    def evaluate_with_metrics(self):
        avg_loss, accuracy, all_labels, all_preds = self.evaluate()

        # Calculate additional metrics
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(all_labels, all_preds)

        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'Precision-Recall AUC: {pr_auc:.4f}')

        return avg_loss, accuracy, all_labels, all_preds, roc_auc, pr_auc

class HyperparameterTuner:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def tune(self, learning_rates, batch_sizes):
        best_val_acc = 0
        best_hyperparams = {}

        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f'Tuning with LR: {lr}, Batch Size: {batch_size}')
                
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                train_loader = torch.utils.data.DataLoader(
                    self.train_loader.dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(
                    self.val_loader.dataset, batch_size=batch_size, shuffle=False)

                trainer = SentimentTrainer(
                    self.model, train_loader, val_loader, criterion, optimizer, self.device)

                trainer.train_epoch()
                val_loss, val_acc = trainer.validate_epoch()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_hyperparams = {'learning_rate': lr, 'batch_size': batch_size}

        print(f'Best Hyperparameters: LR={best_hyperparams["learning_rate"]}, Batch Size={best_hyperparams["batch_size"]}')
        return best_hyperparams

def advanced_fine_tune_model(model, train_data, val_data, tokenizer, device, learning_rates, batch_sizes, num_epochs=3, max_length=128):
    train_dataset = SentimentDataset(train_data['texts'], train_data['labels'], tokenizer, max_length)
    val_dataset = SentimentDataset(val_data['texts'], val_data['labels'], tokenizer, max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    tuner = HyperparameterTuner(model, train_loader, val_loader, device)
    best_hyperparams = tuner.tune(learning_rates, batch_sizes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    advanced_trainer = AdvancedSentimentTrainer(model, train_loader, val_loader, criterion, optimizer, device, target_names=["Negative", "Positive"])
    advanced_trainer.train(num_epochs, save_model_path="advanced_fine_tuned_model.pth")

class ModelExporter:
    @staticmethod
    def export_to_onnx(model, input, onnx_file_path="model.onnx"):
        torch.onnx.export(model, input, onnx_file_path, export_params=True, opset_version=11, do_constant_folding=True,
                          input_names=['input_ids', 'attention_mask', 'token_type_ids'], output_names=['output'],
                          dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'token_type_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"Model exported to {onnx_file_path}")

    @staticmethod
    def load_onnx(onnx_file_path="model.onnx"):
        import onnx
        import onnxruntime as ort

        model = onnx.load(onnx_file_path)
        onnx.checker.check_model(model)
        ort_session = ort.InferenceSession(onnx_file_path)
        print(f"ONNX model loaded from {onnx_file_path}")
        return ort_session

    @staticmethod
    def run_onnx_inference(ort_session, input_ids, attention_mask, token_type_ids):
        inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
            'token_type_ids': token_type_ids.numpy()
        }
        outputs = ort_session.run(None, inputs)
        return outputs

if __name__ == "__main__":
    create_logging()

    tokenizer = load_tokenizer()
    model = CustomBERTModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare augmented data for fine-tuning and hyperparameter tuning
    data = {
        'texts': ["I absolutely love this!", "The service was terrible and slow.", "Great quality!", "Not satisfied at all.", "Superb experience!", "Would not recommend."],
        'labels': [1, 0, 1, 0, 1, 0]
    }

    augmented_texts = augment_data(data['texts'], augment_factor=2)
    data_splits = prepare_data_splits({'texts': augmented_texts, 'labels': data['labels'] * 2})

    # Hyperparameter ranges for tuning
    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes = [16, 32, 64]

    advanced_fine_tune_model(model, data_splits['train'], data_splits['val'], tokenizer, device, learning_rates, batch_sizes)

    # Export the fine-tuned model to ONNX
    input = torch.randint(0, 1000, (1, 128)).to(device)  # Input for export
    ModelExporter.export_to_onnx(model, input)

    # Load and run ONNX model inference
    ort_session = ModelExporter.load_onnx()
    input_ids, attention_mask, token_type_ids = prepare_input(["I love this product!"], tokenizer)
    outputs = ModelExporter.run_onnx_inference(ort_session, input_ids, attention_mask, token_type_ids)
    print("ONNX Model Outputs:", outputs)