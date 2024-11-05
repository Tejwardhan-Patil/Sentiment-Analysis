import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import datetime
import json

class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, dropout_prob=0.3):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = bert_outputs[1]  # CLS token output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def initialize_model(bert_model_name='bert-base-uncased', num_classes=2, dropout_prob=0.3, device='cuda'):
    model = BERTSentimentClassifier(
        bert_model_name=bert_model_name,
        num_classes=num_classes,
        dropout_prob=dropout_prob
    )
    model = model.to(device)
    return model

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = SentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

def train_epoch(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        labels = d['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def compute_metrics(predictions, labels):
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    
    accuracy = accuracy_score(labels_flat, pred_flat)
    precision = precision_score(labels_flat, pred_flat, average='weighted')
    recall = recall_score(labels_flat, pred_flat, average='weighted')
    f1 = f1_score(labels_flat, pred_flat, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    epochs,
    train_size,
    val_size
):
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            train_size
        )

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            val_size
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Val   loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

    return history

def evaluate_performance_on_test_data(
    model,
    test_data_loader,
    loss_fn,
    device,
    test_size
):
    print("Evaluating performance on test data...")
    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        test_size
    )
    print(f'Test loss {test_loss} accuracy {test_acc}')
    return test_acc, test_loss

def save_model(model, output_dir):
    print(f'Saving model to {output_dir}')
    model.save_pretrained(output_dir)

def load_model(output_dir, device='cuda'):
    print(f'Loading model from {output_dir}')
    model = BertModel.from_pretrained(output_dir)
    model = model.to(device)
    return model

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def create_optimizer(model, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    return optimizer

def create_scheduler(optimizer, num_training_steps, num_warmup_steps=0):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler

def predict(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            token_type_ids = d['token_type_ids'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds)
            real_values.extend(d['labels'].to(device))

    predictions = torch.stack(predictions).cpu().numpy()
    real_values = torch.stack(real_values).cpu().numpy()

    return predictions, real_values

def calculate_metrics_on_test_data(model, test_data_loader, device):
    print("Calculating metrics on test data...")
    predictions, real_values = predict(model, test_data_loader, device)
    
    metrics = compute_metrics(predictions, real_values)
    
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1 Score: {metrics["f1"]:.4f}')
    
    return metrics

def save_training_history(history, output_dir):
    print(f'Saving training history to {output_dir}')
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(history, f)

def load_training_history(output_dir):
    print(f'Loading training history from {output_dir}')
    with open(f'{output_dir}/training_history.json', 'r') as f:
        history = json.load(f)
    return history

def plot_training_curves(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def perform_inference(model, tokenizer, text, max_len, device='cuda'):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    logits = output[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

def evaluate_on_sample_texts(model, tokenizer, texts, max_len, device='cuda'):
    print("Evaluating on sample texts...")
    for text in texts:
        predicted_class = perform_inference(model, tokenizer, text, max_len, device)
        print(f'Text: {text}')
        print(f'Predicted Sentiment: {predicted_class}\n')

def fine_tune_model_on_additional_data(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    scheduler, 
    device, 
    epochs
):
    print("Fine-tuning the model on additional data...")

    for epoch in range(epochs):
        print(f'Fine-tuning Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(data_loader.dataset)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

    return model

def analyze_predictions(predictions, real_values):
    print("Analyzing predictions...")
    accuracy = accuracy_score(real_values, predictions)
    precision = precision_score(real_values, predictions, average='weighted')
    recall = recall_score(real_values, predictions, average='weighted')
    f1 = f1_score(real_values, predictions, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

def export_model_to_onnx(model, output_path, input, opset_version=11):
    print(f"Exporting model to ONNX format at {output_path}...")
    torch.onnx.export(
        model,
        input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'token_type_ids': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

def convert_to_torchscript(model, output_path, input):
    print(f"Converting model to TorchScript format at {output_path}...")
    traced_model = torch.jit.trace(model, input)
    traced_model.save(output_path)
    print(f"Model saved to {output_path} as TorchScript.")

def load_torchscript_model(input_path, device='cuda'):
    print(f"Loading TorchScript model from {input_path}...")
    model = torch.jit.load(input_path)
    model = model.to(device)
    return model

def explain_predictions_with_lime(model, tokenizer, text, max_len, device='cuda'):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=['negative', 'positive'])

    def predict_fn(texts):
        model.eval()
        predictions = []
        for text in texts:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                return_token_type_ids=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding['token_type_ids'].to(device)

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            logits = output[0]
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probabilities)

        return np.concatenate(predictions, axis=0)

    explanation = explainer.explain_instance(text, predict_fn, num_features=10)
    explanation.show_in_notebook()

def distill_bert_model(teacher_model, student_model, train_data_loader, val_data_loader, loss_fn, optimizer, device, epochs):
    print("Distilling BERT model...")
    
    for epoch in range(epochs):
        print(f"Distillation Epoch {epoch + 1}/{epochs}")
        teacher_model.eval()
        student_model.train()
        
        distillation_loss = 0
        correct_predictions = 0
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = loss_fn(student_outputs, teacher_outputs[1])  # Distillation loss
            
            distillation_loss += loss.item()
            _, preds = torch.max(student_outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Distillation Loss: {distillation_loss / len(train_data_loader)}")
        print(f"Training Accuracy: {correct_predictions.double() / len(train_data_loader.dataset)}")

def prune_bert_model(model, amount=0.2):
    print(f"Pruning BERT model with amount={amount}...")
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=amount,
    )

    for module, _ in parameters_to_prune:
        torch.nn.utils.prune.remove(module, 'weight')

    print("Pruning complete.")

def quantize_bert_model(model, data_loader, device='cuda'):
    print("Quantizing BERT model...")
    
    model.eval()
    
    def calibration_fn(data_loader):
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    calibration_fn(data_loader)
    
    print("Quantization complete.")
    return model

def optimize_bert_model_with_onnxruntime(onnx_model_path):
    import onnxruntime as ort
    
    print(f"Optimizing ONNX model at {onnx_model_path} with ONNX Runtime...")
    
    session = ort.InferenceSession(onnx_model_path)
    
    return session

def run_inference_with_onnxruntime(session, tokenizer, text, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='np',
        truncation=True
    )

    input_feed = {
        "input_ids": encoding['input_ids'].numpy(),
        "attention_mask": encoding['attention_mask'].numpy(),
        "token_type_ids": encoding['token_type_ids'].numpy(),
    }

    logits = session.run(None, input_feed)[0]
    predicted_class = np.argmax(logits, axis=1).item()

    return predicted_class

def export_training_metrics_to_csv(history, output_path):
    import pandas as pd

    print(f"Exporting training metrics to {output_path}...")
    
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    print(f"Training metrics saved to {output_path}.")

def visualize_attention_weights(model, tokenizer, text, max_len, device='cuda'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    attention = outputs[-1] 
    attention = attention.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention[0], cmap="viridis", ax=ax)
    plt.title("Attention Weights")
    plt.xlabel("Attention Head")
    plt.ylabel("Token Position")
    plt.show()