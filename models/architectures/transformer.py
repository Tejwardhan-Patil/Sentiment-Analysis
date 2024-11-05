import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class TransformerSentimentModel(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate=0.1):
        super(TransformerSentimentModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token's output
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits

def preprocess_data(texts, tokenizer, max_len):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    return inputs['input_ids'], inputs['attention_mask']

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}, Accuracy: {total_correct / len(train_loader.dataset)}")

        evaluate_model(model, val_loader)

def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

    print(f"Validation Accuracy: {total_correct / len(data_loader.dataset)}")

def prepare_data_loader(tokenizer, texts, labels, batch_size=32, max_len=128):
    input_ids, attention_mask = preprocess_data(texts, tokenizer, max_len)
    labels = torch.tensor(labels)
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label)

def create_data_loaders(train_texts, val_texts, train_labels, val_labels, tokenizer, batch_size=32, max_len=128):
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def initialize_model(hidden_size, num_labels, dropout_rate=0.1):
    model = TransformerSentimentModel(hidden_size=hidden_size, num_labels=num_labels, dropout_rate=dropout_rate)
    return model

def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def train_and_evaluate_model(model, train_loader, val_loader, epochs=5, lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss, correct_predictions = 0, 0
        total_samples = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}, Accuracy: {correct_predictions / total_samples}")
        evaluate_model(model, val_loader)

# Texts and labels for training/validation
train_texts = ["I love this product!", "Worst service ever.", "Not bad, could be better."]
train_labels = [1, 0, 1]

val_texts = ["Great experience!", "Terrible quality."]
val_labels = [1, 0]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader, val_loader = create_data_loaders(train_texts, val_texts, train_labels, val_labels, tokenizer)

model = initialize_model(hidden_size=768, num_labels=2)
train_and_evaluate_model(model, train_loader, val_loader)

# Function to save the trained model to a specified path
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

# Function to load a pre-trained model from a specified path
def load_model(path, hidden_size, num_labels, dropout_rate=0.1):
    model = TransformerSentimentModel(hidden_size=hidden_size, num_labels=num_labels, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Function to make predictions on new data
def predict(model, tokenizer, texts, max_len=128):
    model.eval()
    inputs = preprocess_data(texts, tokenizer, max_len)
    input_ids, attention_mask = inputs
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=1)
    return predictions

# Function to evaluate the model on a test set
def evaluate_on_test_set(model, test_texts, test_labels, tokenizer, max_len=128, batch_size=32):
    test_loader = prepare_data_loader(tokenizer, test_texts, test_labels, batch_size=batch_size, max_len=max_len)
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask)
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy}")
    return accuracy

# Function to compute precision, recall, and F1-score
def compute_metrics(logits, labels):
    preds = logits.argmax(dim=1)
    true_positives = (preds & labels).sum().item()
    false_positives = (preds & ~labels).sum().item()
    false_negatives = (~preds & labels).sum().item()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Function to evaluate precision, recall, and F1-score on a data loader
def evaluate_metrics(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask)
            all_preds.append(logits)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    precision, recall, f1_score = compute_metrics(all_preds, all_labels)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    return precision, recall, f1_score

# Creating a custom loss function that combines cross-entropy and an additional regularization term
class CustomLoss(nn.Module):
    def __init__(self, weight=None):
        super(CustomLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        ce_loss = self.cross_entropy_loss(logits, labels)
        reg_loss = torch.sum(torch.abs(logits))  # Regularization term
        total_loss = ce_loss + 0.01 * reg_loss
        return total_loss

# Using the custom loss function in training
def train_with_custom_loss(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = CustomLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")
        evaluate_model(model, val_loader)

# Function to visualize the model's attention weights
def visualize_attention_weights(model, tokenizer, text, max_len=128):
    model.eval()
    inputs = preprocess_data([text], tokenizer, max_len)
    input_ids, attention_mask = inputs
    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attention_weights = outputs.attentions[-1]  # Get attention weights from the last layer
    print(f"Attention Weights for text: {text}")
    return attention_weights

# Function to save the tokenizer
def save_tokenizer(tokenizer, path):
    tokenizer.save_pretrained(path)
    print(f"Tokenizer saved at {path}")

# Function to load the tokenizer
def load_tokenizer(path):
    tokenizer = BertTokenizer.from_pretrained(path)
    print(f"Tokenizer loaded from {path}")
    return tokenizer

# Custom learning rate scheduler that adjusts the learning rate dynamically based on validation loss
class CustomLRScheduler:
    def __init__(self, optimizer, patience=2, cooldown=0, min_lr=1e-7, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.factor = factor
        self.best_loss = float('inf')
        self.cooldown_counter = 0
        self.patience_counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter > self.patience:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                print(f"Learning rate decreased to {new_lr}")
                self.patience_counter = 0
                self.cooldown_counter = self.cooldown

# Function to apply the custom learning rate scheduler during training
def train_with_scheduler(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = CustomLRScheduler(optimizer)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader)
        scheduler.step(val_loss)

# Function to generate attention heatmaps
def generate_attention_heatmap(attention_weights, tokenizer, text):
    import matplotlib.pyplot as plt
    import seaborn as sns

    tokens = tokenizer.tokenize(text)
    attention_weights = attention_weights.squeeze(0).mean(dim=1).cpu().numpy()  # Averaging across heads
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title("Attention Heatmap")
    plt.show()

# Function to evaluate the model with precision, recall, F1, and accuracy metrics
def evaluate_with_metrics(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask)
            all_preds.append(logits)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = compute_accuracy(all_preds, all_labels)
    precision, recall, f1 = compute_metrics(all_preds, all_labels)
    
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy, precision, recall, f1

# Function to perform inference on new text data with detailed metrics
def inference_with_metrics(model, tokenizer, texts, labels, max_len=128):
    model.eval()
    inputs = preprocess_data(texts, tokenizer, max_len)
    input_ids, attention_mask = inputs
    labels = torch.tensor(labels)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    accuracy = compute_accuracy(logits, labels)
    precision, recall, f1 = compute_metrics(logits, labels)
    
    print(f"Inference Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy, precision, recall, f1

# Custom optimizer setup function for more advanced use cases
def setup_optimizer(model, learning_rate=1e-5, weight_decay=0.01):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

# Custom learning rate scheduler setup function
def setup_scheduler(optimizer, step_size=2, gamma=0.1):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler

# Training function with advanced optimizer and scheduler
def train_with_advanced_optim_scheduler(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5, weight_decay=0.01):
    optimizer = setup_optimizer(model, learning_rate, weight_decay)
    scheduler = setup_scheduler(optimizer)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy: {correct_predictions / total_samples}")
        scheduler.step()
        evaluate_model(model, val_loader)

# Function to handle mixed precision training for improving performance on modern GPUs
def train_with_mixed_precision(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5):
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}, Accuracy: {total_correct / len(train_loader.dataset)}")
        evaluate_model(model, val_loader)

# Function to create a learning rate finder to identify the optimal learning rate
def lr_finder(model, train_loader, init_lr=1e-7, final_lr=10, beta=0.98):
    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(final_lr / init_lr) ** (1 / len(train_loader)))
    
    best_loss = float('inf')
    avg_loss, beta_avg_loss = 0, 0
    lr_history, loss_history = [], []

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        beta_avg_loss = avg_loss / (1 - beta ** (len(loss_history) + 1))
        
        if len(loss_history) > 1 and beta_avg_loss > 4 * best_loss:
            break

        if beta_avg_loss < best_loss:
            best_loss = beta_avg_loss

        lr_history.append(lr_schedule.get_last_lr())
        loss_history.append(beta_avg_loss)

        lr_schedule.step()
    
    return lr_history, loss_history

# Advanced function to perform gradient accumulation during training for large batch sizes
def train_with_gradient_accumulation(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5, accumulation_steps=4):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_predictions = 0
        total_samples = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy: {correct_predictions / total_samples}")
        evaluate_model(model, val_loader)

# Function to perform adversarial training by perturbing the input embeddings slightly during training
def train_with_adversarial_perturbation(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5, epsilon=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()

            embeddings = model.bert.embeddings(input_ids)
            perturbation = epsilon * torch.randn_like(embeddings).to(embeddings.device)
            perturbed_embeddings = embeddings + perturbation

            logits = model(perturbed_embeddings, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}, Accuracy: {total_correct / len(train_loader.dataset)}")
        evaluate_model(model, val_loader)

# Main execution
if __name__ == "__main__":
    # Usage of the functions
    hidden_size = 768
    num_labels = 2
    model = TransformerSentimentModel(hidden_size=hidden_size, num_labels=num_labels)

    train_loader = ...
    val_loader = ...

    train_with_mixed_precision(model, train_loader, val_loader)