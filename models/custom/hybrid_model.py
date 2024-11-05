import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim
import numpy as np

# Hybrid Model class combining LSTM and BERT-based architecture
class HybridModel(nn.Module):
    def __init__(self, lstm_hidden_size, num_layers, bert_model_name, num_classes, dropout=0.5):
        super(HybridModel, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        # LSTM layer configuration
        self.lstm = nn.LSTM(input_size=bert_hidden_size, hidden_size=lstm_hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights for the LSTM and fully connected layers
        self.init_weights()

    def init_weights(self):
        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # Initialize fully connected layer weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence_output = bert_output.last_hidden_state

        # LSTM encoding
        lstm_output, _ = self.lstm(bert_sequence_output)

        # Apply dropout
        lstm_output = self.dropout(lstm_output)

        # Fully connected layer
        output = self.fc(lstm_output[:, -1, :])  # Use the last LSTM output for classification

        return output


# Tokenizer setup for preprocessing the input text
class TextProcessor:
    def __init__(self, bert_model_name):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def encode(self, text, max_length=512):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

# Function to compute accuracy of predictions
def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Dataset class to handle text data for sentiment analysis
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, text_processor):
        self.texts = texts
        self.labels = labels
        self.text_processor = text_processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids, attention_mask = self.text_processor.encode(text)
        return input_ids.squeeze(), attention_mask.squeeze(), label


# Model training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0

    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, avg_acc

# Function to evaluate the model's performance on validation/test sets
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    return avg_loss, avg_acc

# Prediction function to make predictions on new data
def predict(model, text, text_processor, device):
    model.eval()
    input_ids, attention_mask = text_processor.encode(text)

    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    probabilities = nn.Softmax(dim=1)(outputs)
    _, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item()

# Training utility to prepare data loaders and initiate training
def run_training_loop(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=3):
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Custom weight decay implementation for regularization
def apply_weight_decay(parameters, lr, decay_factor):
    decay_params = []
    no_decay_params = []
    
    for param in parameters:
        if param.requires_grad:
            if len(param.shape) == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': decay_factor},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)

    return optimizer

# Function to adjust learning rate during training
def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_factor, lr_decay_epoch):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    lr = initial_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f'Model checkpoint saved at epoch {epoch}')

# Function to load a model from checkpoint
def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Loaded model from checkpoint at epoch {epoch}')
    return model, optimizer, epoch

# Data loader preparation function
def prepare_dataloader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Learning rate scheduler with warm-up mechanism
class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        current_step = self._step_count
        if current_step < self.warmup_steps:
            return [base_lr * (current_step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [base_lr * max(0.0, (self.total_steps - current_step) / (self.total_steps - self.warmup_steps))
                    for base_lr in self.base_lrs]

# Function to apply the warmup scheduler during training
def apply_warmup_scheduler(optimizer, warmup_steps, total_steps):
    scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps)
    return scheduler

# Early stopping mechanism to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Function to log training progress
def log_training_progress(epoch, train_loss, train_acc, val_loss, val_acc):
    log_message = (f'Epoch: {epoch+1}, '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    print(log_message)

# Function to initialize model weights
def initialize_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

# Custom learning rate finder for optimal learning rate search
class LearningRateFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def find_lr(self, dataloader, start_lr=1e-7, end_lr=1, num_iter=100):
        self.model.train()
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: (end_lr / start_lr) ** (step / num_iter))

        best_loss = float('inf')
        lr_log = []
        loss_log = []

        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            if i >= num_iter:
                break

            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            lr_log.append(lr)
            loss_log.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()

            print(f'Iteration {i+1}/{num_iter}, LR: {lr:.6f}, Loss: {loss.item():.4f}')

        return lr_log, loss_log

# Function to plot learning rate vs loss during LR search
def plot_lr_finder(lr_log, loss_log):
    import matplotlib.pyplot as plt
    plt.plot(lr_log, loss_log)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

# Function to implement gradient clipping during training
def apply_gradient_clipping(optimizer, model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

# Utility function to calculate F1 score
def calculate_f1_score(predictions, labels):
    from sklearn.metrics import f1_score
    _, predicted = torch.max(predictions, 1)
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    return f1

# Function to track model metrics such as precision, recall, and F1-score
def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision, recall, f1 = compute_precision_recall_f1(all_preds, all_labels)
    return precision, recall, f1

# Function to compute precision, recall, and F1-score using sklearn
def compute_precision_recall_f1(predictions, labels):
    from sklearn.metrics import precision_score, recall_score, f1_score
    _, predicted = torch.max(predictions, 1)
    
    precision = precision_score(labels.cpu(), predicted.cpu(), average='weighted')
    recall = recall_score(labels.cpu(), predicted.cpu(), average='weighted')
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    
    return precision, recall, f1

# Function to handle multi-GPU training if available
def prepare_multi_gpu(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs for training')
    return model

# Custom optimizer with Lookahead mechanism for better convergence
class LookaheadOptimizer:
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.counter = 0
        self.slow_weights = [p.clone().detach() for p in base_optimizer.param_groups[0]['params']]

    def step(self):
        self.base_optimizer.step()
        self.counter += 1
        if self.counter % self.k == 0:
            for param, slow_param in zip(self.base_optimizer.param_groups[0]['params'], self.slow_weights):
                slow_param += self.alpha * (param - slow_param)
                param.data.copy_(slow_param)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

# Function to freeze certain layers of the model during fine-tuning
def freeze_bert_layers(model, freeze_layer_count=0):
    for name, param in model.bert.named_parameters():
        if freeze_layer_count > 0:
            freeze_layer_count -= 1
            param.requires_grad = False
        else:
            param.requires_grad = True

# Function to dynamically adjust batch size based on available GPU memory
def dynamic_batch_sizing(dataloader, initial_batch_size, model, optimizer, criterion, device):
    batch_size = initial_batch_size
    while batch_size > 1:
        try:
            dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=True)
            train_model(model, dataloader, optimizer, criterion, device)
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = batch_size // 2
                print(f"Reducing batch size to {batch_size} due to memory constraints")
            else:
                raise e
    return batch_size

# Function to implement SWA (Stochastic Weight Averaging) for better generalization
def apply_swa(optimizer, swa_start, swa_model, swa_scheduler, epoch):
    if epoch >= swa_start:
        optimizer.update_swa()
        swa_model.update_parameters(optimizer)
        swa_scheduler.step()

# Function to save and load the best model based on validation performance
def save_best_model(model, path, val_acc, best_acc):
    if val_acc > best_acc:
        torch.save(model.state_dict(), path)
        best_acc = val_acc
        print(f'Best model saved with accuracy: {best_acc:.4f}')
    return best_acc

# Mixed precision training to speed up training and save memory
def apply_mixed_precision(model, dataloader, optimizer, criterion, device):
    scaler = torch.cuda.amp.GradScaler()
    
    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Function to fine-tune model on additional datasets
def fine_tune_model_on_new_data(model, new_data_loader, optimizer, criterion, device, epochs=2):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for input_ids, attention_mask, labels in new_data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, labels)

        avg_loss = total_loss / len(new_data_loader)
        avg_acc = total_acc / len(new_data_loader)
        print(f'Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')

# Exponential moving average for model parameters during training
def apply_exponential_moving_average(model, ema_model, decay=0.999):
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(model_param.data, alpha=(1.0 - decay))

# Function to evaluate confusion matrix
def evaluate_confusion_matrix(model, dataloader, device):
    from sklearn.metrics import confusion_matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Utility function for precision-recall curve visualization
def plot_precision_recall_curve(model, dataloader, device):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            probabilities = nn.Softmax(dim=1)(outputs)
            all_preds.append(probabilities[:, 1].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)

    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Function for test-time augmentation (TTA) during inference
def test_time_augmentation(model, text_processor, texts, device, n_augmentations=5):
    model.eval()
    predictions = []

    for text in texts:
        augmented_preds = []
        for _ in range(n_augmentations):
            input_ids, attention_mask = text_processor.encode(text)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probabilities = nn.Softmax(dim=1)(outputs)
                augmented_preds.append(probabilities.cpu().numpy())

        avg_pred = np.mean(augmented_preds, axis=0)
        predictions.append(np.argmax(avg_pred, axis=1).item())

    return predictions