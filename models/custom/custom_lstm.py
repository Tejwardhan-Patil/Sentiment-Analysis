import torch
import torch.nn as nn
import torch.optim as optim

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.3):
        super(CustomLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply dropout
        out = self.dropout(out[:, -1, :])

        # Fully connected layer
        out = self.fc(out)

        return out

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss calculation
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)

                # Validation loss
                val_loss += criterion(outputs, labels).item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Logging for epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, '
              f'Validation Accuracy: {100 * correct / total}%')

def evaluate_model(model, test_loader):
    # Set model to evaluation mode
    model.eval()
    correct = 0
    total = 0

    # Disable gradient computation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            # Forward pass for evaluation
            outputs = model(inputs)
            
            # Predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print final test accuracy
    print(f'Test Accuracy: {100 * correct / total}%')

# Utility function for saving model checkpoints
def save_checkpoint(model, optimizer, epoch, filepath):
    # Save model state, optimizer state, and current epoch
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)

# Utility function for loading model checkpoints
def load_checkpoint(filepath, model, optimizer):
    # Load model checkpoint
    checkpoint = torch.load(filepath)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the epoch from the checkpoint
    return checkpoint['epoch']

# Early stopping function to monitor validation loss
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Initialize an early stopping instance
early_stopping = EarlyStopping(patience=7, delta=0.01)

def run_training_with_early_stopping(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, '
              f'Validation Accuracy: {100 * correct / total}%')

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Function to calculate and print confusion matrix
def calculate_confusion_matrix(model, test_loader):
    from sklearn.metrics import confusion_matrix
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    return cm

# Function to visualize confusion matrix
def plot_confusion_matrix(cm, classes):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Learning rate scheduler
def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epoch=10):
    lr = initial_lr * (0.5 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Function to run training with learning rate adjustment
def run_training_with_scheduler(model, train_loader, val_loader, num_epochs=30, initial_lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    for epoch in range(num_epochs):
        # Adjust learning rate based on epoch
        current_lr = adjust_learning_rate(optimizer, epoch, initial_lr)
        print(f'Epoch {epoch+1}, Learning Rate: {current_lr}')
        
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, '
              f'Validation Accuracy: {100 * correct / total}%')

# Function to calculate precision, recall, and F1-score
def calculate_precision_recall_f1(model, test_loader):
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    
    return precision, recall, f1

# Gradient clipping to avoid exploding gradients
def run_training_with_gradient_clipping(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, max_norm=5.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, '
              f'Validation Accuracy: {100 * correct / total}%')

# Function to track and visualize training and validation loss over epochs
def plot_training_loss(train_losses, val_losses):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.show()

# Define a custom metric for evaluation, such as sentiment accuracy
def custom_metric(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

# Function to run training and track custom metrics
def run_training_with_custom_metrics(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss/len(train_loader))

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss/len(val_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]}, '
              f'Validation Accuracy: {100 * correct / total}%')

    # Plot training and validation losses after training
    plot_training_loss(train_losses, val_losses)

# Function to calculate and log model performance over time
def log_model_performance(epoch, train_loss, val_loss, val_accuracy, log_file="training_log.txt"):
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%\n")

# Function to run training and log performance
def run_training_with_logging(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, log_file="training_log.txt"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}%')

        # Log model performance
        log_model_performance(epoch + 1, train_loss, val_loss, val_accuracy, log_file)

# Implementing a cyclic learning rate scheduler
class CyclicLR:
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, mode="triangular"):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_up
        self.mode = mode
        self.iteration = 0
        self.lr = base_lr

    def step(self):
        cycle = (self.iteration // (self.step_size_up + self.step_size_down)) + 1
        x = abs(self.iteration / self.step_size_up - 2 * cycle + 1)
        scale_factor = max(0, (1 - x))
        if self.mode == "triangular":
            self.lr = self.base_lr + (self.max_lr - self.base_lr) * scale_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self.iteration += 1

# Running training with cyclic learning rate
def run_training_with_cyclic_lr(model, train_loader, val_loader, num_epochs=30, base_lr=0.001, max_lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update learning rate after each batch
            scheduler.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}%')

# Function to fine-tune a pre-trained LSTM model
def fine_tune_model(pretrained_model, train_loader, val_loader, num_epochs=5, learning_rate=1e-5):
    optimizer = optim.Adam(pretrained_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        pretrained_model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(pretrained_model.device), labels.to(pretrained_model.device)
            
            optimizer.zero_grad()
            outputs = pretrained_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pretrained_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(pretrained_model.device), labels.to(pretrained_model.device)
                outputs = pretrained_model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Fine-Tuning Loss: {total_loss/len(train_loader)}, '
              f'Validation Accuracy: {100 * correct / total}%')

# Implementing checkpoint saving during training
def run_training_with_checkpoints(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, checkpoint_interval=5, checkpoint_dir="checkpoints/"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}, Validation Accuracy: {100 * correct / total}%')

        # Save checkpoint at intervals
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}model_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

# Function to load the best model from checkpoints
def load_best_model(checkpoint_dir="checkpoints/"):
    import os
    best_epoch = -1
    best_model_path = None
    
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("model_epoch_") and filename.endswith(".pth"):
            epoch = int(filename.split("_")[2].split(".")[0])
            if epoch > best_epoch:
                best_epoch = epoch
                best_model_path = os.path.join(checkpoint_dir, filename)

    if best_model_path:
        print(f"Loading model from {best_model_path}")
        return torch.load(best_model_path)
    else:
        print("No checkpoint found")
        return None