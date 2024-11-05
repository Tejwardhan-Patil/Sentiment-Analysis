import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import random
import math

class CustomTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src) * math.sqrt(embedded.size(2))
        embedded = self.positional_encoding(embedded)
        encoded_src = self.transformer_encoder(embedded, src_mask)
        output = self.fc_out(self.dropout(encoded_src.mean(dim=1)))
        return F.log_softmax(output, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Extending the positional encoding with more configurable components
class ExtendedPositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=10000, custom_scale_factor=1.0):
        super(ExtendedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale_factor = custom_scale_factor

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term * self.scale_factor)
        pe[:, 1::2] = torch.cos(position * div_term * self.scale_factor)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Additional normalization layer to stabilize the transformer training process
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.layer_norm(x)

# Customized attention mechanism to allow further control over the attention weights
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CustomMultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return self.dropout(attn_output), attn_weights

# Initializing Transformer model with custom multi-head attention
class CustomTransformerWithAttention(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(CustomTransformerWithAttention, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = ExtendedPositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.custom_attention = CustomMultiheadAttention(model_dim, num_heads, dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.layer_norm = CustomLayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src) * math.sqrt(embedded.size(2))
        embedded = self.positional_encoding(embedded)
        encoded_src = self.transformer_encoder(embedded, src_mask)
        attn_output, attn_weights = self.custom_attention(encoded_src, encoded_src, encoded_src, attn_mask=src_mask)
        attn_output = self.layer_norm(attn_output + encoded_src)  # Add & Norm
        output = self.fc_out(self.dropout(attn_output.mean(dim=1)))
        return F.log_softmax(output, dim=-1), attn_weights

# Defining a custom feedforward network to be used within the Transformer layers
class CustomFeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(CustomFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

# Advanced Transformer Encoder Layer that includes a custom feedforward network and layer normalization
class AdvancedTransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(AdvancedTransformerEncoderLayer, self).__init__()
        self.self_attn = CustomMultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.feed_forward = CustomFeedForward(model_dim, ff_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        attn_output = self.layer_norm(attn_output + src)
        output = self.feed_forward(attn_output)
        return output

# Stacking multiple advanced Transformer Encoder Layers to build a deeper Transformer
class AdvancedTransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(AdvancedTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([AdvancedTransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.layer_norm(src)

# Defining the final custom Transformer model, integrating all previous components
class FinalCustomTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, ff_dim, num_layers, output_dim, dropout=0.1):
        super(FinalCustomTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = ExtendedPositionalEncoding(model_dim, dropout)
        self.encoder = AdvancedTransformerEncoder(model_dim, num_heads, ff_dim, num_layers, dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src) * math.sqrt(embedded.size(2))
        embedded = self.positional_encoding(embedded)
        encoded_src = self.encoder(embedded, src_mask)
        output = self.fc_out(self.dropout(encoded_src.mean(dim=1)))
        return F.log_softmax(output, dim=-1)

# Custom Transformer for Sequence to Sequence tasks (e.g., translation)
class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = FinalCustomTransformer(input_dim, model_dim, num_heads, ff_dim, num_layers, model_dim, dropout)
        self.decoder = FinalCustomTransformer(output_dim, model_dim, num_heads, ff_dim, num_layers, model_dim, dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoded_src = self.encoder(src, src_mask)
        decoded_tgt = self.decoder(tgt, tgt_mask)
        output = self.fc_out(self.dropout(decoded_tgt.mean(dim=1)))
        return F.log_softmax(output, dim=-1)

# Define custom masking function for sequence-to-sequence tasks
def generate_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Custom Decoder Layer with attention over the encoder's output
class CustomDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(CustomDecoderLayer, self).__init__()
        self.self_attn = CustomMultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.enc_dec_attn = CustomMultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.feed_forward = CustomFeedForward(model_dim, ff_dim, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.layer_norm(tgt + tgt2)
        tgt2, _ = self.enc_dec_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.layer_norm(tgt + tgt2)
        return self.feed_forward(tgt)

# Custom Decoder that stacks multiple Decoder Layers
class CustomTransformerDecoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([CustomDecoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.layer_norm(tgt)

# Full Transformer model for sequence generation tasks
class FullSeq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(FullSeq2SeqTransformer, self).__init__()
        self.encoder = FinalCustomTransformer(input_dim, model_dim, num_heads, ff_dim, num_layers, model_dim, dropout)
        self.decoder = CustomTransformerDecoder(model_dim, num_heads, ff_dim, num_layers, dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        decoded_output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.fc_out(self.dropout(decoded_output.mean(dim=1)))
        return F.log_softmax(output, dim=-1)

# Mask generation helper function for target sequences
def generate_tgt_mask(size):
    mask = torch.tril(torch.ones(size, size)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Defining the optimizer with custom learning rate scheduler
def configure_optimizer_and_scheduler(model, learning_rate, warmup_steps, weight_decay=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=warmup_steps, gamma=0.95)
    return optimizer, scheduler

# Custom learning rate scheduler to allow for dynamic adjustment of learning rate during training
class CustomLRScheduler:
    def __init__(self, optimizer, warmup_steps, model_dim, factor=1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_lr(self):
        lr = self.factor * (self.model_dim ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        return lr

# Training loop for the Transformer model
def train_model(model, data_loader, optimizer, scheduler, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            src_mask = generate_square_subsequent_mask(src.size(1)).to(device)
            tgt_mask = generate_tgt_mask(tgt_input.size(1)).to(device)

            output, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader)}')

# Evaluation loop for the Transformer model on validation/test data
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = generate_square_subsequent_mask(src.size(1)).to(device)
            tgt_mask = generate_tgt_mask(tgt_input.size(1)).to(device)

            output, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

            total_loss += loss.item()

    return total_loss / len(data_loader)

# Inference function for making predictions on new data using a trained Transformer model
def inference(model, src_input, device, max_length):
    # Define start and end tokens locally if not defined globally
    start_token = 101
    end_token = 102

    model.eval()
    src_input = src_input.to(device)
    src_mask = generate_square_subsequent_mask(src_input.size(1)).to(device)

    with torch.no_grad():
        memory = model.encoder(src_input, src_mask)
        output_sequence = torch.zeros(1, 1).fill_(start_token).type_as(src_input).to(device)
        for _ in range(max_length):
            tgt_mask = generate_tgt_mask(output_sequence.size(1)).to(device)
            decoded_output = model.decoder(output_sequence, memory, tgt_mask=tgt_mask)
            output_prob = model.fc_out(decoded_output[:, -1, :])
            next_token = output_prob.argmax(1).unsqueeze(0)

            output_sequence = torch.cat([output_sequence, next_token], dim=1)
            if next_token.item() == end_token:
                break

    return output_sequence

pad_token = 0
# Utility function for preparing data batches for training and evaluation
def prepare_data_batches(data, batch_size, device):
    src_sequences = [torch.tensor(item['src'], dtype=torch.long) for item in data]
    tgt_sequences = [torch.tensor(item['tgt'], dtype=torch.long) for item in data]
    src_padded = nn.utils.rnn.pad_sequence(src_sequences, batch_first=True, padding_value=pad_token)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_sequences, batch_first=True, padding_value=pad_token)
    return torch.utils.data.DataLoader({'src': src_padded, 'tgt': tgt_padded}, batch_size=batch_size, shuffle=True)

# Function to handle loading pre-trained model weights
def load_pretrained_weights(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model weights from {checkpoint_path}")

# Function to save model weights after training
def save_model_weights(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model weights to {checkpoint_path}")

# Data augmentation through token replacement and sequence expansion
def augment_data(data, augmentation_factor=0.1):
    augmented_data = []
    for item in data:
        augmented_item = item.copy()
        if random.random() < augmentation_factor:
            src_seq = augmented_item['src']
            tgt_seq = augmented_item['tgt']
            augmented_item['src'] = augment_sequence(src_seq)
            augmented_item['tgt'] = augment_sequence(tgt_seq)
        augmented_data.append(augmented_item)
    return augmented_data

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
available_tokens = list(tokenizer.get_vocab().keys())
# Custom sequence augmentation strategy using token replacement and insertion
def augment_sequence(sequence):
    augmented_sequence = sequence.copy()
    # Token replacement
    random_idx = random.randint(0, len(augmented_sequence) - 1)
    augmented_sequence[random_idx] = random.choice(available_tokens)

    # Token insertion
    insert_idx = random.randint(0, len(augmented_sequence))
    augmented_sequence.insert(insert_idx, random.choice(available_tokens))

    return augmented_sequence

vocab_size = 5000
# Function to generate random data samples for testing purposes
def generate_random_data(num_samples, max_seq_length):
    data = []
    for _ in range(num_samples):
        src_sequence = [random.randint(1, vocab_size - 1) for _ in range(random.randint(1, max_seq_length))]
        tgt_sequence = [random.randint(1, vocab_size - 1) for _ in range(random.randint(1, max_seq_length))]
        data.append({'src': src_sequence, 'tgt': tgt_sequence})
    return data

# Visualization function to plot model training curves
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Utility function for computing accuracy during evaluation
def compute_accuracy(predictions, targets):
    _, predicted = torch.max(predictions, dim=1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

# Function for saving training and evaluation metrics to a log file
def log_metrics(log_path, metrics):
    with open(log_path, 'a') as log_file:
        for key, value in metrics.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")

# Utility function to print model architecture summary
def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params} trainable parameters")
    print(model)

# Adding dropout and weight regularization to prevent overfitting during training
def apply_regularization(model, dropout_rate=0.3, weight_decay=0.01):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.mul_((1 - weight_decay))
            if module.bias is not None:
                module.bias.data.mul_((1 - weight_decay))
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate

# Tokenization function for preprocessing text sequences into input tensors
def tokenize_text(text, tokenizer, max_length):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_length]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids += [pad_token] * (max_length - len(token_ids))
    return torch.tensor(token_ids).unsqueeze(0)