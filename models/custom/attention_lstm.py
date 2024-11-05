import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(AttentionLSTM, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Attention Mechanism Layer
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def attention_layer(self, lstm_output, final_hidden_state):
        """
        Attention mechanism to focus on the important parts of the input sequence.
        """
        # lstm_output: Output of the LSTM layer (batch_size, seq_length, hidden_dim * 2)
        # final_hidden_state: Last hidden state of LSTM (batch_size, hidden_dim * 2)

        hidden = final_hidden_state.squeeze(0)  # Reduce unnecessary dimensions
        # Attention weights calculation
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        # Softmax to normalize attention scores
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        # Apply attention weights to LSTM outputs
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, x):
        """
        Forward pass through the Attention LSTM model.
        """
        # Embedding the input tokens (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(self.embedding(x))

        # Passing through the LSTM layer (returns LSTM outputs and hidden states)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Concat the final forward and backward hidden states (for bidirectional LSTM)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Passing through the attention layer
        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Passing through fully connected layer
        output = self.fc(self.dropout(attn_output))
        
        return output


class ExtendedAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, additional_feature_dim):
        """
        Extending the Attention LSTM model to handle additional features along with text input.
        """
        super(ExtendedAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Additional fully connected layer for the extra features
        self.feature_fc = nn.Linear(additional_feature_dim, hidden_dim)
        
        # Redefining the final fully connected layer to handle combined feature and LSTM output
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim, output_dim)

    def forward(self, x, additional_features):
        """
        Forward pass with additional features.
        """
        # Get output from the base Attention LSTM model
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Process the hidden state through attention mechanism
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Process the additional features
        feature_output = F.relu(self.feature_fc(additional_features))

        # Concatenate LSTM output with additional features
        combined_output = torch.cat((attn_output, feature_output), dim=1)

        # Passing through final fully connected layer
        output = self.fc(self.dropout(combined_output))
        
        return output


class AttentionLSTMWithGlobalAttention(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        """
        Attention LSTM with an additional global attention layer.
        """
        super(AttentionLSTMWithGlobalAttention, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # Global attention weights and bias
        self.global_attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.global_bias = nn.Parameter(torch.zeros(hidden_dim * 2))

    def global_attention_layer(self, lstm_output):
        """
        Apply global attention mechanism on the LSTM outputs.
        """
        # Linear transformation with bias
        global_attn = self.global_attention(lstm_output) + self.global_bias

        # Applying tanh activation to focus on relevant parts
        global_attn = torch.tanh(global_attn)

        return global_attn

    def forward(self, x):
        """
        Forward pass with global attention mechanism.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Applying global attention on top of the output
        global_attn_output = self.global_attention_layer(attn_output)

        # Passing through fully connected layer
        output = self.fc(self.dropout(global_attn_output))

        return output


class MultiHeadAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, num_heads):
        """
        Attention LSTM with a multi-head attention mechanism.
        """
        super(MultiHeadAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Multi-head attention layers
        self.multi_head_attention = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(num_heads)])

    def multi_head_attention_layer(self, lstm_output):
        """
        Multi-head attention mechanism to focus on different aspects of the input.
        """
        head_outputs = []
        for attention_layer in self.multi_head_attention:
            # Applying attention from each head
            head_attn = torch.tanh(attention_layer(lstm_output))
            head_outputs.append(head_attn)
        
        # Combining the outputs from all heads
        combined_attn_output = torch.cat(head_outputs, dim=1)

        return combined_attn_output

    def forward(self, x):
        """
        Forward pass with multi-head attention mechanism.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Apply multi-head attention
        multi_head_attn_output = self.multi_head_attention_layer(attn_output)

        # Passing through fully connected layer
        output = self.fc(self.dropout(multi_head_attn_output))

        return output

class HierarchicalAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, sentence_hidden_dim, document_hidden_dim):
        """
        Hierarchical Attention LSTM model that incorporates both word-level and sentence-level attention.
        """
        super(HierarchicalAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # LSTM for sentence-level modeling
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, sentence_hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # LSTM for document-level modeling
        self.document_lstm = nn.LSTM(sentence_hidden_dim * 2, document_hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Fully connected layer for final output
        self.fc = nn.Linear(document_hidden_dim * 2, output_dim)

    def forward(self, sentences):
        """
        Forward pass for hierarchical attention model. 
        Input is a list of sentences where each sentence is processed at word level, then sentence level.
        """
        sentence_embeddings = []

        for sentence in sentences:
            # Word-level attention for each sentence
            embedded_sentence = self.dropout(self.embedding(sentence))
            lstm_output, (hidden, cell) = self.lstm(embedded_sentence)

            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            sentence_attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))
            sentence_embeddings.append(sentence_attn_output)

        # Convert the list of sentence embeddings into a batch of sentences
        sentence_batch = torch.stack(sentence_embeddings)

        # Sentence-level LSTM
        sentence_lstm_output, (sentence_hidden, _) = self.sentence_lstm(sentence_batch)

        # Concatenating the final forward and backward sentence states
        sentence_hidden = torch.cat((sentence_hidden[-2,:,:], sentence_hidden[-1,:,:]), dim=1)

        # Document-level LSTM with attention
        document_lstm_output, (document_hidden, _) = self.document_lstm(sentence_lstm_output)
        document_hidden = torch.cat((document_hidden[-2,:,:], document_hidden[-1,:,:]), dim=1)

        # Final attention mechanism on document level
        document_attn_output = self.attention_layer(document_lstm_output, document_hidden.unsqueeze(0))

        # Passing through fully connected layer
        output = self.fc(self.dropout(document_attn_output))

        return output


class AttentionWithPositionalEncodingLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, max_len=500):
        """
        LSTM with Attention and Positional Encoding.
        """
        super(AttentionWithPositionalEncodingLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # Positional Encoding Layer
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        
        # Redefining embedding to include positional encoding
        self.max_len = max_len

    def forward(self, x):
        """
        Forward pass with positional encoding.
        """
        batch_size, seq_length = x.size()

        # Embed the input
        embedded = self.dropout(self.embedding(x))
        
        # Add positional encoding
        positional_encoding = self.positional_encoding[:, :seq_length, :]
        embedded = embedded + positional_encoding

        # Passing through the LSTM layer
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Concatenating the hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Applying attention mechanism
        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Final output layer
        output = self.fc(self.dropout(attn_output))

        return output


class StackedAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        """
        Stacked Attention LSTM with multiple attention layers.
        """
        super(StackedAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Stacking multiple attention layers
        self.attention_layer_1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_layer_2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attention_layer_3 = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x):
        """
        Forward pass with stacked attention.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # First attention layer
        attn_output_1 = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Second attention layer
        attn_output_2 = F.tanh(self.attention_layer_1(attn_output_1))

        # Third attention layer
        attn_output_3 = F.tanh(self.attention_layer_2(attn_output_2))

        # Final attention layer for output
        final_attn_output = self.attention_layer_3(attn_output_3)

        # Passing through fully connected layer
        output = self.fc(self.dropout(final_attn_output))

        return output


class ResidualAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        """
        Attention LSTM with Residual Connections.
        """
        super(ResidualAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Residual connections across LSTM layers
        self.residual_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x):
        """
        Forward pass with residual connections.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Apply residual connection
        lstm_output_residual, (hidden_residual, _) = self.residual_lstm(lstm_output + embedded)

        hidden = torch.cat((hidden_residual[-2,:,:], hidden_residual[-1,:,:]), dim=1)

        # Attention mechanism after residual LSTM
        attn_output = self.attention_layer(lstm_output_residual, hidden.unsqueeze(0))

        # Final fully connected layer
        output = self.fc(self.dropout(attn_output))

        return output


class SelfAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, attention_dim):
        """
        Self-Attention LSTM model that incorporates a self-attention mechanism on top of the LSTM output.
        """
        super(SelfAttentionLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Self-attention layer
        self.self_attention = nn.Linear(hidden_dim * 2, attention_dim)

        # Fully connected layer
        self.fc = nn.Linear(attention_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the Self-Attention LSTM model.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Apply self-attention
        attention_scores = torch.tanh(self.self_attention(lstm_output))
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        # Passing through fully connected layer
        output = self.fc(self.dropout(context_vector))

        return output

class MultiLayerSelfAttentionLSTM(SelfAttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, attention_dim, num_attention_layers):
        """
        LSTM with multi-layer self-attention mechanism for enhanced focus across multiple layers.
        """
        super(MultiLayerSelfAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, attention_dim)

        # Multiple self-attention layers
        self.multi_attention_layers = nn.ModuleList([nn.Linear(hidden_dim * 2, attention_dim) for _ in range(num_attention_layers)])

        # Fully connected layer to process combined attention output
        self.fc = nn.Linear(attention_dim * num_attention_layers, output_dim)

    def forward(self, x):
        """
        Forward pass with multi-layer self-attention.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        attention_outputs = []
        
        # Apply multiple self-attention layers
        for attention_layer in self.multi_attention_layers:
            attention_scores = torch.tanh(attention_layer(lstm_output))
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            attention_outputs.append(context_vector)

        # Concatenate outputs from all attention layers
        combined_attention_output = torch.cat(attention_outputs, dim=1)

        # Passing through final fully connected layer
        output = self.fc(self.dropout(combined_attention_output))

        return output


class TransformerAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, num_heads):
        """
        LSTM with Transformer-based attention mechanism.
        """
        super(TransformerAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Transformer-style multi-head attention mechanism
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, dropout=dropout)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward pass with Transformer-based attention mechanism.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Multi-head attention in Transformer style
        attn_output, _ = self.multi_head_attention(lstm_output, lstm_output, lstm_output)

        # Final attention output
        attn_output = attn_output.mean(dim=1)

        # Passing through fully connected layer
        output = self.fc(self.dropout(attn_output))

        return output


class BidirectionalAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        """
        Bidirectional Attention LSTM model.
        """
        super(BidirectionalAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Define a second LSTM layer for bidirectional processing
        self.bidirectional_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward pass with bidirectional LSTM.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Pass output through a second bidirectional LSTM layer
        lstm_output, (hidden, cell) = self.bidirectional_lstm(lstm_output)

        # Concatenate hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Attention mechanism
        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Passing through fully connected layer
        output = self.fc(self.dropout(attn_output))

        return output


class HybridAttentionLSTMWithCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, kernel_sizes, num_filters, dropout):
        """
        A hybrid model combining CNN for feature extraction and Attention LSTM for sequence modeling.
        """
        super(HybridAttentionLSTMWithCNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layers with multiple kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(K, embedding_dim)) for K in kernel_sizes
        ])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def conv_and_pool(self, x, conv_layer):
        """
        Apply convolution and max pooling.
        """
        x = F.relu(conv_layer(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """
        Forward pass with CNN and Attention LSTM.
        """
        embedded = self.dropout(self.embedding(x)).unsqueeze(1)  # Add channel dimension for CNN

        # Apply CNN with different kernel sizes
        conv_outputs = [self.conv_and_pool(embedded, conv) for conv in self.conv_layers]

        # Concatenate all convolution outputs
        conv_output = torch.cat(conv_outputs, 1).unsqueeze(1)

        # Passing through LSTM layer
        lstm_output, (hidden, cell) = self.lstm(conv_output)

        # Concatenating hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Attention mechanism
        attn_output = self.attention_layer(lstm_output, hidden.unsqueeze(0))

        # Passing through fully connected layer
        output = self.fc(self.dropout(attn_output))

        return output


class GatedAttentionLSTM(AttentionLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, gate_dim):
        """
        LSTM with Gated Attention mechanism for better control over focus on sequence elements.
        """
        super(GatedAttentionLSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

        # Gated attention layer
        self.gate = nn.Linear(hidden_dim * 2, gate_dim)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def gated_attention(self, lstm_output):
        """
        Gated attention mechanism for controlled focus on sequence elements.
        """
        gate_output = torch.sigmoid(self.gate(lstm_output))
        attn_output = lstm_output * gate_output
        return attn_output

    def forward(self, x):
        """
        Forward pass with Gated Attention LSTM.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Concatenating hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Gated attention mechanism
        gated_output = self.gated_attention(lstm_output)

        # Attention mechanism
        attn_output = self.attention_layer(gated_output, hidden.unsqueeze(0))

        # Passing through fully connected layer
        output = self.fc(self.dropout(attn_output))

        return output