# Model Architectures

This section details the various model architectures implemented in the sentiment analysis project. The models include traditional deep learning models, advanced Transformer-based architectures, and custom hybrid models that blend different techniques to achieve optimal performance.

## 1. LSTM (Long Short-Term Memory)

- **File**: `models/architectures/lstm.py`
- LSTM is a type of recurrent neural network (RNN) that captures long-term dependencies in sequential data.
- The model architecture includes embedding layers, dropout layers, and dense layers for sentiment classification.

## 2. BERT (Bidirectional Encoder Representations from Transformers)

- **File**: `models/architectures/bert.py`
- BERT is a Transformer-based model that uses a bidirectional training mechanism for understanding context in text sequences.
- Fine-tuned on the sentiment dataset for classification.

## 3. Transformer

- **File**: `models/architectures/transformer.py`
- A self-attention-based architecture that handles sequential data without the limitations of RNNs.
- This model is particularly powerful for parallelizing sentiment analysis tasks.

## 4. Custom LSTM

- **File**: `models/custom/custom_lstm.py`
- A modified LSTM model with optimizations like attention mechanisms or custom hyperparameters to improve sentiment classification accuracy.

## 5. Custom Transformer

- **File**: `models/custom/custom_transformer.py`
- Customized Transformer architecture that combines standard attention mechanisms with domain-specific tuning for sentiment analysis.

## 6. Hybrid Models

- **File**: `models/custom/hybrid_model.py`
- This model combines multiple architectures, such as LSTM with Transformer, to capture both local dependencies (LSTM) and global context (Transformer).

## 7. Attention-based LSTM

- **File**: `models/custom/attention_lstm.py`
- LSTM with an attention mechanism added to improve focus on important parts of the input sequence, which enhances the overall sentiment classification.

## 8. R-based Models

- **File**: `models/r_models.R`
- Various models implemented using R for specific sentiment analysis tasks, including classical models like `randomForest` and `glmnet`.
