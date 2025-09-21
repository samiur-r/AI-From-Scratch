# Recurrent Neural Networks (RNNs) Quick Reference

Specialized neural network architectures designed for processing sequential data by maintaining memory through recurrent connections. RNNs excel at modeling temporal dependencies and variable-length sequences, making them fundamental for natural language processing, time series analysis, and any task involving sequential patterns.

## What the Algorithm Does

Recurrent Neural Networks process sequences step-by-step, maintaining an internal hidden state that captures information from previous time steps. Unlike feedforward networks, RNNs have cyclic connections that allow information to persist, enabling them to learn patterns across time and handle variable-length sequences.

**Core concept**: Memory-enabled neural networks that process sequences by maintaining hidden states that encode temporal context and dependencies.

**Algorithm type**: Supervised learning for sequence modeling tasks including sequence classification, sequence-to-sequence mapping, and sequence generation.

**Mathematical Foundation**:
For a basic RNN at time step $t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Where:
- $h_t$ = hidden state at time $t$
- $x_t$ = input at time $t$
- $y_t$ = output at time $t$
- $W$ = weight matrices
- $b$ = bias vectors

**Key Components**:
1. **Hidden State**: Memory that carries information across time steps
2. **Recurrent Connections**: Links between consecutive time steps
3. **Input Processing**: Transformation of sequential inputs
4. **Output Generation**: Predictions at each time step or sequence end
5. **Backpropagation Through Time (BPTT)**: Training algorithm for sequences
6. **Gating Mechanisms**: Advanced variants (LSTM/GRU) for better memory control

**RNN Variants**:
- **Vanilla RNN**: Basic recurrent architecture with simple hidden state updates
- **LSTM**: Long Short-Term Memory with gating mechanisms for long-term dependencies
- **GRU**: Gated Recurrent Unit with simplified gating compared to LSTM
- **Bidirectional RNN**: Processes sequences in both forward and backward directions
- **Stacked RNN**: Multiple RNN layers for increased representational capacity

## When to Use It

### Problem Types
- **Natural Language Processing**: Text classification, language modeling, machine translation
- **Time series forecasting**: Stock prices, weather prediction, demand forecasting
- **Speech recognition**: Converting audio sequences to text
- **Sequence generation**: Text generation, music composition, synthetic data creation
- **Video analysis**: Action recognition, video captioning, temporal event detection
- **Anomaly detection**: Identifying unusual patterns in sequential data
- **Recommendation systems**: Sequential recommendation based on user behavior

### Data Characteristics
- **Sequential structure**: Data with temporal or positional ordering
- **Variable length sequences**: Inputs/outputs of different lengths
- **Temporal dependencies**: Patterns that depend on historical context
- **Contextual relationships**: Meaning depends on surrounding elements
- **Streaming data**: Real-time sequences requiring online processing
- **Long-term patterns**: Dependencies spanning many time steps

### Business Contexts
- **Finance**: Algorithmic trading, risk assessment, fraud detection in transaction sequences
- **Healthcare**: Patient monitoring, medical time series analysis, clinical note processing
- **E-commerce**: Customer behavior modeling, sequential recommendations, price optimization
- **Manufacturing**: Predictive maintenance, quality control, process optimization
- **Media**: Content recommendation, sentiment analysis, automated content generation
- **Telecommunications**: Network traffic analysis, call pattern recognition
- **Transportation**: Route optimization, traffic prediction, autonomous vehicle control

### Comparison with Alternatives
- **Choose RNNs over CNNs** for sequential/temporal data rather than spatial patterns
- **Choose LSTMs/GRUs over Vanilla RNNs** for long sequences with distant dependencies
- **Choose Transformers over RNNs** for very long sequences and when parallelization is important
- **Choose RNNs over MLPs** when temporal order and context matter
- **Choose Bidirectional RNNs over Unidirectional** when future context is available

## Strengths & Weaknesses

### Strengths
- **Temporal modeling**: Natural ability to capture sequential patterns and dependencies
- **Variable length handling**: Can process sequences of different lengths without padding
- **Memory persistence**: Hidden states maintain information across time steps
- **Parameter sharing**: Same weights used across all time steps, reducing parameters
- **Online processing**: Can handle streaming data and real-time applications
- **Interpretable representations**: Hidden states provide insight into temporal patterns
- **Flexible architectures**: Many-to-one, one-to-many, many-to-many configurations
- **Context awareness**: Decisions based on full sequence history

### Weaknesses
- **Vanishing gradient problem**: Difficulty learning long-term dependencies in vanilla RNNs
- **Sequential processing**: Cannot parallelize across time steps during training
- **Computational expense**: Training can be slow for very long sequences
- **Memory limitations**: Hidden state size constrains information storage capacity
- **Gradient instability**: Gradients can explode or vanish during backpropagation
- **Limited bidirectional context**: Standard RNNs only see past, not future
- **Architecture complexity**: LSTM/GRU require careful tuning of gating mechanisms
- **Overfitting tendency**: Can memorize training sequences without generalizing

## Important Hyperparameters

### Architecture Parameters

**hidden_size** (32, 64, 128, 256, 512)
- **Purpose**: Dimensionality of the hidden state vector
- **Range**: 32-1024, commonly 128-512
- **Impact**: Larger = more memory capacity but higher computation
- **Trade-off**: Capacity vs overfitting and computational cost
- **Guidelines**: Start with 128, increase for complex patterns

**num_layers** (1, 2, 3, 4)
- **Purpose**: Number of stacked RNN layers
- **Range**: 1-6, commonly 1-3
- **Impact**: Deeper = more abstract representations but harder to train
- **Recommendations**: Start with 1-2 layers, add more if underfitting

**rnn_type** ('RNN', 'LSTM', 'GRU')
- **RNN**: Simple, fast, suitable for short sequences
- **LSTM**: Best for long sequences with complex dependencies
- **GRU**: Compromise between RNN and LSTM, fewer parameters
- **Choice**: Use LSTM as default, GRU for efficiency, vanilla RNN for simple tasks

**bidirectional** (True/False)
- **Purpose**: Process sequences in both forward and backward directions
- **Benefits**: Access to future context, better representations
- **Cost**: 2× parameters and computation
- **Usage**: Enable when future context is available and helpful

### Training Parameters

**sequence_length** (10, 50, 100, 500)
- **Purpose**: Maximum length of input sequences for training
- **Range**: 10-1000+, depends on task and memory
- **Trade-off**: Longer = more context but higher memory usage
- **Truncation**: Use for very long sequences to manage memory

**batch_size** (16, 32, 64, 128)
- **Purpose**: Number of sequences processed together
- **Range**: 16-256, limited by sequence length and memory
- **RNN specific**: Memory usage scales with batch_size × sequence_length
- **Guidelines**: Smaller batches for longer sequences

**learning_rate** (0.0001-0.01)
- **Range**: 0.0001-0.01, typically 0.001-0.003
- **RNN specific**: Often needs smaller rates than CNNs
- **Scheduling**: Use decay or adaptive schedules for stability
- **LSTM/GRU**: Can handle slightly higher rates than vanilla RNNs

**gradient_clipping** (1.0, 5.0, 10.0)
- **Purpose**: Prevent exploding gradients in RNN training
- **Range**: 1.0-10.0, commonly 5.0
- **Critical**: Essential for stable RNN training
- **Implementation**: Clip by norm or value

### Regularization Parameters

**dropout** (0.2-0.5)
- **Input dropout**: Applied to input vectors
- **Recurrent dropout**: Applied to hidden state connections
- **Output dropout**: Applied before final output layer
- **Guidelines**: 0.2-0.3 for recurrent, 0.5 for output

**weight_decay** (1e-5 to 1e-3)
- **Purpose**: L2 regularization on weights
- **Range**: 1e-5 to 1e-3
- **RNN specific**: Apply carefully to avoid disrupting temporal patterns

### LSTM/GRU Specific Parameters

**forget_bias** (1.0)
- **Purpose**: Initial bias for LSTM forget gate
- **Default**: 1.0 (forget gate starts open)
- **Impact**: Helps with gradient flow in early training

**cell_clip** (None, 3.0, 10.0)
- **Purpose**: Clip LSTM cell state values
- **Usage**: Prevent cell state explosion
- **Range**: 3.0-10.0 when used

### Default Recommendations
```python
# Standard LSTM configuration
lstm_config = {
    'input_size': 100,  # depends on input features
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': False,
    'batch_first': True
}

# Training parameters
training_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'gradient_clip': 5.0,
    'sequence_length': 100,
    'optimizer': 'Adam'
}
```

## Key Assumptions

### Data Assumptions
- **Sequential ordering**: Data has meaningful temporal or positional order
- **Markov property**: Future states depend on current and past states
- **Stationarity**: Statistical properties remain consistent across time
- **Sufficient sequence length**: Enough context for meaningful pattern learning
- **Consistent sampling**: Regular intervals between sequence elements

### Temporal Assumptions
- **Dependency structure**: Earlier elements influence later ones
- **Memory requirements**: Patterns require remembering past information
- **Context window**: Relevant dependencies within learnable time horizon
- **Temporal smoothness**: Gradual changes rather than abrupt discontinuities

### Algorithmic Assumptions
- **Gradient flow**: Backpropagation through time can effectively train the network
- **Hidden state capacity**: Hidden vectors can encode necessary temporal information
- **Parameter sharing**: Same transformation appropriate for all time steps
- **Sequence alignment**: Training sequences properly aligned and meaningful

### Violations and Consequences
- **Non-sequential data**: RNNs provide no benefit over simpler models
- **Very long sequences**: Vanilla RNNs struggle, need LSTM/GRU or Transformers
- **Irregular timing**: May need preprocessing or specialized architectures
- **Insufficient data**: Overfitting to training sequences without generalization
- **Distribution shift**: Performance degrades when deployment sequences differ

### Preprocessing Requirements
- **Sequence padding**: Ensure consistent batch processing
- **Normalization**: Scale features for stable training
- **Tokenization**: Convert text to numerical representations
- **Sequence truncation**: Limit maximum length for memory management
- **Missing value handling**: Impute or mask missing time steps

## Performance Characteristics

### Time Complexity
- **Training**: O(T × B × H²) where T=sequence length, B=batch size, H=hidden size
- **Inference**: O(T × H²) per sequence
- **LSTM/GRU**: ~4× and ~3× computational cost compared to vanilla RNN
- **Sequential bottleneck**: Cannot parallelize across time dimension

### Space Complexity
- **Memory**: O(B × T × H) for storing hidden states during training
- **Parameters**: O(H² + I×H + H×O) where I=input size, O=output size
- **Backpropagation**: Stores hidden states for all time steps
- **Gradient computation**: Memory grows linearly with sequence length

### Scalability
- **Sequence length**: Linear increase in memory and computation
- **Batch size**: Linear scaling limited by memory constraints
- **Hidden size**: Quadratic impact on computation and parameters
- **Depth**: Linear increase in parameters and computation per layer

### Convergence Properties
- **Gradient challenges**: Vanishing/exploding gradients for long sequences
- **Training stability**: Requires gradient clipping and careful initialization
- **Learning rate sensitivity**: More sensitive than feedforward networks
- **Convergence speed**: Generally slower than CNN training

## How to Evaluate & Compare Models

### Appropriate Metrics

**Sequence Classification**:
- **Accuracy**: Overall correctness for sequence-level predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of classification errors
- **Per-class metrics**: Performance on individual sequence types

**Sequence Generation**:
- **Perplexity**: Exponential of cross-entropy loss (language modeling)
- **BLEU Score**: Quality of generated text compared to references
- **ROUGE Score**: Overlap-based evaluation for summarization
- **Human evaluation**: Subjective quality assessment

**Time Series Forecasting**:
- **MSE/RMSE**: Mean squared/root mean squared error
- **MAE**: Mean absolute error (robust to outliers)
- **MAPE**: Mean absolute percentage error
- **Directional accuracy**: Correct prediction of up/down movements

**Sequence Labeling**:
- **Token-level accuracy**: Accuracy at each position
- **Entity-level F1**: F1 score for complete entity recognition
- **Sequence accuracy**: Percentage of perfectly labeled sequences

### Cross-Validation Strategies
- **Temporal splits**: Chronological train/validation/test splits
- **Rolling window**: Moving window validation for time series
- **Blocked CV**: Contiguous blocks to preserve temporal structure
- **Leave-one-sequence-out**: For datasets with distinct sequences

### Baseline Comparisons
- **Simple baselines**: Last value, moving average, linear trend
- **Classical methods**: ARIMA, exponential smoothing for time series
- **N-gram models**: For language tasks
- **Logistic regression**: With temporal features for classification
- **Transformer models**: Modern alternative for sequence tasks

### Statistical Significance
- **Multiple random seeds**: Account for initialization variance
- **Temporal bootstrap**: Preserve temporal structure in resampling
- **Confidence intervals**: Estimate uncertainty in predictions
- **Significance tests**: Compare models while accounting for dependencies

## Practical Usage Guidelines

### Implementation Tips
```python
# PyTorch LSTM implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=False, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)

        # Adjust for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)

        # Pack sequences for efficiency
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths,
                                          batch_first=True, enforce_sorted=False)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use final hidden state for classification
        if self.lstm.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Classification
        output = self.dropout(hidden)
        output = self.fc(output)

        return output

# TensorFlow/Keras implementation
import tensorflow as tf
from tensorflow.keras import layers, models

def create_lstm_model(vocab_size, embed_dim=100, hidden_dim=128,
                      output_dim=1, n_layers=2, dropout=0.3):
    model = models.Sequential([
        layers.Embedding(vocab_size, embed_dim, mask_zero=True),

        layers.LSTM(hidden_dim, return_sequences=True if n_layers > 1 else False,
                   dropout=dropout, recurrent_dropout=dropout),

        # Additional LSTM layers
        *[layers.LSTM(hidden_dim, return_sequences=False if i == n_layers-2 else True,
                     dropout=dropout, recurrent_dropout=dropout)
          for i in range(n_layers-1)],

        layers.Dropout(dropout),
        layers.Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
    ])

    return model
```

### Common Mistakes
- **No gradient clipping**: Leads to exploding gradients and training instability
- **Wrong sequence padding**: Improper handling of variable-length sequences
- **Ignoring temporal order**: Shuffling data inappropriately during training
- **Inappropriate architecture**: Using vanilla RNN for long sequences
- **Poor initialization**: Starting with suboptimal weights for RNN gates
- **Overfitting**: Not using enough regularization for complex sequences
- **Memory management**: Running out of memory with long sequences
- **Incorrect loss computation**: Not masking padded positions properly

### Debugging Strategies
```python
# Monitor gradient norms
def monitor_gradients(model):
    total_norm = 0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    return total_norm

# Visualize hidden states
def plot_hidden_states(model, sample_sequence, layer_idx=0):
    model.eval()
    with torch.no_grad():
        # Get hidden states for each time step
        hidden_states = []
        h = torch.zeros(1, 1, model.hidden_size)
        c = torch.zeros(1, 1, model.hidden_size)

        for token in sample_sequence:
            output, (h, c) = model.lstm(token.unsqueeze(0).unsqueeze(0), (h, c))
            hidden_states.append(h.squeeze().numpy())

    # Plot
    hidden_states = np.array(hidden_states)
    plt.figure(figsize=(12, 8))
    plt.imshow(hidden_states.T, aspect='auto', cmap='viridis')
    plt.xlabel('Time Steps')
    plt.ylabel('Hidden Units')
    plt.title('LSTM Hidden State Evolution')
    plt.colorbar()
    plt.show()

# Check for common issues
def diagnose_rnn_training(losses, gradients):
    if len(losses) > 10:
        recent_improvement = losses[-10] - losses[-1]
        if recent_improvement < 0.01:
            print("⚠️ Loss plateau detected - consider learning rate adjustment")

    if max(gradients) > 10:
        print("⚠️ Large gradients detected - increase gradient clipping")

    if min(gradients) < 1e-6:
        print("⚠️ Very small gradients - check for vanishing gradients")

    if np.std(losses[-20:]) < 0.001:
        print("⚠️ Training appears to have converged")
```

### Production Considerations
- **Sequence batching**: Efficiently batch variable-length sequences
- **State management**: Handle stateful vs stateless processing
- **Memory optimization**: Use sequence packing and gradient checkpointing
- **Inference optimization**: Consider ONNX export and quantization
- **Real-time processing**: Implement streaming inference for online applications
- **Model versioning**: Track different RNN architectures and hyperparameters
- **Monitoring**: Track sequence length distributions and processing times

## Complete Example

### Step 1: Data Preparation
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import re

# What's happening: Creating a sentiment analysis dataset for RNN demonstration
# Why this step: Sentiment analysis showcases RNN's ability to understand
# sequential patterns and context in natural language

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simulate movie review dataset
def generate_synthetic_reviews(n_samples=5000):
    """Generate synthetic movie reviews for demonstration"""

    # Positive review templates
    positive_templates = [
        "This movie was absolutely amazing! The acting was superb and the plot was engaging.",
        "I loved every minute of this film. The characters were well developed and believable.",
        "An outstanding performance by the lead actor. Highly recommend this movie!",
        "Brilliant cinematography and excellent direction. One of the best films I've seen.",
        "Fantastic storyline with unexpected twists. The ending was perfect.",
        "Incredible action sequences and great special effects. Truly entertaining.",
        "The dialogue was witty and the pacing was excellent throughout.",
        "A masterpiece of filmmaking. Every scene was beautifully crafted.",
        "Outstanding performances from the entire cast. Emotionally powerful.",
        "This film exceeded all my expectations. Absolutely worth watching."
    ]

    # Negative review templates
    negative_templates = [
        "This movie was terrible. Poor acting and a confusing plot.",
        "I wasted my time watching this film. The story made no sense.",
        "Boring and predictable. The characters were poorly developed.",
        "Awful direction and terrible screenplay. Complete disappointment.",
        "The worst movie I've seen this year. Avoid at all costs.",
        "Poorly executed with bad special effects. Very disappointing.",
        "The dialogue was cringe-worthy and the pacing was awful.",
        "A complete mess of a film. Nothing worked in this movie.",
        "Terrible performances and a ridiculous plot. Total waste of time.",
        "This film was boring and uninspiring. Definitely not recommended."
    ]

    # Add variety with sentiment words
    positive_words = ['excellent', 'amazing', 'fantastic', 'brilliant', 'outstanding',
                     'wonderful', 'great', 'superb', 'incredible', 'perfect']
    negative_words = ['terrible', 'awful', 'horrible', 'disappointing', 'boring',
                     'bad', 'worst', 'poor', 'ridiculous', 'waste']

    reviews = []
    labels = []

    for i in range(n_samples):
        if i % 2 == 0:  # Positive review
            base_review = np.random.choice(positive_templates)
            # Add random positive words
            for _ in range(np.random.randint(0, 3)):
                word = np.random.choice(positive_words)
                base_review += f" {word.capitalize()}!"
            labels.append(1)
        else:  # Negative review
            base_review = np.random.choice(negative_templates)
            # Add random negative words
            for _ in range(np.random.randint(0, 3)):
                word = np.random.choice(negative_words)
                base_review += f" {word.capitalize()}."
            labels.append(0)

        reviews.append(base_review.lower())

    return reviews, labels

# Generate dataset
print("Generating synthetic movie review dataset...")
reviews, labels = generate_synthetic_reviews(5000)

print(f"Dataset created:")
print(f"  Total reviews: {len(reviews)}")
print(f"  Positive reviews: {sum(labels)}")
print(f"  Negative reviews: {len(labels) - sum(labels)}")

# Show sample reviews
print("\nSample reviews:")
for i in range(5):
    sentiment = "Positive" if labels[i] == 1 else "Negative"
    print(f"  {sentiment}: {reviews[i][:80]}...")

# Text preprocessing
def preprocess_text(text):
    """Basic text preprocessing"""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Split into words
    words = text.split()
    return words

# Build vocabulary
def build_vocabulary(texts, min_freq=2):
    """Build vocabulary from text data"""
    word_counts = Counter()

    for text in texts:
        words = preprocess_text(text)
        word_counts.update(words)

    # Filter by frequency and create vocabulary
    vocab = {'<UNK>': 0, '<PAD>': 1}  # Unknown and padding tokens
    idx = 2

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

# Create vocabulary
print("\nBuilding vocabulary...")
vocab = build_vocabulary(reviews)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Create reverse vocabulary for decoding
idx_to_word = {idx: word for word, idx in vocab.items()}

# Convert text to sequences
def text_to_sequence(text, vocab):
    """Convert text to sequence of token indices"""
    words = preprocess_text(text)
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    return sequence

# Convert all reviews to sequences
sequences = [text_to_sequence(review, vocab) for review in reviews]

# Analyze sequence lengths
seq_lengths = [len(seq) for seq in sequences]
print(f"\nSequence length statistics:")
print(f"  Mean: {np.mean(seq_lengths):.1f}")
print(f"  Median: {np.median(seq_lengths):.1f}")
print(f"  Min: {min(seq_lengths)}")
print(f"  Max: {max(seq_lengths)}")
print(f"  95th percentile: {np.percentile(seq_lengths, 95):.1f}")

# Visualize sequence length distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(seq_lengths, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Lengths')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(seq_lengths)
plt.ylabel('Sequence Length')
plt.title('Sequence Length Box Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    sequences, labels, test_size=0.4, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nDataset splits:")
print(f"  Training: {len(X_train)} sequences")
print(f"  Validation: {len(X_val)} sequences")
print(f"  Test: {len(X_test)} sequences")

# Dataset class for PyTorch
class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    sequences, labels = zip(*batch)

    # Get lengths before padding
    lengths = [len(seq) for seq in sequences]

    # Pad sequences
    sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab['<PAD>'])
    labels = torch.stack(labels)

    return sequences, labels, lengths

# Create data loaders
batch_size = 32
train_dataset = ReviewDataset(X_train, y_train)
val_dataset = ReviewDataset(X_val, y_val)
test_dataset = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Data loaders created with batch size: {batch_size}")
```

### Step 2: RNN Architecture Implementation
```python
# What's happening: Implementing different RNN architectures for comparison
# Why this step: Different RNN types show trade-offs between complexity and performance

class VanillaRNN(nn.Module):
    """Basic RNN for sentiment classification"""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=1, n_layers=1, dropout=0.3):
        super(VanillaRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)

        # Pack sequences for efficiency
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # RNN
        rnn_out, hidden = self.rnn(embedded)

        # Use final hidden state
        if lengths is not None:
            hidden = hidden[-1]  # Take last layer
        else:
            hidden = hidden[-1]

        # Classification
        output = self.dropout(hidden)
        output = self.fc(output)

        return output.squeeze()

class LSTMClassifier(nn.Module):
    """LSTM for sentiment classification"""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=1,
                 n_layers=2, bidirectional=False, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
                           bidirectional=bidirectional)

        # Adjust for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)

        # Pack sequences for efficiency
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use final hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Classification
        output = self.dropout(hidden)
        output = self.fc(output)

        return output.squeeze()

class GRUClassifier(nn.Module):
    """GRU for sentiment classification"""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=1,
                 n_layers=2, bidirectional=False, dropout=0.3):
        super(GRUClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers,
                         batch_first=True, dropout=dropout if n_layers > 1 else 0,
                         bidirectional=bidirectional)

        # Adjust for bidirectional
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_dim, output_dim)

    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)

        # Pack sequences for efficiency
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # GRU
        gru_out, hidden = self.gru(embedded)

        # Use final hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Classification
        output = self.dropout(hidden)
        output = self.fc(output)

        return output.squeeze()

# Model analysis function
def analyze_rnn_model(model, model_name):
    """Analyze RNN model architecture and parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate embedding parameters
    embed_params = model.embedding.weight.numel()

    # Calculate RNN parameters
    if hasattr(model, 'lstm'):
        rnn_params = sum(p.numel() for p in model.lstm.parameters())
    elif hasattr(model, 'gru'):
        rnn_params = sum(p.numel() for p in model.gru.parameters())
    else:
        rnn_params = sum(p.numel() for p in model.rnn.parameters())

    # Calculate FC parameters
    fc_params = model.fc.weight.numel() + model.fc.bias.numel()

    print(f"\n{model_name} Architecture Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Embedding parameters: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
    print(f"  RNN parameters: {rnn_params:,} ({rnn_params/total_params*100:.1f}%)")
    print(f"  FC parameters: {fc_params:,} ({fc_params/total_params*100:.1f}%)")

    # Model size in MB
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"  Model size: {model_size:.2f} MB")

    return total_params

# Create models for comparison
models_config = {
    'Vanilla RNN': VanillaRNN(vocab_size, hidden_dim=64, n_layers=2),
    'LSTM': LSTMClassifier(vocab_size, hidden_dim=128, n_layers=2),
    'GRU': GRUClassifier(vocab_size, hidden_dim=128, n_layers=2),
    'Bidirectional LSTM': LSTMClassifier(vocab_size, hidden_dim=64, n_layers=2, bidirectional=True)
}

print("RNN Architecture Comparison:")
print("=" * 60)

for name, model in models_config.items():
    total_params = analyze_rnn_model(model, name)
    print("-" * 40)
```

### Step 3: Training Process with Gradient Monitoring
```python
# What's happening: Training RNN models with careful monitoring of gradient flow
# What the algorithm is learning: Sequential patterns and context dependencies in text

def train_rnn_model(model, train_loader, val_loader, num_epochs=15, learning_rate=0.001):
    """Train RNN model with comprehensive monitoring"""

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'grad_norms': [], 'learning_rates': []
    }

    best_val_acc = 0.0
    best_model_state = None

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        grad_norms = []

        for batch_idx, (sequences, labels, lengths) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping and monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_norms.append(grad_norm.item())

            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Print progress
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        avg_grad_norm = np.mean(grad_norms)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['grad_norms'].append(avg_grad_norm)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Avg Grad Norm: {avg_grad_norm:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)

        scheduler.step()

        # Early stopping check
        if len(history['val_loss']) > 5:
            if all(history['val_loss'][-1] >= history['val_loss'][-5:-1]):
                print("Early stopping triggered - validation loss not improving")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

    return model, history

# Train different models
training_results = {}

for name, model in models_config.items():
    print(f"\n{'='*20} Training {name} {'='*20}")

    # Reset model
    model = model.__class__(vocab_size, **{
        'hidden_dim': 128 if 'Bidirectional' not in name else 64,
        'n_layers': 2,
        'bidirectional': 'Bidirectional' in name
    })

    # Train model
    trained_model, history = train_rnn_model(
        model, train_loader, val_loader,
        num_epochs=15, learning_rate=0.001
    )

    training_results[name] = {
        'model': trained_model,
        'history': history
    }

    print(f"Completed training {name}")
    print("="*60)

# What RNNs learned during training:
print(f"\nRNN Learning Process Analysis:")
print("Sequential pattern recognition concepts learned:")
print("• Word embeddings: Dense representations capturing semantic similarity")
print("• Context modeling: Hidden states encode information from previous words")
print("• Sentiment indicators: Recognition of positive/negative sentiment words")
print("• Negation handling: Understanding how 'not' affects sentiment")
print("• Long-term dependencies: Maintaining context across entire reviews")
print("• Sequence composition: Combining word-level features into document-level predictions")
```

### Step 4: Evaluation and Analysis
```python
# What's happening: Comprehensive evaluation of RNN variants on sentiment analysis
# How to interpret results: Comparison reveals strengths of different RNN architectures

def evaluate_rnn_model(model, test_loader, model_name):
    """Comprehensive RNN model evaluation"""
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences, lengths)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)

    return {
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities)
    }

# Evaluate all models
evaluation_results = {}

for name, result in training_results.items():
    model = result['model']
    eval_result = evaluate_rnn_model(model, test_loader, name)
    evaluation_results[name] = eval_result

    print(f"\n{name} Test Results:")
    print(f"  Test Accuracy: {eval_result['accuracy']:.2f}%")

# Visualize training progress
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Training curves
for idx, (name, result) in enumerate(training_results.items()):
    if idx >= 2:  # Show only first 2 models for space
        break

    history = result['history']

    # Loss curves
    ax1 = axes[0, idx]
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f'{name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = axes[1, idx]
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'{name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Gradient norms
    ax3 = axes[2, idx]
    ax3.plot(history['grad_norms'], label='Gradient Norm', linewidth=2, color='red')
    ax3.set_title(f'{name} - Gradient Norms')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Norm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model comparison summary
print("\nModel Comparison Summary:")
print("=" * 80)
print(f"{'Model':<20} {'Test Acc':<10} {'Parameters':<12} {'Overfitting':<12}")
print("-" * 80)

for name in models_config.keys():
    model = training_results[name]['model']
    history = training_results[name]['history']
    test_acc = evaluation_results[name]['accuracy']

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate overfitting
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    overfitting = final_train_acc - final_val_acc

    print(f"{name:<20} {test_acc:<10.2f} {total_params:<12,} {overfitting:<12.2f}")

# Detailed confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, eval_result) in enumerate(evaluation_results.items()):
    cm = confusion_matrix(eval_result['targets'], eval_result['predictions'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    axes[idx].set_title(f'{name}\nAccuracy: {eval_result["accuracy"]:.2f}%')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Find best model for detailed analysis
best_model_name = max(evaluation_results.keys(),
                     key=lambda x: evaluation_results[x]['accuracy'])
best_eval = evaluation_results[best_model_name]
best_model = training_results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_eval['accuracy']:.2f}%")

# Classification report
from sklearn.metrics import classification_report
print(f"\nDetailed Classification Report - {best_model_name}:")
print(classification_report(best_eval['targets'], best_eval['predictions'],
                          target_names=['Negative', 'Positive']))

# Confidence analysis
probabilities = best_eval['probabilities']
predictions = best_eval['predictions']
targets = best_eval['targets']

# Analyze prediction confidence
correct_predictions = (predictions == targets)
correct_confidence = probabilities[correct_predictions.astype(bool)]
incorrect_confidence = probabilities[~correct_predictions.astype(bool)]

# Adjust confidence for negative predictions
confidence_scores = np.where(predictions == 1, probabilities, 1 - probabilities)
correct_conf_adj = confidence_scores[correct_predictions.astype(bool)]
incorrect_conf_adj = confidence_scores[~correct_predictions.astype(bool)]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist([correct_conf_adj, incorrect_conf_adj], bins=20, alpha=0.7,
         label=['Correct', 'Incorrect'], color=['green', 'red'])
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(range(len(confidence_scores)), confidence_scores,
           c=correct_predictions, alpha=0.6, cmap='RdYlGn')
plt.xlabel('Sample Index')
plt.ylabel('Confidence')
plt.title('Confidence vs Correctness')
plt.colorbar(label='Correct')

plt.subplot(1, 3, 3)
# ROC-like curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(targets, probabilities)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nConfidence Analysis:")
print(f"  Mean confidence (correct): {correct_conf_adj.mean():.3f}")
print(f"  Mean confidence (incorrect): {incorrect_conf_adj.mean():.3f}")
print(f"  AUC Score: {roc_auc:.3f}")
```

### Step 5: Hidden State Analysis and Interpretation
```python
# What's happening: Analyzing what RNN hidden states learn and how they evolve
# How to interpret results: Hidden state patterns reveal sequential processing

def analyze_hidden_states(model, sample_texts, vocab, max_length=50):
    """Analyze hidden state evolution for sample texts"""
    model.eval()
    model = model.to(device)

    results = []

    for text in sample_texts:
        # Preprocess text
        sequence = text_to_sequence(text, vocab)[:max_length]
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

        # Forward pass with hidden state tracking
        with torch.no_grad():
            embedded = model.embedding(input_tensor)

            # Manual forward pass to capture hidden states
            hidden_states = []
            if hasattr(model, 'lstm'):
                h = torch.zeros(model.lstm.num_layers, 1, model.hidden_dim).to(device)
                c = torch.zeros(model.lstm.num_layers, 1, model.hidden_dim).to(device)

                for i in range(embedded.size(1)):
                    input_step = embedded[:, i:i+1, :]
                    output, (h, c) = model.lstm(input_step, (h, c))
                    hidden_states.append(h[-1].squeeze().cpu().numpy())

            # Final prediction
            final_output = model.fc(model.dropout(h[-1]))
            probability = torch.sigmoid(final_output).item()

        results.append({
            'text': text,
            'sequence': sequence,
            'hidden_states': hidden_states,
            'probability': probability,
            'predicted_sentiment': 'Positive' if probability > 0.5 else 'Negative'
        })

    return results

# Sample texts for analysis
sample_texts = [
    "this movie was absolutely amazing and fantastic",
    "the film was terrible and completely boring",
    "i loved the first half but the ending was disappointing",
    "not bad but not great either just average",
    "incredible acting superb direction outstanding movie"
]

print("Analyzing Hidden State Evolution:")
print("=" * 50)

hidden_analysis = analyze_hidden_states(best_model, sample_texts, vocab)

for i, result in enumerate(hidden_analysis):
    print(f"\nSample {i+1}: {result['text']}")
    print(f"Predicted: {result['predicted_sentiment']} (prob: {result['probability']:.3f})")

    # Plot hidden state evolution
    if result['hidden_states']:
        hidden_matrix = np.array(result['hidden_states'])

        plt.figure(figsize=(12, 8))

        # Hidden state heatmap
        plt.subplot(2, 2, 1)
        plt.imshow(hidden_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Activation')
        plt.xlabel('Time Steps (Words)')
        plt.ylabel('Hidden Units')
        plt.title(f'Hidden State Evolution\n{result["predicted_sentiment"]} (prob: {result["probability"]:.3f})')

        # Average hidden state magnitude over time
        plt.subplot(2, 2, 2)
        avg_magnitude = np.mean(np.abs(hidden_matrix), axis=1)
        plt.plot(avg_magnitude, linewidth=2)
        plt.xlabel('Time Steps')
        plt.ylabel('Average |Hidden State|')
        plt.title('Hidden State Magnitude')
        plt.grid(True, alpha=0.3)

        # Hidden state variance over time
        plt.subplot(2, 2, 3)
        hidden_var = np.var(hidden_matrix, axis=1)
        plt.plot(hidden_var, linewidth=2, color='red')
        plt.xlabel('Time Steps')
        plt.ylabel('Hidden State Variance')
        plt.title('Hidden State Diversity')
        plt.grid(True, alpha=0.3)

        # Final hidden state distribution
        plt.subplot(2, 2, 4)
        final_hidden = hidden_matrix[-1]
        plt.hist(final_hidden, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Hidden Unit Value')
        plt.ylabel('Frequency')
        plt.title('Final Hidden State Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Attention-like analysis: which words contribute most to final prediction
def analyze_word_importance(model, text, vocab):
    """Analyze word importance using gradient-based attribution"""
    model.eval()
    model = model.to(device)

    # Preprocess
    sequence = text_to_sequence(text, vocab)
    input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    input_tensor.requires_grad_(False)

    # Get embeddings
    embeddings = model.embedding(input_tensor)
    embeddings.requires_grad_(True)

    # Forward pass
    if hasattr(model, 'lstm'):
        lstm_out, (hidden, cell) = model.lstm(embeddings)
        output = model.fc(model.dropout(hidden[-1]))
    else:
        # Handle other RNN types
        rnn_out, hidden = model.rnn(embeddings)
        output = model.fc(model.dropout(hidden[-1]))

    # Backward pass
    output.backward()

    # Get gradients
    gradients = embeddings.grad.squeeze().cpu().numpy()
    importance_scores = np.linalg.norm(gradients, axis=1)

    # Get words
    words = [idx_to_word.get(idx, '<UNK>') for idx in sequence]

    return words, importance_scores

print(f"\nWord Importance Analysis using {best_model_name}:")
print("=" * 50)

for text in sample_texts[:3]:
    words, importance = analyze_word_importance(best_model, text, vocab)

    print(f"\nText: {text}")
    print("Word importance scores:")

    # Sort by importance
    word_importance = list(zip(words, importance))
    word_importance.sort(key=lambda x: x[1], reverse=True)

    for word, score in word_importance[:10]:  # Top 10 words
        print(f"  {word}: {score:.3f}")

    # Visualize
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(words)), importance, alpha=0.7)
    plt.xlabel('Word Position')
    plt.ylabel('Importance Score')
    plt.title(f'Word Importance Scores\n"{text}"')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Error analysis: examine misclassified examples
print(f"\nError Analysis:")
print("=" * 30)

# Find misclassified examples
misclassified_indices = np.where(best_eval['predictions'] != best_eval['targets'])[0]
print(f"Total misclassified: {len(misclassified_indices)} out of {len(best_eval['targets'])}")

# Show some misclassified examples
print(f"\nMisclassified Examples:")
for i in range(min(5, len(misclassified_indices))):
    idx = misclassified_indices[i]
    original_idx = idx  # Note: this would need proper mapping in real scenario

    actual = 'Positive' if best_eval['targets'][idx] == 1 else 'Negative'
    predicted = 'Positive' if best_eval['predictions'][idx] == 1 else 'Negative'
    confidence = best_eval['probabilities'][idx]

    print(f"\nExample {i+1}:")
    print(f"  Actual: {actual}")
    print(f"  Predicted: {predicted}")
    print(f"  Confidence: {confidence:.3f}")
    # Note: In a real scenario, you'd want to store original texts for analysis

# Model insights summary
print(f"\nRNN Model Insights:")
print("=" * 40)
print(f"Best performing model: {best_model_name}")
print(f"Key findings:")
print(f"• LSTM/GRU significantly outperform vanilla RNN for sentiment analysis")
print(f"• Bidirectional processing can improve performance by using future context")
print(f"• Hidden states effectively encode sentiment progression through sequences")
print(f"• Gradient clipping is essential for stable RNN training")
print(f"• Word importance analysis reveals model focus on sentiment-bearing words")
```

### Step 6: Production Deployment and Optimization
```python
# What's happening: Preparing RNN model for production deployment
# How to use in practice: Optimization techniques and monitoring for real-world applications

import pickle
import time
from torch.jit import script

def optimize_rnn_for_production(model, vocab, sample_input):
    """Optimize RNN model for production deployment"""

    # Model quantization
    print("Applying dynamic quantization...")
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.GRU, nn.Linear}, dtype=torch.qint8
    )

    # TorchScript compilation
    print("Compiling to TorchScript...")
    model.eval()
    example_input = torch.randint(0, len(vocab), (1, 20))  # Example sequence

    try:
        model_scripted = torch.jit.trace(model, example_input)
    except:
        print("Trace failed, using script mode...")
        model_scripted = torch.jit.script(model)

    return {
        'original': model,
        'quantized': model_quantized,
        'scripted': model_scripted
    }

def benchmark_rnn_inference(models_dict, test_sequences, num_runs=100):
    """Benchmark inference speed for RNN variants"""
    results = {}

    # Prepare test batch
    test_batch = test_sequences[:32]  # Use batch of 32
    max_len = max(len(seq) for seq in test_batch)

    # Pad sequences
    padded_batch = []
    for seq in test_batch:
        padded = seq + [vocab['<PAD>']] * (max_len - len(seq))
        padded_batch.append(padded)

    test_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)

    for name, model in models_dict.items():
        model.eval()
        model = model.to(device)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_tensor)

        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_tensor)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # ms

        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        results[name] = {
            'inference_time_ms': avg_time,
            'model_size_mb': model_size,
            'throughput_samples_per_sec': len(test_batch) * 1000 / avg_time
        }

        print(f"{name}:")
        print(f"  Inference time: {avg_time:.2f} ms/batch")
        print(f"  Model size: {model_size:.2f} MB")
        print(f"  Throughput: {results[name]['throughput_samples_per_sec']:.1f} samples/sec")
        print()

    return results

# Optimize best model for production
optimized_models = optimize_rnn_for_production(best_model, vocab, None)

print("Production Model Optimization:")
print("=" * 50)

# Benchmark optimized models
benchmark_results = benchmark_rnn_inference(optimized_models, X_test[:100])

# Save model and artifacts for deployment
model_save_path = './models/'
import os
os.makedirs(model_save_path, exist_ok=True)

# Save complete model package
model_package = {
    'model_state_dict': best_model.state_dict(),
    'model_class': best_model.__class__.__name__,
    'model_config': {
        'vocab_size': vocab_size,
        'hidden_dim': best_model.hidden_dim,
        'n_layers': best_model.n_layers,
        'bidirectional': getattr(best_model, 'bidirectional', False)
    },
    'vocab': vocab,
    'idx_to_word': idx_to_word,
    'test_accuracy': best_eval['accuracy'],
    'preprocessing_info': {
        'pad_token': '<PAD>',
        'unk_token': '<UNK>',
        'max_sequence_length': 100
    }
}

torch.save(model_package, f"{model_save_path}best_rnn_model.pth")

# Save vocabulary separately for fast loading
with open(f"{model_save_path}vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)

print(f"Model artifacts saved to {model_save_path}")

# Production inference class
class RNNSentimentClassifier:
    """Production-ready RNN sentiment classifier"""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)

        # Load model package
        package = torch.load(model_path, map_location=self.device)

        # Reconstruct model
        if package['model_class'] == 'LSTMClassifier':
            self.model = LSTMClassifier(**package['model_config'])
        elif package['model_class'] == 'GRUClassifier':
            self.model = GRUClassifier(**package['model_config'])
        # Add other model types as needed

        self.model.load_state_dict(package['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)

        self.vocab = package['vocab']
        self.idx_to_word = package['idx_to_word']
        self.max_length = package['preprocessing_info']['max_sequence_length']

    def preprocess_text(self, text):
        """Preprocess input text"""
        # Basic preprocessing
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()

        # Convert to sequence
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

        # Truncate or pad
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        return sequence

    def predict(self, text):
        """Predict sentiment for input text"""
        # Preprocess
        sequence = self.preprocess_text(text)

        # Convert to tensor
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()

        return {
            'sentiment': 'positive' if probability > 0.5 else 'negative',
            'confidence': probability if probability > 0.5 else 1 - probability,
            'raw_probability': probability
        }

    def predict_batch(self, texts):
        """Predict sentiment for batch of texts"""
        sequences = [self.preprocess_text(text) for text in texts]

        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences) if sequences else 0
        padded_sequences = []

        for seq in sequences:
            padded = seq + [self.vocab['<PAD>']] * (max_len - len(seq))
            padded_sequences.append(padded)

        # Convert to tensor
        input_tensor = torch.tensor(padded_sequences, dtype=torch.long).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()

        results = []
        for prob in probabilities:
            results.append({
                'sentiment': 'positive' if prob > 0.5 else 'negative',
                'confidence': prob if prob > 0.5 else 1 - prob,
                'raw_probability': prob
            })

        return results

# Example usage
print("\nProduction Inference Example:")
print("=" * 40)

# Create inference instance
classifier = RNNSentimentClassifier(f"{model_save_path}best_rnn_model.pth", device=str(device))

# Test with sample texts
test_texts = [
    "This movie was absolutely fantastic and amazing!",
    "Terrible film with poor acting and boring plot.",
    "Not bad but could have been better.",
    "Outstanding performance by all actors. Highly recommended!",
    "The worst movie I have ever seen in my life."
]

print("Sample predictions:")
for text in test_texts:
    result = classifier.predict(text)
    print(f"Text: {text[:50]}...")
    print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
    print()

# Batch prediction example
batch_results = classifier.predict_batch(test_texts)
print("Batch prediction results:")
for text, result in zip(test_texts, batch_results):
    print(f"'{text[:30]}...' -> {result['sentiment']} ({result['confidence']:.3f})")

# Production deployment checklist
print("\nProduction Deployment Checklist:")
print("=" * 50)

checklist = {
    "Model Optimization": [
        "✅ Model quantization applied for faster inference",
        "✅ TorchScript compilation for deployment optimization",
        "✅ Batch processing support for better throughput",
        "⚠️ Consider ONNX export for cross-platform deployment",
        "⚠️ Implement model ensemble for improved accuracy"
    ],
    "Text Preprocessing": [
        "✅ Robust text cleaning and normalization",
        "✅ Vocabulary handling with UNK tokens",
        "✅ Sequence length management and padding",
        "⚠️ Add support for multiple languages",
        "⚠️ Implement advanced tokenization (subword, BPE)"
    ],
    "Performance Monitoring": [
        "⚠️ Log inference times and throughput metrics",
        "⚠️ Monitor prediction confidence distributions",
        "⚠️ Track accuracy on validation samples",
        "⚠️ Set up alerts for performance degradation",
        "⚠️ Monitor memory usage and sequence length patterns"
    ],
    "API and Serving": [
        "⚠️ Implement REST API with proper error handling",
        "⚠️ Add input validation and sanitization",
        "⚠️ Implement rate limiting and authentication",
        "⚠️ Set up load balancing for multiple instances",
        "⚠️ Add model versioning and A/B testing capabilities"
    ],
    "Data Pipeline": [
        "⚠️ Implement continuous data collection and labeling",
        "⚠️ Set up model retraining pipeline",
        "⚠️ Add data drift detection and monitoring",
        "⚠️ Implement feedback collection mechanism",
        "⚠️ Regular vocabulary updates and expansion"
    ]
}

for category, items in checklist.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

# Performance expectations
print(f"\nPerformance Expectations:")
print(f"• Expected accuracy: {best_eval['accuracy']:.1f}% ± 3%")
print(f"• Inference time: {benchmark_results['original']['inference_time_ms']:.1f}ms per batch (32 samples)")
print(f"• Throughput: {benchmark_results['original']['throughput_samples_per_sec']:.0f} samples/second")
print(f"• Memory usage: {benchmark_results['original']['model_size_mb']:.1f}MB model size")
print(f"• Sequence length: Up to {100} tokens efficiently")

print(f"\nRNN deployment guide completed successfully!")
```

## Summary

### Key Takeaways

- **Sequential memory**: RNNs maintain hidden states that encode temporal context across sequences
- **Architecture evolution**: Vanilla RNN → LSTM → GRU → Transformers, each addressing specific limitations
- **Gradient management**: Gradient clipping essential for stable training, LSTM/GRU help with vanishing gradients
- **Bidirectional processing**: Can significantly improve performance when future context is available
- **Variable length handling**: Natural ability to process sequences of different lengths
- **Context sensitivity**: Performance heavily depends on understanding sequential dependencies

### Quick Reference

```python
# Standard LSTM setup for sequence classification
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 output_dim=1, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = self.dropout(hidden[-1])
        return self.fc(output)

# Training essentials
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### When to Choose RNNs

- **Sequential data**: Time series, text, speech, or any temporally ordered data
- **Variable length inputs**: When sequences have different lengths naturally
- **Context dependency**: When understanding requires memory of previous elements
- **Real-time processing**: Streaming applications where data arrives sequentially
- **Limited computational resources**: When Transformers are too expensive

### When to Choose Alternatives

- **Very long sequences**: Transformers handle longer contexts better
- **Parallelization important**: CNNs or Transformers for faster training
- **Non-sequential data**: MLPs for tabular data, CNNs for images
- **Simple patterns**: Traditional ML for basic time series forecasting
- **State-of-the-art NLP**: Transformer-based models (BERT, GPT) for most language tasks

Recurrent Neural Networks provide the foundation for understanding sequential modeling in deep learning. While Transformers have largely replaced RNNs in many NLP applications, RNNs remain valuable for understanding temporal dynamics, real-time processing, and resource-constrained environments. Master RNN concepts before exploring more advanced sequential architectures.