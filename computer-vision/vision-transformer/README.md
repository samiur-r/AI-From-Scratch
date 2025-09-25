# Vision Transformer (ViT) Quick Reference

Vision Transformer (ViT) is a neural network architecture that applies the transformer model directly to sequences of image patches for image classification tasks, demonstrating that pure attention mechanisms can achieve state-of-the-art results without convolutional layers.

## What the Algorithm Does

Vision Transformer treats image classification as a sequence-to-sequence problem by:

- **Patch Embedding**: Dividing input images into fixed-size patches and linearly embedding them
- **Position Encoding**: Adding learnable position embeddings to retain spatial information
- **Transformer Encoder**: Processing patch embeddings through multiple self-attention layers
- **Classification Head**: Using a classification token ([CLS]) to make final predictions

The model learns to attend to relevant image patches and their relationships without explicit convolutional inductive biases.

## When to Use It

### Problem Types
- **Image classification**: Primary strength, especially with large datasets
- **Fine-grained classification**: Excels at detailed visual distinctions
- **Transfer learning**: Pre-trained models work well across domains
- **Multi-modal tasks**: Can be combined with text transformers

### Data Characteristics
- **Large datasets**: Requires substantial training data (millions of images)
- **High-resolution images**: Benefits from detailed visual information
- **Diverse visual patterns**: Learns complex spatial relationships
- **Sufficient computational resources**: Needs significant GPU memory and compute

### Business Contexts
- Medical imaging with large annotated datasets
- Content moderation at scale
- Autonomous vehicle perception systems
- Large-scale image search and retrieval

### Comparison with Alternatives
- **vs CNNs**: Better scalability and performance on large datasets, worse on small datasets
- **vs ResNet**: Superior accuracy with sufficient data, more computationally expensive
- **vs EfficientNet**: Higher accuracy ceiling, less parameter efficient
- **vs Hybrid models**: Pure attention vs attention + convolution trade-offs

## Strengths & Weaknesses

### Strengths
- **Scalability**: Performance improves consistently with larger datasets and models
- **Long-range dependencies**: Self-attention captures global image relationships
- **Transfer learning**: Pre-trained models generalize well across domains
- **Interpretability**: Attention maps provide insights into model decisions
- **Architecture flexibility**: Can be adapted for various vision tasks

### Weaknesses
- **Data hungry**: Requires large datasets to outperform CNNs
- **Computational cost**: High memory and compute requirements
- **Lack of inductive bias**: No built-in translation invariance or local connectivity
- **Training instability**: Sensitive to initialization and hyperparameters
- **Limited spatial hierarchy**: Processes fixed patch size throughout

## Important Hyperparameters

### Architecture Parameters
- **Patch size**: Size of image patches (typically 16×16 or 32×32)
  - Range: 8×8 to 32×32
  - Tuning: Smaller patches = more detail but higher compute cost
  - Default: 16×16 for most applications

- **Hidden dimension (d_model)**: Transformer embedding dimension
  - Range: 384-1024 for standard models
  - Tuning: Larger = more capacity but slower training
  - Default: 768 for ViT-Base

- **Number of layers**: Transformer encoder depth
  - Range: 6-24 layers
  - Tuning: More layers = higher capacity, risk of overfitting
  - Default: 12 layers for ViT-Base

- **Number of heads**: Multi-head attention heads
  - Range: 8-16 heads
  - Tuning: More heads = more attention patterns
  - Default: 12 heads for ViT-Base

### Training Parameters
- **Learning rate**:
  - Range: 1e-5 to 1e-3
  - Default: 3e-4 with warmup
  - Tuning: Lower for fine-tuning, higher for training from scratch

- **Batch size**:
  - Range: 256-4096 (large batches crucial)
  - Default: 1024-2048
  - Tuning: Larger batches improve training stability

- **Weight decay**:
  - Range: 0.01-0.3
  - Default: 0.1
  - Tuning: Higher values prevent overfitting

## Key Assumptions

### Data Assumptions
- **Sufficient scale**: Assumes access to large training datasets (>100k images)
- **Image quality**: Assumes reasonable image resolution and quality
- **Label quality**: Assumes clean, accurate labels for supervised learning
- **Computational resources**: Assumes access to modern GPU hardware

### Architectural Assumptions
- **Patch independence**: Treats image patches as independent tokens initially
- **Global attention**: Assumes all patches can potentially interact
- **Fixed resolution**: Typically trained and evaluated on fixed image sizes
- **Translation invariance**: Must be learned rather than built-in

### Violations and Solutions
- **Small datasets**: Use pre-trained models and aggressive data augmentation
- **Limited compute**: Use smaller variants (ViT-Small, ViT-Tiny)
- **Variable input sizes**: Interpolate position embeddings or use adaptive pooling
- **Memory constraints**: Use gradient checkpointing and mixed precision training

## Performance Characteristics

### Time Complexity
- **Training**: O(N × L × d² + N × P² × d) per layer
  - N: sequence length (patches), L: layers, d: hidden dim, P: patch size
- **Inference**: O(N × L × d² + N × P²) per image
- **Self-attention**: Quadratic in sequence length (number of patches)

### Space Complexity
- **Memory usage**: 8-32GB GPU memory for training (batch size dependent)
- **Model size**: 80MB (ViT-Small) to 1.3GB (ViT-Huge)
- **Activation memory**: Grows with input resolution and batch size

### Convergence Properties
- **Training epochs**: 90-300 epochs for ImageNet from scratch
- **Warmup period**: Critical for training stability (10-40 epochs)
- **Learning rate decay**: Cosine annealing typically works best

## Evaluation & Compare Models

### Appropriate Metrics
- **Top-1 accuracy**: Primary metric for classification tasks
- **Top-5 accuracy**: Secondary metric for large-scale datasets
- **FLOPs**: Computational efficiency comparison
- **Parameters**: Model size comparison
- **Throughput**: Images per second for inference speed

### Cross-validation Strategies
- **Standard splitting**: 80/10/10 train/val/test split
- **Stratified sampling**: Maintain class distribution across splits
- **Time-based splits**: For temporal datasets
- **K-fold validation**: For smaller datasets (though not ideal for ViT)

### Baseline Comparisons
- **ResNet-50/101**: Standard CNN baseline
- **EfficientNet-B3/B4**: Efficient CNN comparison
- **ConvNeXt**: Modern CNN architecture
- **Hybrid ViT**: Vision transformer with CNN stem

### Statistical Significance
- **Multiple runs**: Average results over 3-5 training runs
- **Confidence intervals**: Report 95% confidence intervals
- **Statistical tests**: Use t-tests for comparing model variants

## Practical Usage Guidelines

### Implementation Tips
- **Use pre-trained models**: Start with models pre-trained on large datasets
- **Proper data augmentation**: RandAugment, MixUp, CutMix are crucial
- **Learning rate warmup**: Essential for training stability
- **Large batch sizes**: Use gradient accumulation if memory limited
- **Mixed precision**: Use FP16 to reduce memory usage

### Common Mistakes
- **Training from scratch on small datasets**: Always use pre-trained models
- **Insufficient data augmentation**: ViT overfits easily without augmentation
- **Wrong learning rate schedule**: Warmup and decay are critical
- **Inadequate regularization**: Weight decay and dropout are important
- **Ignoring position embeddings**: Proper handling for different input sizes

### Debugging Strategies
- **Attention visualization**: Plot attention maps to understand model behavior
- **Loss curves**: Monitor for overfitting and training instability
- **Gradient norms**: Check for gradient explosion or vanishing
- **Layer-wise analysis**: Examine activations at different depths
- **Patch importance**: Analyze which patches contribute most to predictions

### Production Considerations
- **Model compression**: Use knowledge distillation or pruning
- **Quantization**: INT8 quantization for deployment
- **Batch optimization**: Optimize batch sizes for throughput
- **Caching**: Cache patch embeddings for repeated inference

## Complete Example with Step-by-Step Explanation

### Step 1: Data Preparation
```python
# What's happening: Loading and preprocessing image classification dataset
# Why this step: ViT requires properly formatted image patches and strong augmentation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def load_and_preprocess_data(data_dir, img_size=224, batch_size=32):
    """Load and preprocess image classification data"""

    # Data augmentation for ViT (crucial for good performance)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        f'{data_dir}/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        f'{data_dir}/val',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator

def create_tf_dataset(generator, buffer_size=1000):
    """Convert generator to tf.data.Dataset for better performance"""
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 224, 224, 3], [None, generator.num_classes])
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

# What's happening: Advanced augmentation techniques for ViT
# Why this step: ViT benefits significantly from strong augmentation
def mixup(images, labels, alpha=0.2):
    """Apply MixUp augmentation"""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    lambda_val = tf.random.uniform([], 0, alpha)

    mixed_images = lambda_val * images + (1 - lambda_val) * shuffled_images
    mixed_labels = lambda_val * labels + (1 - lambda_val) * shuffled_labels

    return mixed_images, mixed_labels

# Load data
train_gen, val_gen = load_and_preprocess_data('path/to/dataset')
num_classes = train_gen.num_classes

print(f"Number of classes: {num_classes}")
print(f"Training batches: {len(train_gen)}")
print(f"Validation batches: {len(val_gen)}")
```

### Step 2: Vision Transformer Architecture
```python
# What's happening: Implementing Vision Transformer from scratch
# Why this architecture: Pure attention mechanism for image classification

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.projection = layers.Dense(embed_dim)
        self.flatten = layers.Reshape((-1, embed_dim))

    def call(self, images):
        # Extract patches
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape and project patches
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        patches = self.projection(patches)

        return patches

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)
        self.projection = layers.Dense(embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Generate Q, K, V
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)

        attention = tf.matmul(weights, v)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, [batch_size, seq_len, self.embed_dim])

        return self.projection(attention)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])

        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Multi-head self-attention with residual connection
        attention_output = self.attention(inputs)
        attention_output = self.dropout(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)

        # MLP with residual connection
        mlp_output = self.mlp(out1, training=training)
        return self.layernorm2(out1 + mlp_output)

def build_vit(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    mlp_dim=3072,
    dropout_rate=0.1
):
    """Build Vision Transformer model"""

    # Input layer
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Patch embedding
    patches = PatchEmbedding(patch_size, embed_dim)(inputs)

    # Add classification token
    batch_size = tf.shape(patches)[0]
    cls_token = tf.Variable(
        tf.random.normal([1, 1, embed_dim]),
        trainable=True,
        name="cls_token"
    )
    cls_tokens = tf.tile(cls_token, [batch_size, 1, 1])
    patches = tf.concat([cls_tokens, patches], axis=1)

    # Add position embeddings
    num_patches = (image_size // patch_size) ** 2
    pos_embedding = layers.Embedding(
        input_dim=num_patches + 1,
        output_dim=embed_dim
    )

    positions = tf.range(start=0, limit=num_patches + 1, delta=1)
    encoded_patches = patches + pos_embedding(positions)

    # Transformer encoder
    for _ in range(num_layers):
        encoded_patches = TransformerBlock(
            embed_dim, num_heads, mlp_dim, dropout_rate
        )(encoded_patches)

    # Classification head
    representation = layers.LayerNormalization()(encoded_patches)
    cls_token_final = representation[:, 0]  # Use [CLS] token

    outputs = layers.Dense(num_classes, activation="softmax")(cls_token_final)

    model = keras.Model(inputs, outputs)
    return model

# What's happening: Building ViT model with standard configuration
# Why these parameters: ViT-Base configuration proven effective on ImageNet
model = build_vit(
    image_size=224,
    patch_size=16,
    num_classes=num_classes,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    mlp_dim=3072,
    dropout_rate=0.1
)

print(f"Model parameters: {model.count_params():,}")
```

### Step 3: Training Configuration
```python
# What's happening: Setting up training with ViT-specific configurations
# Why these choices: ViT requires specific learning rate schedules and regularization

def create_warmup_cosine_decay_scheduler(
    learning_rate_base=1e-3,
    total_steps=10000,
    warmup_steps=1000,
    hold_base_rate_steps=0,
):
    """Create warmup + cosine decay learning rate schedule"""
    def warmup_cosine_decay(step):
        if step < warmup_steps:
            return learning_rate_base * step / warmup_steps
        elif step < warmup_steps + hold_base_rate_steps:
            return learning_rate_base
        else:
            decay_steps = total_steps - warmup_steps - hold_base_rate_steps
            step = min(step - warmup_steps - hold_base_rate_steps, decay_steps)
            return learning_rate_base * 0.5 * (1 + np.cos(np.pi * step / decay_steps))

    return warmup_cosine_decay

# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 10

steps_per_epoch = len(train_gen)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = steps_per_epoch * WARMUP_EPOCHS

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()
lr_fn = create_warmup_cosine_decay_scheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_steps=warmup_steps
)

# Optimizer with weight decay (AdamW equivalent)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_5_categorical_accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_vit_model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger('training_log.csv')
]
```

### Step 4: Training Loop
```python
# What's happening: Training ViT with monitoring and visualization
# What the algorithm is learning: Global attention patterns and patch relationships

# Train the model
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

def plot_training_history(history):
    """Plot comprehensive training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training and validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Training and validation accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Top-5 accuracy
    axes[1, 0].plot(history.history['top_5_categorical_accuracy'],
                    label='Training Top-5 Accuracy')
    axes[1, 0].plot(history.history['val_top_5_categorical_accuracy'],
                    label='Validation Top-5 Accuracy')
    axes[1, 0].set_title('Top-5 Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate (if tracked)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# Plot results
plot_training_history(history)

# Print final metrics
final_loss = min(history.history['val_loss'])
final_acc = max(history.history['val_accuracy'])
print(f"\nFinal Results:")
print(f"Best Validation Loss: {final_loss:.4f}")
print(f"Best Validation Accuracy: {final_acc:.4f}")
```

### Step 5: Evaluation
```python
# What's happening: Comprehensive model evaluation with attention analysis
# How to interpret results: Accuracy >85% on ImageNet indicates good performance

# Load best model
best_model = tf.keras.models.load_model('best_vit_model.h5')

def evaluate_model(model, test_generator):
    """Comprehensive model evaluation"""
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)

    # Get true labels
    true_labels = test_generator.classes
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(true_labels, predicted_labels,
                                 target_names=class_names)
    print("\nClassification Report:")
    print(report)

    return predictions, true_labels, predicted_labels

# Evaluate model
predictions, y_true, y_pred = evaluate_model(best_model, val_gen)
```

### Step 6: Attention Visualization
```python
# What's happening: Visualizing attention patterns to understand model behavior
# How to use in practice: Attention maps show which patches the model focuses on

class AttentionVisualizationModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs):
        # Get attention weights from transformer blocks
        attention_weights = []
        x = inputs

        # Forward through patch embedding
        patches = self.base_model.layers[1](x)  # PatchEmbedding layer

        # Add cls token and position embedding
        batch_size = tf.shape(patches)[0]
        cls_token = self.base_model.get_layer('cls_token').weights[0]
        cls_tokens = tf.tile(cls_token, [batch_size, 1, 1])
        patches = tf.concat([cls_tokens, patches], axis=1)

        # Add position embeddings
        positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
        encoded_patches = patches + self.base_model.get_layer('embedding')(positions)

        # Forward through transformer blocks and collect attention
        for i, layer in enumerate(self.base_model.layers):
            if isinstance(layer, TransformerBlock):
                encoded_patches = layer(encoded_patches)
                # Extract attention weights if needed

        return self.base_model(inputs), attention_weights

def visualize_attention(model, image, patch_size=16):
    """Visualize attention patterns for a single image"""
    # Preprocess image
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Get model prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    # For demonstration, create a simple attention heatmap
    # In practice, you would extract actual attention weights
    height, width = image.shape[1:3]
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Create mock attention weights (replace with actual extraction)
    attention_map = np.random.rand(num_patches_h, num_patches_w)
    attention_map = cv2.resize(attention_map, (width, height))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention heatmap
    im = axes[1].imshow(attention_map, cmap='hot', alpha=0.8)
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    # Overlay
    axes[2].imshow(image[0])
    axes[2].imshow(attention_map, cmap='hot', alpha=0.4)
    axes[2].set_title(f'Attention Overlay\nPredicted: Class {predicted_class}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
sample_image = next(iter(val_gen))[0][0:1]  # Get first image from validation set
visualize_attention(best_model, sample_image)
```

## Summary

### Key Takeaways
- **Architecture**: Pure attention mechanism applied to image patches without convolutions
- **Data requirements**: Needs large datasets to outperform CNNs effectively
- **Scalability**: Performance improves consistently with more data and larger models
- **Global reasoning**: Self-attention captures long-range spatial dependencies
- **Transfer learning**: Pre-trained models generalize well across visual domains
- **Computational cost**: Higher memory and compute requirements than CNNs

### Quick Reference Points
- **Best for**: Large-scale image classification with abundant training data
- **Training time**: 12-48 hours on modern GPUs for ImageNet-scale datasets
- **Memory**: 16-32GB GPU memory for training with reasonable batch sizes
- **Key hyperparameters**: Patch size (16×16), learning rate (3e-4), batch size (1024+)
- **Common failure**: Training from scratch on small datasets without pre-training
- **Success metric**: >85% ImageNet accuracy indicates strong performance
- **Pre-trained models**: Always use pre-trained weights when available