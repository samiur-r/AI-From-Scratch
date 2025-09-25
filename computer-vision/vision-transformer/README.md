# Vision Transformer (ViT) Quick Reference

Vision Transformer is a computer vision framework that applies transformer architectures directly to image classification by treating images as sequences of patches, enabling pure attention-based visual understanding.

### Installation
```bash
# Install with TensorFlow and transformers
pip install tensorflow transformers

# For GPU support
pip install tensorflow[and-cuda]

# Additional utilities
pip install pillow requests matplotlib scikit-learn
```

### Importing Vision Transformer
```python
# TensorFlow/Keras with Transformers
import tensorflow as tf
from transformers import TFViTForImageClassification, ViTImageProcessor

# Core TensorFlow
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
```

* * * * *

## 1. Loading Pre-trained Models
```python
# Using TensorFlow with Hugging Face Transformers
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Available model variants
# google/vit-tiny-patch16-224 (5.7M params)
# google/vit-small-patch16-224 (22M params)
# google/vit-base-patch16-224 (86M params)
# google/vit-large-patch16-224 (307M params)

# Load specific model size
model_small = TFViTForImageClassification.from_pretrained('google/vit-small-patch16-224')
model_large = TFViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

print(f"Model loaded with {model.num_parameters():,} parameters")
```

## 2. Basic Image Classification
```python
import requests

# Load and preprocess image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process image with ViT processor
inputs = processor(images=image, return_tensors="tf")

# Make prediction
outputs = model(**inputs)
logits = outputs.logits

# Get predicted class
predicted_class_idx = tf.argmax(logits, axis=-1)[0]
predicted_class = model.config.id2label[int(predicted_class_idx)]

print(f"Predicted class: {predicted_class}")

# Get top-5 predictions with probabilities
probabilities = tf.nn.softmax(logits[0])
top5_indices = tf.nn.top_k(probabilities, k=5).indices

print("\nTop-5 predictions:")
for i in range(5):
    class_idx = int(top5_indices[i])
    class_name = model.config.id2label[class_idx]
    probability = float(probabilities[class_idx])
    print(f"{class_name}: {probability:.4f}")
```

## 3. Fine-tuning on Custom Dataset
```python
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

def create_dataset(images, labels, processor, batch_size=16):
    """Create TensorFlow dataset for training"""

    def preprocess_function(image, label):
        # Process image using ViT processor
        inputs = processor(images=image, return_tensors="tf")
        return inputs['pixel_values'][0], label

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Setup for fine-tuning
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = TFViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes,  # Your number of classes
    ignore_mismatched_sizes=True
)

# Prepare datasets
train_dataset = create_dataset(train_images, train_labels, processor, batch_size=16)
val_dataset = create_dataset(val_images, val_labels, processor, batch_size=16)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Setup callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

# Fine-tune model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save fine-tuned model
model.save_pretrained('./vit-finetuned')
```

## 4. Feature Extraction and Embeddings
```python
# Load model for feature extraction
model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Process image
inputs = processor(images=image, return_tensors="tf")

# Extract features using the ViT backbone
outputs = model.vit(**inputs)

# Get patch embeddings (sequence_length x hidden_size)
patch_embeddings = outputs.last_hidden_state

# Get CLS token embedding (global image representation)
cls_embedding = patch_embeddings[:, 0]  # First token is [CLS]

# Get patch tokens (spatial features)
patch_tokens = patch_embeddings[:, 1:]  # Remaining tokens are patches

print(f"CLS embedding shape: {cls_embedding.shape}")
print(f"Patch embeddings shape: {patch_tokens.shape}")

# Save embeddings for later use
tf.saved_model.save({
    'cls_embedding': cls_embedding,
    'patch_tokens': patch_tokens
}, 'vit_features')
```

## 5. Attention Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

# Get attention weights
outputs = model.vit(**inputs, output_attentions=True)
attentions = outputs.attentions  # List of attention matrices

# Visualize attention from last layer, first head
last_layer_attention = attentions[-1][0, 0].numpy()  # [seq_len, seq_len]

# Attention from CLS token to patches
cls_attention = last_layer_attention[0, 1:]  # Remove CLS to CLS attention

# Reshape to spatial dimensions (assuming 14x14 patches for 224x224 image)
patch_size = 16
image_size = 224
num_patches = (image_size // patch_size) ** 2
grid_size = int(np.sqrt(num_patches))

attention_map = cls_attention.reshape(grid_size, grid_size)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis('off')

im = ax2.imshow(attention_map, cmap='hot', interpolation='nearest')
ax2.set_title("Attention Map")
ax2.axis('off')
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()
```

## 6. Custom ViT Architecture
```python
from transformers import ViTConfig

# Create custom ViT configuration
config = ViTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_labels=1000,
)

# Create model with custom config
custom_model = TFViTForImageClassification(config)

# Or modify existing model
model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Freeze backbone and only train classification head
for layer in model.vit.layers:
    layer.trainable = False

# Only classification head will be updated
model.classifier.trainable = True

print(f"Trainable parameters: {model.count_params()}")
```

## 7. Batch Processing and Inference Optimization
```python
# Batch processing for efficiency
images_batch = [image1, image2, image3, image4]  # List of PIL Images

# Process batch
inputs = processor(images=images_batch, return_tensors="tf")

# Make batch predictions
outputs = model(**inputs)
predictions = tf.nn.softmax(outputs.logits, axis=-1)

# Get top predictions for each image
for i, pred in enumerate(predictions):
    top5_values, top5_indices = tf.nn.top_k(pred, k=5)
    print(f"Image {i+1} top predictions:")
    for j in range(5):
        class_idx = int(top5_indices[j])
        class_name = model.config.id2label[class_idx]
        probability = float(top5_values[j])
        print(f"  {class_name}: {probability:.4f}")

# Optimize for inference with mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Convert to TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_concrete_functions([model.call.get_concrete_function(inputs)])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('vit_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 8. Integration with Other Frameworks
```python
# Export to SavedModel format
tf.saved_model.save(model, "vit_savedmodel")

# Load SavedModel
loaded_model = tf.saved_model.load("vit_savedmodel")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("vit_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('vit_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use TFLite model for inference
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor
input_data = inputs['pixel_values'].numpy().astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
predictions = tf.nn.softmax(output_data, axis=-1)
```

## 9. Data Augmentation for ViT
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Standard augmentation pipeline for ViT
def create_augmentation_pipeline():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest',
        preprocessing_function=tf.keras.applications.imagenet_utils.preprocess_input
    )

# Advanced augmentations using tf.image
@tf.function
def augment_image(image):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image

# Apply augmentation to dataset
augmented_dataset = train_dataset.map(
    lambda x, y: (augment_image(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
```

## 10. Performance Monitoring and Evaluation
```python
import time
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_performance(model, test_dataset):
    all_predictions = []
    all_labels = []
    total_time = 0
    batch_count = 0

    for batch in test_dataset:
        inputs, labels = batch

        start_time = time.time()
        outputs = model(inputs, training=False)
        end_time = time.time()

        total_time += (end_time - start_time)
        batch_count += 1

        predictions = tf.argmax(outputs.logits, axis=-1)
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_inference_time = total_time / batch_count

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print(f"Classification Report:")
    print(classification_report(all_labels, all_predictions))

    return accuracy, avg_inference_time

# Benchmark different model sizes
models_to_test = [
    'google/vit-tiny-patch16-224',
    'google/vit-small-patch16-224',
    'google/vit-base-patch16-224',
    'google/vit-large-patch16-224'
]

for model_name in models_to_test:
    print(f"\nEvaluating {model_name}:")
    model = TFViTForImageClassification.from_pretrained(model_name)
    accuracy, inference_time = evaluate_model_performance(model, test_dataset)

    # Memory usage
    model_size = model.count_params()
    print(f"Model parameters: {model_size:,}")
```

* * * * *

Summary
=======

- **Patch-based processing** treats images as sequences of patches for transformer processing
- **Pre-trained models** available in multiple sizes from tiny (5.7M) to huge (632M parameters)
- **Transfer learning** excellent performance when fine-tuning on domain-specific data
- **Attention visualization** provides interpretable insights into model focus areas
- **Scalable architecture** performance improves with larger models and more training data
- **Framework flexibility** available across PyTorch, TensorFlow, and JAX ecosystems
- **Production ready** optimizable for inference with ONNX export and quantization support