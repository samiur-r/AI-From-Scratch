# Vision Transformer (ViT) Quick Reference

Vision Transformer is a computer vision framework that applies transformer architectures directly to image classification by treating images as sequences of patches, enabling pure attention-based visual understanding.

### Installation
```bash
# Install with transformers (Hugging Face)
pip install transformers torch torchvision

# Install with timm (PyTorch Image Models)
pip install timm

# For TensorFlow/Keras
pip install tensorflow transformers

# For JAX/Flax
pip install flax transformers
```

### Importing Vision Transformer
```python
# Hugging Face Transformers
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# PyTorch Image Models (timm)
import timm

# TensorFlow/Keras
import tensorflow as tf
from transformers import TFViTForImageClassification
```

* * * * *

## 1. Loading Pre-trained Models
```python
# Using Hugging Face Transformers
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Using timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Available model variants
# vit-tiny-patch16-224 (5.7M params)
# vit-small-patch16-224 (22M params)
# vit-base-patch16-224 (86M params)
# vit-large-patch16-224 (307M params)
# vit-huge-patch14-224 (632M params)
```

## 2. Basic Image Classification
```python
from PIL import Image
import requests

# Load and preprocess image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Using Hugging Face
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_idx = logits.argmax(-1).item()
    print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")

# Using timm
transform = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**transform)

input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(5):
        print(f"{top5_catid[i].item()}: {top5_prob[i].item():.4f}")
```

## 3. Fine-tuning on Custom Dataset
```python
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        encoding = self.processor(image, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)

        return encoding

# Setup for fine-tuning
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes,  # Your number of classes
    ignore_mismatched_sizes=True
)

# Create datasets
train_dataset = CustomImageDataset(train_images, train_labels, processor)
val_dataset = CustomImageDataset(val_images, val_labels, processor)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

## 4. Feature Extraction and Embeddings
```python
# Extract features before classification head
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Remove classification head for feature extraction
feature_extractor = model.vit

# Process image
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = feature_extractor(**inputs)

    # Get patch embeddings (sequence_length x hidden_size)
    patch_embeddings = outputs.last_hidden_state

    # Get CLS token embedding (global image representation)
    cls_embedding = patch_embeddings[:, 0]  # First token is [CLS]

    # Get patch tokens (spatial features)
    patch_tokens = patch_embeddings[:, 1:]  # Remaining tokens are patches

print(f"CLS embedding shape: {cls_embedding.shape}")
print(f"Patch embeddings shape: {patch_tokens.shape}")
```

## 5. Attention Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

# Get attention weights
model.eval()
with torch.no_grad():
    outputs = model.vit(**inputs, output_attentions=True)
    attentions = outputs.attentions  # List of attention matrices

# Visualize attention from last layer, first head
last_layer_attention = attentions[-1][0, 0].cpu().numpy()  # [seq_len, seq_len]

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
custom_model = ViTForImageClassification(config)

# Or modify existing model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Freeze backbone and only train classification head
for param in model.vit.parameters():
    param.requires_grad = False

# Only classification head parameters will be updated
for param in model.classifier.parameters():
    param.requires_grad = True
```

## 7. Batch Processing and Inference Optimization
```python
# Batch processing for efficiency
images_batch = [image1, image2, image3, image4]  # List of PIL Images

# Process batch
inputs = processor(images=images_batch, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get top predictions for each image
for i, pred in enumerate(predictions):
    top5_prob, top5_idx = torch.topk(pred, 5)
    print(f"Image {i+1} top predictions:")
    for j in range(5):
        class_name = model.config.id2label[top5_idx[j].item()]
        probability = top5_prob[j].item()
        print(f"  {class_name}: {probability:.4f}")

# Optimize for inference
model.eval()
torch.set_grad_enabled(False)

# Use half precision for faster inference
model.half()
inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
```

## 8. Integration with Other Frameworks
```python
# Export to ONNX
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "vit_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# Use with TensorFlow
tf_model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Convert PIL image to TensorFlow tensor
import tensorflow as tf
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
inputs = processor(images=image, return_tensors="tf")

outputs = tf_model(**inputs)
predictions = tf.nn.softmax(outputs.logits, axis=-1)
```

## 9. Data Augmentation for ViT
```python
from torchvision import transforms
import timm.data.transforms as timm_transforms

# Standard augmentation pipeline for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Advanced augmentations (RandAugment, AutoAugment)
from timm.data import create_transform

transform = create_transform(
    input_size=224,
    is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic',
    re_prob=0.25,
    re_mode='pixel',
    re_count=1,
)
```

## 10. Performance Monitoring and Evaluation
```python
import time
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_performance(model, test_loader, device='cuda'):
    model.eval()
    all_predictions = []
    all_labels = []
    total_time = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            total_time += (end_time - start_time)

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_inference_time = total_time / len(test_loader)

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
    model = ViTForImageClassification.from_pretrained(model_name)
    accuracy, inference_time = evaluate_model_performance(model, test_loader)
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