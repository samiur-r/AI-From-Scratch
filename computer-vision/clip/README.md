# CLIP (Contrastive Language-Image Pre-training) Quick Reference

CLIP is a multimodal framework that learns joint representations of images and text through contrastive learning, enabling zero-shot classification and cross-modal retrieval without task-specific training.

### Installation
```bash
# Install CLIP with TensorFlow and Transformers
pip install tensorflow transformers

# For GPU support
pip install tensorflow[and-cuda]

# Additional dependencies
pip install pillow requests numpy matplotlib scikit-learn
```

### Importing CLIP
```python
# TensorFlow with Hugging Face Transformers
import tensorflow as tf
from transformers import TFCLIPModel, CLIPProcessor

# Additional utilities
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
```

* * * * *

## 1. Loading Pre-trained Models
```python
# Using TensorFlow with Hugging Face Transformers
model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Available CLIP models:
# openai/clip-vit-base-patch32
# openai/clip-vit-base-patch16
# openai/clip-vit-large-patch14

# Load different model sizes
model_large = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor_large = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print(f"Model loaded successfully with {model.config.vision_config.hidden_size} hidden size")
```

## 2. Zero-Shot Image Classification
```python
def zero_shot_classification(image_path, candidate_labels):
    image = Image.open(image_path)

    # Create text prompts
    text_prompts = [f"a photo of a {label}" for label in candidate_labels]

    # Process inputs
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="tf",
        padding=True
    )

    # Get predictions
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = tf.nn.softmax(logits_per_image, axis=1)

    # Return results
    results = []
    for i, label in enumerate(candidate_labels):
        results.append({
            'label': label,
            'score': float(probs[0][i])
        })

    return sorted(results, key=lambda x: x['score'], reverse=True)

# Example usage
labels = ["dog", "cat", "bird", "car", "airplane"]
results = zero_shot_classification("sample_image.jpg", labels)

for result in results:
    print(f"{result['label']}: {result['score']:.4f}")
```

## 3. Image-Text Similarity
```python
# Compute similarity between images and text
def compute_similarity(images, texts):
    # Process inputs
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="tf",
        padding=True
    )

    # Get embeddings
    image_embeds = model.get_image_features(pixel_values=inputs['pixel_values'])
    text_embeds = model.get_text_features(input_ids=inputs['input_ids'])

    # Normalize embeddings
    image_embeds = tf.nn.l2_normalize(image_embeds, axis=-1)
    text_embeds = tf.nn.l2_normalize(text_embeds, axis=-1)

    # Compute similarity
    similarity = tf.matmul(text_embeds, image_embeds, transpose_b=True)

    return similarity.numpy()

# Example
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
texts = ["a red car", "a cute dog", "a beautiful landscape"]

similarity_matrix = compute_similarity(images, texts)
print("Similarity matrix shape:", similarity_matrix.shape)
```

## 4. Image Retrieval
```python
def image_retrieval(query_text, image_database, top_k=5):
    """Retrieve most relevant images for a text query"""

    # Process all images and the query text
    inputs = processor(
        text=query_text,
        images=image_database,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        # Get embeddings
        text_embeds = model.get_text_features(input_ids=inputs['input_ids'])
        image_embeds = model.get_image_features(pixel_values=inputs['pixel_values'])

        # Normalize
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(text_embeds, image_embeds.T).squeeze(0)

    # Get top-k results
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'image_index': idx.item(),
            'similarity': similarities[idx].item()
        })

    return results

# Example usage
database_images = [Image.open(f"image_{i}.jpg") for i in range(100)]
query = "a red sports car"
results = image_retrieval(query, database_images, top_k=5)

print(f"Top results for '{query}':")
for result in results:
    print(f"Image {result['image_index']}: {result['similarity']:.4f}")
```

## 5. Text Retrieval
```python
def text_retrieval(query_image, text_database, top_k=5):
    """Retrieve most relevant texts for an image query"""

    inputs = processor(
        text=text_database,
        images=query_image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        text_embeds = model.get_text_features(input_ids=inputs['input_ids'])
        image_embeds = model.get_image_features(pixel_values=inputs['pixel_values'])

        # Normalize
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(image_embeds, text_embeds.T).squeeze(0)

    # Get top-k results
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'text': text_database[idx],
            'similarity': similarities[idx].item()
        })

    return results

# Example
query_image = Image.open("sample_image.jpg")
captions = [
    "A red car driving on a highway",
    "A dog playing in the park",
    "A beautiful sunset over mountains",
    "A person riding a bicycle",
    "A cat sleeping on a couch"
]

results = text_retrieval(query_image, captions)
for result in results:
    print(f"'{result['text']}': {result['similarity']:.4f}")
```

## 6. Feature Extraction
```python
def extract_features(images=None, texts=None):
    """Extract CLIP features for images and/or texts"""

    features = {}

    if images is not None:
        # Process images
        image_inputs = processor(images=images, return_tensors="tf")
        image_features = model.get_image_features(**image_inputs)
        # Normalize features
        image_features = tf.nn.l2_normalize(image_features, axis=-1)
        features['image_features'] = image_features

    if texts is not None:
        # Process texts
        text_inputs = processor(text=texts, return_tensors="tf", padding=True)
        text_features = model.get_text_features(**text_inputs)
        # Normalize features
        text_features = tf.nn.l2_normalize(text_features, axis=-1)
        features['text_features'] = text_features

    return features

# Example usage
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
texts = ["a red car", "a cute dog"]

features = extract_features(images=images, texts=texts)
print("Image features shape:", features['image_features'].shape)
print("Text features shape:", features['text_features'].shape)

# Save features for later use
np.savez('clip_features.npz',
         image_features=features['image_features'].numpy(),
         text_features=features['text_features'].numpy())
```

## 7. Batch Processing for Efficiency
```python
def batch_process_images(image_paths, batch_size=32):
    """Process images in batches for better efficiency"""

    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(path) for path in batch_paths]

        # Process batch
        inputs = processor(images=batch_images, return_tensors="tf")

        features = model.get_image_features(**inputs)
        features = tf.nn.l2_normalize(features, axis=-1)

        all_features.append(features)
        print(f"Processed batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

    return tf.concat(all_features, axis=0)

# Example
image_paths = [f"dataset/image_{i:05d}.jpg" for i in range(1000)]
all_image_features = batch_process_images(image_paths, batch_size=32)
print(f"Total features shape: {all_image_features.shape}")

# Process with tf.data for even better performance
def create_image_dataset(image_paths, batch_size=32):
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
```

## 8. Custom Classification with Prompt Engineering
```python
class CLIPClassifier:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def create_prompts(self, class_names, templates=None):
        """Create multiple prompt templates for robust classification"""
        if templates is None:
            templates = [
                "a photo of a {}",
                "a picture of a {}",
                "an image of a {}",
                "this is a {}",
                "{}",
            ]

        all_prompts = []
        for class_name in class_names:
            class_prompts = [template.format(class_name) for template in templates]
            all_prompts.extend(class_prompts)

        return all_prompts, len(templates)

    def classify_with_ensembling(self, image, class_names, templates=None):
        """Classify using prompt ensembling for better accuracy"""

        prompts, num_templates = self.create_prompts(class_names, templates)

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image.squeeze(0)

        # Reshape and average over templates
        logits = logits.view(len(class_names), num_templates)
        avg_logits = logits.mean(dim=1)
        probs = torch.softmax(avg_logits, dim=0)

        results = []
        for i, class_name in enumerate(class_names):
            results.append({
                'class': class_name,
                'probability': probs[i].item()
            })

        return sorted(results, key=lambda x: x['probability'], reverse=True)

# Example usage
classifier = CLIPClassifier(model, processor)
image = Image.open("test_image.jpg")
classes = ["dog", "cat", "bird", "car", "airplane"]

results = classifier.classify_with_ensembling(image, classes)
print("Classification results:")
for result in results:
    print(f"{result['class']}: {result['probability']:.4f}")
```

## 9. Fine-tuning CLIP
```python
def create_dataset_for_training(image_paths, texts, processor, batch_size=16):
    """Create TensorFlow dataset for CLIP training"""

    def preprocess_function(image_path, text):
        # Load and process image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0

        return {'pixel_values': image, 'text': text}

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, texts))
    dataset = dataset.map(preprocess_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def contrastive_loss(logits_per_image, logits_per_text):
    """CLIP contrastive loss"""
    batch_size = tf.shape(logits_per_image)[0]
    labels = tf.range(batch_size)

    loss_img = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits_per_image, from_logits=True
    )
    loss_txt = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits_per_text, from_logits=True
    )

    return (tf.reduce_mean(loss_img) + tf.reduce_mean(loss_txt)) / 2

def fine_tune_clip(model, train_dataset, val_dataset, epochs=5, lr=1e-5):
    """Fine-tune CLIP on custom data"""

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                training=True
            )

            loss = contrastive_loss(
                outputs.logits_per_image,
                outputs.logits_per_text
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_dataset:
            loss = train_step(batch)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return model

# Example setup
# train_dataset = create_dataset_for_training(train_image_paths, train_texts, processor)
# val_dataset = create_dataset_for_training(val_image_paths, val_texts, processor)
# fine_tuned_model = fine_tune_clip(model, train_dataset, val_dataset)
```

## 10. Performance Optimization
```python
def optimize_for_inference(model):
    """Optimize CLIP model for faster inference"""

    # Use mixed precision for faster inference
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Compile functions for better performance
    @tf.function
    def optimized_image_features(pixel_values):
        return model.get_image_features(pixel_values=pixel_values)

    @tf.function
    def optimized_text_features(input_ids):
        return model.get_text_features(input_ids=input_ids)

    return optimized_image_features, optimized_text_features

def benchmark_clip(model, processor, num_images=100, num_texts=100):
    """Benchmark CLIP performance"""
    import time

    # Generate dummy data
    dummy_images = [Image.new('RGB', (224, 224)) for _ in range(num_images)]
    dummy_texts = [f"sample text {i}" for i in range(num_texts)]

    # Benchmark image encoding
    start_time = time.time()
    inputs = processor(images=dummy_images, return_tensors="tf")
    image_features = model.get_image_features(**inputs)
    image_time = time.time() - start_time

    # Benchmark text encoding
    start_time = time.time()
    inputs = processor(text=dummy_texts, return_tensors="tf", padding=True)
    text_features = model.get_text_features(**inputs)
    text_time = time.time() - start_time

    print(f"Image encoding: {image_time:.4f}s for {num_images} images")
    print(f"Text encoding: {text_time:.4f}s for {num_texts} texts")
    print(f"Images per second: {num_images/image_time:.2f}")
    print(f"Texts per second: {num_texts/text_time:.2f}")

    return image_time, text_time

# Optimize and benchmark
optimized_img_fn, optimized_txt_fn = optimize_for_inference(model)
image_time, text_time = benchmark_clip(model, processor)

# Convert to TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_concrete_functions([
    optimized_img_fn.get_concrete_function(tf.TensorSpec([None, 3, 224, 224], tf.float32))
])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('clip_image_encoder.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 11. Integration with Vector Databases
```python
def create_image_index(image_paths, model, processor, batch_size=32):
    """Create searchable index of image embeddings"""

    all_embeddings = []
    image_metadata = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(path) for path in batch_paths]

        inputs = processor(images=batch_images, return_tensors="pt")

        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(embeddings)
        image_metadata.extend([{'path': path, 'index': i + j}
                             for j, path in enumerate(batch_paths)])

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return {
        'embeddings': all_embeddings,
        'metadata': image_metadata
    }

def search_similar_images(query_embedding, image_index, top_k=10):
    """Search for similar images using cosine similarity"""

    similarities = torch.matmul(query_embedding, image_index['embeddings'].T)
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'metadata': image_index['metadata'][idx],
            'similarity': similarities[idx].item()
        })

    return results

# Example usage
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "..."]
index = create_image_index(image_paths, model, processor)

# Search with text query
query_text = "a red car"
text_inputs = processor(text=query_text, return_tensors="pt")
query_embedding = model.get_text_features(**text_inputs)
query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)

similar_images = search_similar_images(query_embedding, index)
print(f"Top similar images for '{query_text}':")
for result in similar_images[:5]:
    print(f"{result['metadata']['path']}: {result['similarity']:.4f}")
```

* * * * *

Summary
=======

- **Zero-shot capabilities** classify images using natural language without training examples
- **Multimodal embeddings** joint image-text representations enable cross-modal retrieval
- **Pre-trained models** available in multiple architectures (ViT and ResNet backbones)
- **Flexible prompting** prompt engineering significantly improves classification accuracy
- **Scalable processing** efficient batch processing for large-scale applications
- **Framework integration** works with PyTorch, TensorFlow, and popular ML frameworks
- **Production ready** optimizable with quantization, caching, and vector databases