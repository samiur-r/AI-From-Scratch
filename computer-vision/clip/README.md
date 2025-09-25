# CLIP (Contrastive Language-Image Pre-training) Quick Reference

CLIP is a multimodal framework that learns joint representations of images and text through contrastive learning, enabling zero-shot classification and cross-modal retrieval without task-specific training.

### Installation
```bash
# Install CLIP with Hugging Face Transformers
pip install transformers torch torchvision

# Install OpenAI's original CLIP
pip install clip-by-openai

# Alternative installation
pip install git+https://github.com/openai/CLIP.git

# For additional dependencies
pip install ftfy regex tqdm Pillow
```

### Importing CLIP
```python
# Using Hugging Face Transformers
from transformers import CLIPProcessor, CLIPModel
import torch

# Using OpenAI's CLIP
import clip

# Additional utilities
from PIL import Image
import requests
import numpy as np
```

* * * * *

## 1. Loading Pre-trained Models
```python
# Using Hugging Face Transformers
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Using OpenAI's CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Available CLIP models:
# ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
# RN50, RN101, RN50x4, RN50x16, RN50x64
```

## 2. Zero-Shot Image Classification
```python
# Using Hugging Face
def zero_shot_classification(image_path, candidate_labels):
    image = Image.open(image_path)

    # Create text prompts
    text_prompts = [f"a photo of a {label}" for label in candidate_labels]

    # Process inputs
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Return results
    results = []
    for i, label in enumerate(candidate_labels):
        results.append({
            'label': label,
            'score': probs[0][i].item()
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
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        # Get embeddings
        image_embeds = model.get_image_features(pixel_values=inputs['pixel_values'])
        text_embeds = model.get_text_features(input_ids=inputs['input_ids'])

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute similarity
        similarity = torch.matmul(text_embeds, image_embeds.T)

    return similarity.cpu().numpy()

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
        image_inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            # Normalize features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        features['image_features'] = image_features

    if texts is not None:
        # Process texts
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            # Normalize features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        features['text_features'] = text_features

    return features

# Example usage
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
texts = ["a red car", "a cute dog"]

features = extract_features(images=images, texts=texts)
print("Image features shape:", features['image_features'].shape)
print("Text features shape:", features['text_features'].shape)

# Save features for later use
torch.save(features, 'clip_features.pt')
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
        inputs = processor(images=batch_images, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        all_features.append(features)
        print(f"Processed batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

    return torch.cat(all_features, dim=0)

# Example
image_paths = [f"dataset/image_{i:05d}.jpg" for i in range(1000)]
all_image_features = batch_process_images(image_paths, batch_size=32)
print(f"Total features shape: {all_image_features.shape}")
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
from torch.utils.data import DataLoader, Dataset

class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        text = self.texts[idx]

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        return {k: v.squeeze() for k, v in inputs.items()}

def fine_tune_clip(model, train_dataset, val_dataset, epochs=5, lr=1e-5):
    """Fine-tune CLIP on custom data"""

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(**batch)

            # CLIP contrastive loss
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            batch_size = logits_per_image.shape[0]
            labels = torch.arange(batch_size, device=logits_per_image.device)

            loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return model

# Example setup
# train_dataset = ImageTextDataset(train_image_paths, train_texts, processor)
# val_dataset = ImageTextDataset(val_image_paths, val_texts, processor)
# fine_tuned_model = fine_tune_clip(model, train_dataset, val_dataset)
```

## 10. Performance Optimization
```python
def optimize_for_inference(model):
    """Optimize CLIP model for faster inference"""

    # Set to evaluation mode
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Use half precision if available
    if torch.cuda.is_available():
        model.half()

    return model

def benchmark_clip(model, processor, num_images=100, num_texts=100):
    """Benchmark CLIP performance"""
    import time

    # Generate dummy data
    dummy_images = [Image.new('RGB', (224, 224)) for _ in range(num_images)]
    dummy_texts = [f"sample text {i}" for i in range(num_texts)]

    # Benchmark image encoding
    start_time = time.time()
    inputs = processor(images=dummy_images, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_time = time.time() - start_time

    # Benchmark text encoding
    start_time = time.time()
    inputs = processor(text=dummy_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_time = time.time() - start_time

    print(f"Image encoding: {image_time:.4f}s for {num_images} images")
    print(f"Text encoding: {text_time:.4f}s for {num_texts} texts")
    print(f"Images per second: {num_images/image_time:.2f}")
    print(f"Texts per second: {num_texts/text_time:.2f}")

# Optimize and benchmark
optimized_model = optimize_for_inference(model)
benchmark_clip(optimized_model, processor)
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