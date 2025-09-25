# Computer Vision Quick Reference Guide

## 1. OpenCV

**What it is:**\
OpenCV (Open Source Computer Vision Library) is a highly optimized
library focused on real-time applications. It provides tools for
image/video processing, feature detection, tracking, and basic machine
learning algorithms.

**Setup:**

``` bash
pip install opencv-python
```

**Example:**

``` python
import cv2
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
```

------------------------------------------------------------------------

## 2. YOLO (You Only Look Once)

**What it is:**\
YOLO is a family of real-time object detection models. Unlike
sliding-window or region-based methods, YOLO predicts bounding boxes and
class labels in a single forward pass.

**Setup:**

``` bash
pip install ultralytics
```

**Example:**

``` python
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
results = model('image.jpg')
results.show()
```

------------------------------------------------------------------------

## 3. U-Net

**What it is:**\
U-Net is a convolutional neural network architecture designed for
biomedical image segmentation. It follows an encoder-decoder structure
with skip connections, allowing precise localization.

**Setup:**

``` bash
pip install tensorflow keras
```

**Example (simplified):**

``` python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

inputs = Input((128, 128, 3))
conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# ... encoder-decoder continues ...
outputs = Conv2D(1, 1, activation='sigmoid')(conv1)
model = Model(inputs=[inputs], outputs=[outputs])
```

------------------------------------------------------------------------

## 4. Vision Transformer (ViT)

**What it is:**\
ViT applies the transformer architecture, originally designed for NLP,
to image patches. Each image is split into fixed-size patches,
flattened, and embedded, then processed like tokens.

**Setup:**

``` bash
pip install transformers torch
```

**Example:**

``` python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
```

------------------------------------------------------------------------

## 5. CLIP (Contrastive Language--Image Pretraining)

**What it is:**\
CLIP is trained to connect text and images. It learns a joint embedding
space using contrastive learning, enabling zero-shot classification,
retrieval, and multimodal tasks.

**Setup:**

``` bash
pip install git+https://github.com/openai/CLIP.git
```

**Example:**

``` python
import torch, clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("image.jpg")).unsqueeze(0)
text = clip.tokenize(["a cat", "a dog"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", probs)
```

------------------------------------------------------------------------

# Quick Decision Guide

-   **OpenCV** → Basic image ops, preprocessing, classical CV\
-   **YOLO** → Real-time object detection (bounding boxes + labels)\
-   **U-Net** → Semantic segmentation (pixel-level masks)\
-   **ViT** → Image classification with transformer architecture\
-   **CLIP** → Connect images ↔ text (zero-shot, retrieval)

------------------------------------------------------------------------

# Comparison Table

  -------------------------------------------------------------------------------------------
  Library/Model   Primary Use Case  Strengths             Weaknesses        Typical
                                                                            Applications
  --------------- ----------------- --------------------- ----------------- -----------------
  **OpenCV**      Image processing, Lightweight, fast,    Limited deep      Preprocessing,
                  CV basics         classical methods     learning support  filters, tracking

  **YOLO**        Object detection  Real-time detection,  May miss very     Autonomous
                                    good accuracy         small objects     driving,
                                                                            surveillance

  **U-Net**       Image             Pixel-level accuracy, Requires lots of  Medical imaging,
                  segmentation      strong localization   labeled data      remote sensing

  **ViT**         Image             Transformer-based,    Data-hungry,      Classification,
                  classification    strong on large       computationally   transfer learning
                                    datasets              heavy             

  **CLIP**        Vision-language   Zero-shot, multimodal Requires large    Search engines,
                  tasks             understanding         compute to train  multimodal AI
  -------------------------------------------------------------------------------------------
