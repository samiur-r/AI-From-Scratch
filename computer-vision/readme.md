
# Computer Vision Quick Reference Guide

This cheat sheet covers five core tools you can mix and match for most day‑to‑day CV work. Each section includes a one‑liner, setup, and a minimal example.

---

## 1) OpenCV

**What it is:** A fast, production‑ready library for image/video I/O, transforms, drawing, filtering, features, basic tracking, and classical CV.

**Install:**

```bash
pip install opencv-python        # desktop environments
# or, on servers/CI without GUI deps:
pip install opencv-python-headless
# for extra contrib modules:
pip install opencv-contrib-python
```

**Minimal example:**

```python
import cv2

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("edges.jpg", edges)
```

**Good for:** I/O, resizing, color spaces, contour ops, feature matching (ORB/SIFT/SURF*), optical flow, simple tracking.  
*Some patented/legacy features are in `opencv-contrib`.*

---

## 2) YOLO (You Only Look Once)

**What it is:** Real‑time object detection family; predicts boxes + labels in a single forward pass.

**Install:**

```bash
pip install ultralytics
```

**Minimal example (YOLOv8):**

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")           # tiny, fast model
result = model("image.jpg")[0]        # first image result

for xyxy, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    print(xyxy.tolist(), result.names[int(cls)], float(conf))
```

**Good for:** Real‑time detection on images/video/RTSP, easy export (ONNX, TensorRT), quick prototypes to production.

---

## 3) U‑Net

**What it is:** Encoder‑decoder CNN with skip connections for **semantic segmentation** (pixel‑wise masks), widely used in medical & remote sensing.

**Install:**

```bash
pip install tensorflow keras
# or PyTorch if you prefer that ecosystem
```

**Minimal Keras sketch (toy):**

```python
from tensorflow.keras import layers, Model, Input

def conv_block(x, c):
    x = layers.Conv2D(c, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(c, 3, padding="same", activation="relu")(x)
    return x

inp = Input((128, 128, 3))
c1 = conv_block(inp, 32); p1 = layers.MaxPool2D()(c1)
c2 = conv_block(p1, 64);  p2 = layers.MaxPool2D()(c2)
b  = conv_block(p2, 128)
u2 = layers.UpSampling2D()(b);  m2 = layers.Concatenate()([u2, c2]); c3 = conv_block(m2, 64)
u1 = layers.UpSampling2D()(c3);  m1 = layers.Concatenate()([u1, c1]); c4 = conv_block(m1, 32)
out = layers.Conv2D(1, 1, activation="sigmoid")(c4)

model = Model(inp, out)
model.compile(optimizer="adam", loss="binary_crossentropy")
```

**Good for:** Pixel‑accurate masks when you have paired (image, mask) data.

---

## 4) Vision Transformer (ViT)

**What it is:** Applies Transformers to image **patches** for image **classification** (and as a backbone for other tasks).

**Install:**

```bash
pip install transformers torch torchvision
```

**Minimal example:**

```python
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

image = Image.open("image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
pred = model(**inputs).logits.argmax(-1).item()
print("Class id:", pred, "->", model.config.id2label[pred])
```

**Good for:** Strong classification with large‑scale pretraining; robust features for transfer learning.

---

## 5) CLIP (Contrastive Language–Image Pretraining)

**What it is:** Joint text–image embeddings trained with contrastive learning → zero‑shot classification, retrieval, and multimodal tricks.

**Install:**

```bash
pip install git+https://github.com/openai/CLIP.git
# or: pip install open_clip_torch   # popular community variant
```

**Minimal example (OpenAI CLIP):**

```python
import torch, clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a cat", "a dog"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", probs.squeeze().tolist())
```

**Good for:** Zero‑shot tagging, cross‑modal retrieval, deduping via embeddings, semantic search.

---

## Quick Decision Guide

- **OpenCV** → Pre/post‑processing, I/O, classical CV, glue for pipelines.
- **YOLO** → Real‑time **object detection** (boxes + labels).
- **U‑Net** → **Semantic segmentation** (pixel‑level masks).
- **ViT** → **Image classification** / transfer learning backbone.
- **CLIP** → **Zero‑shot** classification & **image↔text** retrieval.

---

## Comparison Table (GFM‑compatible)

| Library/Model | Primary Use Case                     | Strengths                                                  | Weaknesses / Caveats                              | Typical Applications                    |
| --- | --- | --- | --- | --- |
| **OpenCV** | Image processing & classical CV | Fast, lightweight; huge API surface; great I/O & utils | Limited built‑in deep learning; contrib split | Preprocessing, filters, contours, tracking |
| **YOLO** | Object detection (real‑time) | Strong pretrained models; easy training/export; fast | Small/occluded objects can be hard; boxes only (use seg variants for masks) | Robotics, CCTV, retail analytics |
| **U‑Net** | Semantic segmentation | Precise localization via skips; works with small images | Needs labeled masks; memory heavy for big inputs | Medical imaging, satellite/remote sensing |
| **ViT** | Image classification | Excellent with pretraining; good transfer | Data‑hungry; slower on small models vs CNNs | Tagging, QA, feature backbone |
| **CLIP** | Vision↔Text zero‑shot & retrieval | Zero‑shot; flexible prompts; embeddings for search | Prompt/bias sensitivity; not task‑specific boxes/masks | Moderation, semantic search, auto‑tagging |

---

## Task → Starting Point

| Task | Recommended Start | Notes |
| --- | --- | --- |
| Preprocessing, I/O, drawing | **OpenCV** | Resize, crop, colorspaces, denoise, codecs. |
| Object detection | **YOLO (ultralytics)** | Train on your data; export to ONNX/TensorRT for speed. |
| Semantic segmentation | **U‑Net** | Start small (128–256px), scale up; use augmentations. |
| Image classification | **ViT** (or ResNet) | Fine‑tune from pretrained; watch input size (224×224). |
| Zero‑shot labels / retrieval | **CLIP** | Engineer prompts; normalize embeddings for search. |
| OCR (bonus) | **Tesseract** / **EasyOCR** | Combine with OpenCV binarization & denoise. |
| Pose/landmarks (bonus) | **MediaPipe**, **OpenPose** | Real‑time body/hand/face pose estimation. |

---

## Practical Tips

- **Data first:** Balanced, labeled data beats model tweaks. Add augmentations (flip/rotate/color jitter).  
- **Evaluate properly:** Keep a held‑out test set; use task‑specific metrics (mAP for detection, IoU/Dice for segmentation).  
- **Export & serve:** Prefer **ONNX** → TensorRT/OpenVINO for edge latency. Quantize (INT8/FP16) where acceptable.  
- **Batching & tiling:** For big images, tile for inference; for video, batch frames and reuse pre/post steps.  
- **Repro:** Pin package versions; save `model.eval()` checkpoints and preprocessing configs.

---

*This guide is intentionally concise—use it as a launchpad for deeper dives when needed.*