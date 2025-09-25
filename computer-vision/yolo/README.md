# YOLO (You Only Look Once) Quick Reference

YOLO is a real-time object detection framework that treats detection as a single regression problem, directly predicting bounding boxes and class probabilities from full images in one evaluation.

### Installation
```bash
# Install YOLOv8 (Ultralytics)
pip install ultralytics

# Install YOLOv5 (alternative)
pip install yolov5

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Importing YOLO
```python
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
```

* * * * *

## 1. Loading Pre-trained Models
```python
# Load different YOLO model sizes
model_nano = YOLO('yolov8n.pt')    # Nano - fastest, least accurate
model_small = YOLO('yolov8s.pt')   # Small
model_medium = YOLO('yolov8m.pt')  # Medium
model_large = YOLO('yolov8l.pt')   # Large
model_xlarge = YOLO('yolov8x.pt')  # Extra Large - slowest, most accurate

# Load custom trained model
model_custom = YOLO('path/to/custom/weights.pt')
```

## 2. Basic Object Detection
```python
# Single image detection
results = model('image.jpg')

# Multiple images
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# From numpy array
image_array = cv2.imread('image.jpg')
results = model(image_array)

# From PIL Image
pil_image = Image.open('image.jpg')
results = model(pil_image)
```

## 3. Processing Detection Results
```python
# Iterate through results
for result in results:
    # Get bounding boxes
    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    # Get class names
    class_names = result.names

    # Print detections
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        class_name = class_names[int(cls_id)]
        print(f"Detected {class_name} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

## 4. Visualization and Saving Results
```python
# Automatically annotate and display
annotated_frame = results[0].plot()
cv2.imshow('YOLO Detection', annotated_frame)
cv2.waitKey(0)

# Save annotated image
cv2.imwrite('detected_image.jpg', annotated_frame)

# Save results to file
results[0].save('output_directory/')
```

## 5. Video Processing and Real-time Detection
```python
# Process video file
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLO inference
        results = model(frame)

        # Visualize results
        annotated_frame = results[0].plot()

        # Display
        cv2.imshow("YOLO Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Real-time webcam detection
cap = cv2.VideoCapture(0)  # Use 0 for default camera
# ... same processing loop as above
```

## 6. Training Custom Models
```python
# Prepare dataset in YOLO format
# Directory structure:
# dataset/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── labels/
#       ├── train/
#       └── val/

# Create data.yaml file
data_yaml = """
path: /path/to/dataset
train: images/train
val: images/val

nc: 2  # number of classes
names: ['class1', 'class2']  # class names
"""

# Train model
model = YOLO('yolov8n.pt')  # Start with pre-trained model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_model'
)
```

## 7. Model Validation and Testing
```python
# Validate model performance
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Test on specific images
results = model('test_image.jpg')
results[0].show()  # Display results

# Benchmark model speed
model.benchmark()  # Shows FPS performance
```

## 8. Advanced Configuration
```python
# Prediction with custom parameters
results = model.predict(
    source='image.jpg',
    conf=0.25,        # Confidence threshold
    iou=0.45,         # NMS IoU threshold
    max_det=300,      # Maximum detections
    classes=[0, 2, 3], # Filter specific classes
    device='cuda:0'   # Use specific GPU
)

# Track objects across frames
results = model.track(source='video.mp4', tracker='bytetrack.yaml')
```

## 9. Export Models for Deployment
```python
# Export to different formats
model.export(format='onnx')        # ONNX format
model.export(format='torchscript') # TorchScript
model.export(format='tflite')      # TensorFlow Lite
model.export(format='coreml')      # Core ML (iOS)
model.export(format='engine')      # TensorRT

# Use exported model
onnx_model = YOLO('yolov8n.onnx')
results = onnx_model('image.jpg')
```

## 10. Performance Optimization
```python
# Use half precision for faster inference
model = YOLO('yolov8n.pt')
model.half()  # Convert to FP16

# Batch processing for better GPU utilization
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model(images, batch=8)

# Optimize image size for speed vs accuracy trade-off
results = model('image.jpg', imgsz=320)  # Smaller = faster
results = model('image.jpg', imgsz=1280) # Larger = more accurate
```

## 11. Working with Different YOLO Variants
```python
# Instance Segmentation (YOLOv8n-seg)
seg_model = YOLO('yolov8n-seg.pt')
results = seg_model('image.jpg')

# Extract segmentation masks
for result in results:
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        # Process masks...

# Classification (YOLOv8n-cls)
cls_model = YOLO('yolov8n-cls.pt')
results = cls_model('image.jpg')

# Pose Estimation (YOLOv8n-pose)
pose_model = YOLO('yolov8n-pose.pt')
results = pose_model('image.jpg')
```

## 12. Integration with Other Libraries
```python
# Integration with Roboflow datasets
from roboflow import Roboflow

rf = Roboflow(api_key="your_api_key")
project = rf.workspace("workspace").project("project")
dataset = project.version(1).download("yolov8")

# Train on Roboflow dataset
model = YOLO('yolov8n.pt')
results = model.train(data=f'{dataset.location}/data.yaml', epochs=100)

# Integration with Weights & Biases
import wandb

# Log training to W&B
wandb.login()
results = model.train(data='data.yaml', project='yolo-training', name='experiment-1')
```

* * * * *

Summary
=======

- **Real-time detection**: Single-pass object detection with excellent speed-accuracy trade-off
- **Multiple variants**: Object detection, segmentation, classification, and pose estimation
- **Easy deployment**: Export to multiple formats (ONNX, TensorRT, TensorFlow Lite)
- **Pre-trained models**: Ready-to-use models trained on COCO dataset with 80+ classes
- **Custom training**: Simple API for training on custom datasets
- **Video processing**: Built-in support for video files and real-time webcam inference
- **Active development**: Regular updates with state-of-the-art improvements
- **Community support**: Large ecosystem with extensive documentation and tutorials