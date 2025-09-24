# OpenCV Quick Reference

OpenCV (Open Source Computer Vision Library) is a comprehensive framework for computer vision, image processing, and machine learning tasks. It provides efficient implementations of thousands of algorithms for real-time computer vision applications.

### Installation
```bash
# Basic installation
pip install opencv-python

# With additional contrib modules
pip install opencv-contrib-python

# For headless servers (no GUI support)
pip install opencv-python-headless

# Full installation with extra modules
pip install opencv-contrib-python-headless
```

### Importing OpenCV
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Verify installation
print(cv2.__version__)
```

* * * * *

## 1. Image Loading and Display

```python
# Load image
img = cv2.imread('image.jpg')  # BGR format by default
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Display image (requires GUI environment)
cv2.imshow('Image', img)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

# Save image
cv2.imwrite('output.jpg', img)

# Image properties
height, width, channels = img.shape
print(f"Image size: {width}x{height}, Channels: {channels}")
```

## 2. Basic Image Operations

```python
# Resize image
resized = cv2.resize(img, (800, 600))  # (width, height)
resized_prop = cv2.resize(img, None, fx=0.5, fy=0.5)  # Scale by factor

# Crop image
cropped = img[100:300, 50:250]  # [y1:y2, x1:x2]

# Rotate image
center = (width//2, height//2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

# Flip image
flipped_horizontal = cv2.flip(img, 1)  # 1 for horizontal
flipped_vertical = cv2.flip(img, 0)    # 0 for vertical
```

## 3. Color Space Conversions

```python
# Convert between color spaces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Color thresholding
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img, img, mask=mask)
```

## 4. Image Filtering and Enhancement

```python
# Gaussian blur
blurred = cv2.GaussianBlur(img, (15, 15), 0)

# Median filter (removes salt-and-pepper noise)
median_filtered = cv2.medianBlur(img, 5)

# Bilateral filter (edge-preserving smoothing)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Sharpening kernel
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, kernel)

# Histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
```

## 5. Edge Detection and Morphological Operations

```python
# Edge detection
edges_canny = cv2.Canny(gray, 100, 200)
edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Morphological operations
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(gray, kernel, iterations=1)
dilation = cv2.dilate(gray, kernel, iterations=1)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
```

## 6. Feature Detection

```python
# Harris corner detection
gray_float = np.float32(gray)
corners_harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)

# SIFT features (Scale-Invariant Feature Transform)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
img_sift = cv2.drawKeypoints(img, keypoints, None)

# ORB features (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0,255,0))

# Feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
```

## 7. Object Detection and Template Matching

```python
# Template matching
template = cv2.imread('template.jpg', 0)
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h, w = template.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)

# Contour detection
contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)

# Hough line detection
lines = cv2.HoughLines(edges_canny, 1, np.pi/180, threshold=100)
if lines is not None:
    for rho, theta in lines[0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
```

## 8. Video Processing

```python
# Read from camera
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display frame
    cv2.imshow('Video', gray_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Read from video file
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Write video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
```

## 9. Face Detection and Recognition

```python
# Load pre-trained classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Detect eyes within face region
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```

## 10. Machine Learning Integration

```python
# K-means clustering for image segmentation
data = img.reshape((-1, 3))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
k = 8
_, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to uint8 and reshape to original image shape
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# Support Vector Machine for classification
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
result = svm.predict(test_data)
```

## 11. Advanced Image Processing

```python
# Image pyramids
# Gaussian pyramid (downsampling)
G = img.copy()
gp = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gp.append(G)

# Laplacian pyramid
lp = [gp[5]]
for i in range(5, 0, -1):
    size = (gp[i-1].shape[1], gp[i-1].shape[0])
    GE = cv2.pyrUp(gp[i], dstsize=size)
    L = cv2.subtract(gp[i-1], GE)
    lp.append(L)

# Watershed segmentation
dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
markers = cv2.watershed(img, markers)

# Optical flow
# Lucas-Kanade optical flow
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
```

## 12. Camera Calibration and 3D Vision

```python
# Camera calibration
# Prepare object points
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistort image
undistorted = cv2.undistort(img, mtx, dist, None, mtx)

# Stereo vision
# Stereo calibration and depth map computation
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
```

* * * * *

Summary
=======

- **Comprehensive computer vision library** with 2500+ algorithms
- **Multi-language support** (Python, C++, Java, JavaScript)
- **Real-time processing** optimized for performance
- **Hardware acceleration** support for GPU computing
- **Cross-platform** compatibility (Windows, Linux, macOS, Android, iOS)
- **Extensive documentation** and active community support
- **Machine learning integration** with built-in ML algorithms
- **Industry standard** used in autonomous vehicles, robotics, and AR/VR
- **Free and open source** with permissive BSD license
- **Active development** with regular updates and new features