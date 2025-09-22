# TensorFlow Quick Reference

TensorFlow is an open-source machine learning framework developed by Google that provides a comprehensive ecosystem for building and deploying machine learning models at scale. It offers both high-level APIs (Keras) for rapid prototyping and low-level APIs for research and production optimization.

### Installation
```bash
# CPU only
pip install tensorflow

# GPU support (automatically includes CUDA dependencies)
pip install tensorflow[and-cuda]

# Development version
pip install tf-nightly

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

### Importing TensorFlow

```python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, optimizers, losses, metrics
import numpy as np
```

* * * * *

## 1. Tensors and Basic Operations

```python
# Creating tensors
x = tf.constant([1, 2, 3])              # From list
y = tf.zeros((3, 4))                    # Zeros tensor
z = tf.ones((2, 3))                     # Ones tensor
random = tf.random.normal((2, 3))       # Random normal

# Tensor properties
print(x.shape)     # TensorShape([3])
print(x.dtype)     # <dtype: 'int32'>
print(x.device)    # /job:localhost/replica:0/task:0/device:CPU:0

# GPU operations
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        x_gpu = tf.constant([1.0, 2.0, 3.0])
        print(x_gpu.device)

# Basic operations
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

add = tf.add(a, b)                      # or a + b
multiply = tf.multiply(a, b)            # or a * b
dot_product = tf.tensordot(a, b, axes=1)
matmul = tf.linalg.matmul(tf.expand_dims(a, 0), tf.expand_dims(b, 1))

# Reshaping
x = tf.random.normal((4, 6))
y = tf.reshape(x, (2, 12))              # Reshape to 2x12
z = tf.reshape(x, (-1, 3))              # Reshape to ?x3 (automatic size)

# Automatic differentiation
with tf.GradientTape() as tape:
    x = tf.Variable(2.0)
    y = x**2 + 3*x + 1

gradients = tape.gradient(y, x)
print(f"dy/dx = {gradients}")           # dy/dx = 7.0
```

* * * * *

## 2. Keras Sequential API

```python
# Simple Sequential model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Alternative Sequential syntax
model = tf.keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Count parameters
total_params = model.count_params()
print(f"Total parameters: {total_params:,}")
```

* * * * *

## 3. Keras Functional API

```python
# More flexible model definition
inputs = layers.Input(shape=(28, 28, 1))

# Convolutional layers
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Dense layers
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs, name='cnn_model')

# Multi-input/multi-output example
input1 = layers.Input(shape=(10,), name='input1')
input2 = layers.Input(shape=(10,), name='input2')

# Process inputs separately
x1 = layers.Dense(32, activation='relu')(input1)
x2 = layers.Dense(32, activation='relu')(input2)

# Combine inputs
combined = layers.concatenate([x1, x2])
output1 = layers.Dense(1, activation='sigmoid', name='output1')(combined)
output2 = layers.Dense(3, activation='softmax', name='output2')(combined)

model = Model(inputs=[input1, input2], outputs=[output1, output2])

# Compile with multiple losses
model.compile(
    optimizer='adam',
    loss={'output1': 'binary_crossentropy', 'output2': 'categorical_crossentropy'},
    metrics={'output1': ['accuracy'], 'output2': ['accuracy']}
)
```

* * * * *

## 4. Data Pipeline with tf.data

```python
# Create dataset from numpy arrays
import numpy as np

x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000,))

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Dataset operations
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Data preprocessing pipeline
def preprocess(image, label):
    # Normalize images
    image = tf.cast(image, tf.float32) / 255.0
    # One-hot encode labels
    label = tf.one_hot(label, depth=10)
    return image, label

dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Load image data
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Create dataset from file paths
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
labels = [0, 1, 2]

path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(labels)

dataset = tf.data.Dataset.zip((image_ds, label_ds))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Built-in datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def augment_data(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
```

* * * * *

## 5. Training and Optimization

```python
# Basic training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    verbose=1
)

# Custom optimizers
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduling
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

# Custom training loop
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    print(f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result():.4f}, '
          f'Accuracy: {train_accuracy.result() * 100:.2f}%')
```

* * * * *

## 6. Callbacks and Training Control

```python
# Model checkpointing
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Early stopping
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard logging
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Custom callback
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print(f"\nReached 95% accuracy at epoch {epoch + 1}, stopping training!")
            self.model.stop_training = True

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train with callbacks
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[
        checkpoint_cb,
        early_stopping_cb,
        reduce_lr_cb,
        tensorboard_cb,
        CustomCallback(),
        lr_scheduler_cb
    ],
    verbose=1
)

# View TensorBoard
# Run: tensorboard --logdir logs/fit
```

* * * * *

## 7. Model Evaluation and Metrics

```python
# Basic evaluation
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

# Predictions
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Custom metrics
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Compile with custom metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', F1Score()]
)

# Built-in metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        'categorical_accuracy',
        'top_k_categorical_accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
)

# Confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get true labels (assuming sparse labels)
y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

# Classification report
class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
print("Classification Report:")
print(classification_report(y_true, predicted_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

* * * * *

## 8. Model Saving and Loading

```python
# Save entire model
model.save('my_model.h5')
model.save('my_model')  # SavedModel format (recommended)

# Load model
loaded_model = tf.keras.models.load_model('my_model.h5')
loaded_model = tf.keras.models.load_model('my_model')

# Save/load weights only
model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')

# Save model architecture only
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Load architecture and weights separately
from tensorflow.keras.models import model_from_json

with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_weights.h5')

# SavedModel format (for TensorFlow Serving)
tf.saved_model.save(model, 'saved_model/')
loaded_model = tf.saved_model.load('saved_model/')

# Model checkpointing during training
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size  # Save every 5 batches
)

# Continue training from checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
```

* * * * *

## 9. Transfer Learning and Pre-trained Models

```python
# Load pre-trained model
base_model = tf.keras.applications.VGG16(
    weights='imagenet',  # Pre-trained on ImageNet
    include_top=False,   # Don't include classifier
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning: unfreeze and train with lower learning rate
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001/10),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Other pre-trained models
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
inception = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# TensorFlow Hub integration
import tensorflow_hub as hub

# Use pre-trained model from TensorFlow Hub
hub_model = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
    trainable=False
)

model = tf.keras.Sequential([
    hub_model,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

* * * * *

## 10. Distributed Training

```python
# Single machine, multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Build model within strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train normally - TensorFlow handles distribution
model.fit(train_ds, epochs=10, validation_data=val_ds)

# Multi-worker distributed training
import json
import os

# Set up cluster configuration
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    # Model definition and training
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, epochs=10, validation_data=val_ds)

# Parameter server strategy (for very large models)
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

# TPU strategy
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, epochs=10, validation_data=val_ds)
```

* * * * *

## 11. Model Optimization and Deployment

```python
# TensorFlow Lite conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Post-training quantization
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3)
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
quantized_tflite_model = converter.convert()

# TensorFlow.js conversion
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, 'path/to/tfjs_model')

# TensorRT optimization (for NVIDIA GPUs)
from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=8000000000
)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='saved_model/',
    conversion_params=conversion_params
)

converter.convert()
converter.save('tensorrt_model/')

# Inference with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# TensorFlow Serving setup
# 1. Save model in SavedModel format
tf.saved_model.save(model, 'serving_model/1/')

# 2. Start TensorFlow Serving (via Docker)
# docker run -p 8501:8501 --mount type=bind,source=/path/to/serving_model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving

# 3. Make predictions via REST API
import requests
import json

data = json.dumps({
    "signature_name": "serving_default",
    "instances": input_data.tolist()
})

headers = {"content-type": "application/json"}
response = requests.post('http://localhost:8501/v1/models/my_model:predict',
                        data=data, headers=headers)
predictions = json.loads(response.text)['predictions']
```

* * * * *

Summary
=======

- **Keras** provides user-friendly high-level APIs for rapid model development
- **tf.data** enables efficient and scalable data preprocessing pipelines
- **Eager execution** allows immediate operation execution for debugging and development
- **Graph execution** enables optimization and deployment advantages for production
- **Callbacks** provide modular training control with checkpointing, early stopping, and monitoring
- **Distributed training** scales seamlessly across multiple GPUs and machines
- **TensorBoard** offers comprehensive visualization and experiment tracking
- **Model optimization** supports deployment across web, mobile, and edge devices
- **Production ecosystem** includes TensorFlow Serving, TensorFlow Lite, and TensorFlow.js