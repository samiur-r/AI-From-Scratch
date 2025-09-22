# PyTorch Quick Reference

PyTorch is an open-source deep learning framework developed by Meta that provides a flexible and intuitive platform for building and training neural networks. It offers dynamic computational graphs, seamless GPU acceleration, and a Pythonic API that makes it popular for both research and production.

### Installation
```bash
# CPU only
pip install torch torchvision torchaudio

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Importing PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
```

* * * * *

## 1. Tensors and Basic Operations

```python
# Creating tensors
x = torch.tensor([1, 2, 3])  # From list
y = torch.zeros(3, 4)        # Zeros tensor
z = torch.ones(2, 3)         # Ones tensor
random = torch.randn(2, 3)   # Random normal distribution

# Tensor properties
print(x.shape)     # torch.Size([3])
print(x.dtype)     # torch.int64
print(x.device)    # cpu

# Move to GPU
if torch.cuda.is_available():
    x_gpu = x.cuda()  # or x.to('cuda')
    print(x_gpu.device)  # cuda:0

# Basic operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

add = a + b              # Element-wise addition
multiply = a * b         # Element-wise multiplication
dot_product = torch.dot(a, b)  # Dot product
matmul = torch.mm(a.unsqueeze(0), b.unsqueeze(1))  # Matrix multiplication

# Reshaping
x = torch.randn(4, 6)
y = x.view(2, 12)        # Reshape to 2x12
z = x.view(-1, 3)        # Reshape to ?x3 (automatic size)
```

* * * * *

## 2. Automatic Differentiation (Autograd)

```python
# Enable gradient computation
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Forward pass
z = x**2 + y**3
loss = z.sum()

# Backward pass
loss.backward()

# Access gradients
print(x.grad)  # dz/dx = 2*x = 4.0
print(y.grad)  # dz/dy = 3*y^2 = 27.0

# Gradient accumulation
x.grad.zero_()  # Clear gradients
z = x**3
z.backward()
print(x.grad)  # dz/dx = 3*x^2 = 12.0

# Disable gradients for inference
with torch.no_grad():
    result = x**2  # No gradient tracking

# Context for evaluation mode
torch.set_grad_enabled(False)
result = x**2  # No gradients
torch.set_grad_enabled(True)
```

* * * * *

## 3. Neural Network Modules

```python
# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN(784, 128, 10)
print(model)

# Model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Common layers
conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
batch_norm = nn.BatchNorm2d(64)
max_pool = nn.MaxPool2d(2, 2)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

# Sequential model
sequential_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

* * * * *

## 4. Data Loading and Preprocessing

```python
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10)
])

# Built-in datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Split dataset
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Iterate through data
for batch_idx, (data, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx}: Data shape {data.shape}, Targets shape {targets.shape}")
    if batch_idx == 0:  # Just show first batch
        break
```

* * * * *

## 5. Training and Optimization

```python
# Loss functions
criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.MSELoss()
criterion_binary = nn.BCEWithLogitsLoss()

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Training loop
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion_classification, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion_classification, device)

    scheduler.step()  # Update learning rate

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
```

* * * * *

## 6. Model Evaluation and Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print('Confusion Matrix:')
    print(cm)

    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_targets, all_preds))

    return accuracy, precision, recall, f1

# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pth'))
accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
```

* * * * *

## 7. Computer Vision with PyTorch

```python
import torchvision.models as models

# Pre-trained models
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

# Modify for different number of classes
num_classes = 10
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# Feature extraction (freeze pre-trained layers)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18.fc.requires_grad = True  # Only train final layer

# Custom CNN
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Image transforms for data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

* * * * *

## 8. Model Saving and Loading

```python
# Save model state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load model state dict
model = SimpleNN(784, 128, 10)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Save entire model
torch.save(model, 'complete_model.pth')

# Load entire model
model = torch.load('complete_model.pth')

# Save checkpoint with additional info
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

* * * * *

## 9. Model Deployment and Optimization

```python
# TorchScript for production
model.eval()

# Trace the model
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save('traced_model.pt')

# Load traced model
loaded_traced_model = torch.jit.load('traced_model.pt')

# Optimize for inference
optimized_model = torch.jit.optimize_for_inference(traced_model)

# ONNX export
torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Quantization for mobile deployment
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, targets in train_loader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

* * * * *

## 10. Advanced Features

```python
# Custom loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Custom learning rate scheduler
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.max_lr * self.step_count / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Hooks for debugging
def print_grad_norm(module, grad_input, grad_output):
    print(f'Gradient norm: {grad_output[0].norm()}')

# Register hook
model.fc.register_backward_hook(print_grad_norm)

# Model parallelism (multiple GPUs)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Distributed training setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model with DDP
model = DDP(model, device_ids=[local_rank])
```

* * * * *

## 11. PyTorch Ecosystem

```python
# PyTorch Lightning (high-level wrapper)
import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleNN(784, 128, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Hugging Face Transformers integration
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tensorboard logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/Train', train_loss, epoch)
writer.add_scalar('Accuracy/Train', train_acc, epoch)
writer.close()

# Weights & Biases integration
import wandb

wandb.init(project="my-project")
wandb.log({"train_loss": train_loss, "train_acc": train_acc})
```

* * * * *

Summary
=======

- **Dynamic graphs** make PyTorch intuitive for research and debugging
- **Autograd** automatically computes gradients for backpropagation
- **nn.Module** provides a clean way to define neural network architectures
- **DataLoader** efficiently handles data loading and batching
- **GPU acceleration** is seamless with `.cuda()` or `.to(device)`
- **TorchScript** optimizes models for production deployment
- **Rich ecosystem** includes computer vision, NLP, and audio libraries
- **Flexible design** supports custom layers, losses, and training loops
- **Production ready** with ONNX export and mobile optimization