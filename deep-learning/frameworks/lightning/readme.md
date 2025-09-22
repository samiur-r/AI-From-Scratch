# PyTorch Lightning Quick Reference

PyTorch Lightning is a high-level wrapper built on top of PyTorch that organizes PyTorch code to remove boilerplate and enable scalable, reproducible, and maintainable deep learning research. It provides a structured approach to building and training models while maintaining the flexibility of native PyTorch.

### Installation
```bash
# Install Lightning
pip install pytorch-lightning

# With specific PyTorch version
pip install pytorch-lightning torch==2.0.0

# With additional packages
pip install pytorch-lightning[extra]  # Includes wandb, tensorboard, etc.

# Verify installation
python -c "import pytorch_lightning as pl; print(pl.__version__)"
```

### Importing Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
```

* * * * *

## 1. Lightning Module Basics

```python
# Basic LightningModule structure
class SimpleLightningModel(LightningModule):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves all __init__ parameters

        # Define model layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics (optional)
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        """Define what happens in one training step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Define what happens in one validation step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)

        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Define what happens in one test step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log test metrics
        self.log('test_loss', loss, on_epoch=True)

        return {'test_loss': loss, 'logits': logits, 'y': y}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# Initialize model
model = SimpleLightningModel(input_size=784, hidden_size=256, num_classes=10)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

* * * * *

## 2. Lightning Data Module

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

class MNISTDataModule(LightningDataModule):
    """Lightning DataModule for MNIST dataset"""

    def __init__(self, data_dir='./data', batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        """Download data (called once per node)"""
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Setup datasets (called on every GPU)"""
        if stage == 'fit' or stage is None:
            mnist_full = torchvision.datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            # Split into train/val
            train_size = int(0.8 * len(mnist_full))
            val_size = len(mnist_full) - train_size
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [train_size, val_size]
            )

        if stage == 'test' or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# Initialize data module
data_module = MNISTDataModule(batch_size=128, num_workers=4)
```

* * * * *

## 3. Training with Lightning Trainer

```python
# Configure callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_top_k=3,
    filename='best-{epoch:02d}-{val_acc:.2f}',
    save_last=True
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Configure logger
logger = TensorBoardLogger(
    save_dir='lightning_logs',
    name='mnist_experiment',
    version='v1.0'
)

# Create trainer
trainer = Trainer(
    max_epochs=20,
    accelerator='auto',  # Automatically detect GPU/CPU
    devices='auto',      # Use all available devices

    # Callbacks
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],

    # Logger
    logger=logger,

    # Performance
    precision=16,        # Mixed precision training
    benchmark=True,      # Optimize for consistent input shapes
    deterministic=True,  # Reproducible results

    # Development options
    fast_dev_run=False,  # Set to True for quick debugging
    enable_progress_bar=True,
    enable_model_summary=True,

    # Validation
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
)

# Train the model
trainer.fit(model, data_module)

# Test the model
trainer.test(model, data_module, ckpt_path='best')

print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
```

* * * * *

## 4. Callbacks and Customization

```python
# Custom callback example
class CustomCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")

    def on_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} ended")

    def on_train_end(self, trainer, pl_module):
        print("Training finished!")

# Built-in callbacks
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    DeviceStatsMonitor,
    TQDMProgressBar
)

# Progress bar customization
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        batch_progress="green_yellow"
    )
)

# Device monitoring
device_stats = DeviceStatsMonitor()

# Configure trainer with multiple callbacks
trainer = Trainer(
    callbacks=[
        checkpoint_callback,
        early_stopping,
        lr_monitor,
        progress_bar,
        device_stats,
        CustomCallback()
    ]
)
```

* * * * *

## 5. Logging and Experiment Tracking

```python
# TensorBoard Logger
tensorboard_logger = TensorBoardLogger(
    save_dir='logs',
    name='my_experiment',
    version='v1'
)

# Weights & Biases Logger
wandb_logger = WandbLogger(
    project='my-project',
    name='experiment-1',
    tags=['baseline', 'mnist']
)

# MLflow Logger
from pytorch_lightning.loggers import MLFlowLogger
mlflow_logger = MLFlowLogger(
    experiment_name='mnist_experiments',
    tracking_uri='file:./ml-runs'
)

# Multiple loggers
trainer = Trainer(
    logger=[tensorboard_logger, wandb_logger],
    max_epochs=10
)

# Custom logging in Lightning Module
class LoggingModel(LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Basic logging
        self.log('train_loss', loss)

        # Advanced logging options
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)

        # Log with custom name
        self.log('my_custom_metric', loss.item() * 2)

        # Log hyperparameters
        self.log_dict({'lr': self.trainer.optimizers[0].param_groups[0]['lr']})

        # Log images (for TensorBoard/WandB)
        if batch_idx == 0:
            self.logger.experiment.add_image(
                'sample_images',
                x[0],
                self.current_epoch
            )

        return loss
```

* * * * *

## 6. Multi-GPU and Distributed Training

```python
# Single GPU
trainer = Trainer(
    accelerator='gpu',
    devices=1
)

# Multiple GPUs on single node
trainer = Trainer(
    accelerator='gpu',
    devices=4  # Use 4 GPUs
)

# All available GPUs
trainer = Trainer(
    accelerator='gpu',
    devices='auto'
)

# Distributed Data Parallel (DDP)
trainer = Trainer(
    accelerator='gpu',
    devices=4,
    strategy='ddp'
)

# DDP with multiple nodes
trainer = Trainer(
    accelerator='gpu',
    devices=4,
    num_nodes=2,
    strategy='ddp'
)

# DeepSpeed integration
trainer = Trainer(
    accelerator='gpu',
    devices=4,
    strategy='deepspeed_stage_2',
    precision=16
)

# TPU training
trainer = Trainer(
    accelerator='tpu',
    devices=8
)

# Model parallelism for large models
from pytorch_lightning.strategies import FSDPStrategy

trainer = Trainer(
    accelerator='gpu',
    devices=4,
    strategy=FSDPStrategy(
        auto_wrap_policy={nn.Linear},
        mixed_precision=True
    )
)
```

* * * * *

## 7. Model Checkpointing and Resuming

```python
# Basic checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    save_last=True
)

# Advanced checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_top_k=5,
    save_weights_only=False,
    filename='best-{epoch:02d}-{val_acc:.3f}',
    auto_insert_metric_name=False,
    every_n_epochs=1,
    save_on_train_epoch_end=False
)

# Resume training from checkpoint
trainer = Trainer(
    resume_from_checkpoint='path/to/checkpoint.ckpt'
)

# Or use ckpt_path in fit
trainer.fit(model, data_module, ckpt_path='path/to/checkpoint.ckpt')

# Load model from checkpoint
model = SimpleLightningModel.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    # Override hyperparameters if needed
    learning_rate=1e-4
)

# Manual saving and loading
trainer.save_checkpoint('my_checkpoint.ckpt')
```

* * * * *

## 8. Hyperparameter Tuning

```python
# Using Optuna with Lightning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Create model and data module
    model = SimpleLightningModel(
        hidden_size=hidden_size,
        learning_rate=lr,
        dropout=dropout
    )
    data_module = MNISTDataModule(batch_size=batch_size)

    # Pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_acc')

    # Trainer
    trainer = Trainer(
        max_epochs=10,
        callbacks=[pruning_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False
    )

    try:
        trainer.fit(model, data_module)
        return trainer.callback_metrics['val_acc'].item()
    except optuna.TrialPruned:
        raise optuna.TrialPruned()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")

# Ray Tune integration
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def tune_model(config):
    model = SimpleLightningModel(
        learning_rate=config['lr'],
        hidden_size=config['hidden_size']
    )

    trainer = Trainer(
        max_epochs=10,
        callbacks=[TuneReportCallback(['val_acc'], on='validation_end')],
        enable_progress_bar=False
    )

    trainer.fit(model, data_module)

# Define search space
config = {
    'lr': tune.loguniform(1e-5, 1e-1),
    'hidden_size': tune.choice([64, 128, 256, 512])
}

# Run tuning
analysis = tune.run(
    tune_model,
    config=config,
    num_samples=20,
    resources_per_trial={'cpu': 2, 'gpu': 0.5}
)
```

* * * * *

## 9. Custom Training Loops and Advanced Features

```python
# Custom training step with multiple optimizers
class GANLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()

    def configure_optimizers(self):
        # Multiple optimizers
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        # Train generator
        if optimizer_idx == 0:
            z = torch.randn(real_imgs.size(0), 100)
            fake_imgs = self.generator(z)
            fake_labels = torch.ones(real_imgs.size(0), 1)

            g_loss = self.criterion(self.discriminator(fake_imgs), fake_labels)
            self.log('g_loss', g_loss)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            # Real images
            real_labels = torch.ones(real_imgs.size(0), 1)
            real_loss = self.criterion(self.discriminator(real_imgs), real_labels)

            # Fake images
            z = torch.randn(real_imgs.size(0), 100)
            fake_imgs = self.generator(z).detach()
            fake_labels = torch.zeros(real_imgs.size(0), 1)
            fake_loss = self.criterion(self.discriminator(fake_imgs), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss)
            return d_loss

# Manual optimization
class ManualOptimizationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
        self.model = nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # Forward pass
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        # Manual backward
        self.manual_backward(loss)

        # Update weights
        opt.step()
        opt.zero_grad()

        self.log('train_loss', loss)

# Gradient accumulation
class GradientAccumulationModel(LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Scale loss by accumulation steps
        loss = loss / self.trainer.accumulate_grad_batches

        self.log('train_loss', loss)
        return loss

# Use gradient accumulation in trainer
trainer = Trainer(
    accumulate_grad_batches=4,  # Accumulate gradients over 4 batches
    max_epochs=10
)
```

* * * * *

## 10. Testing and Validation

```python
# Comprehensive testing
class TestingModel(LightningModule):
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return {
            'test_loss': loss,
            'test_acc': acc,
            'preds': preds,
            'targets': y
        }

    def test_epoch_end(self, outputs):
        # Aggregate results from all test steps
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        # Additional computations
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])

        # Compute confusion matrix or other metrics
        self.log('test_avg_loss', avg_loss)
        self.log('test_avg_acc', avg_acc)

# Cross-validation with Lightning
from sklearn.model_selection import KFold

def cross_validate_lightning(model_class, data, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create data loaders for this fold
        train_subset = torch.utils.data.Subset(data, train_ids)
        val_subset = torch.utils.data.Subset(data, val_ids)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Create fresh model
        model = model_class()

        # Train
        trainer = Trainer(max_epochs=10, enable_progress_bar=False)
        trainer.fit(model, train_loader, val_loader)

        # Test
        results = trainer.test(model, val_loader)
        cv_results.append(results[0])

    return cv_results

# Run cross-validation
results = cross_validate_lightning(SimpleLightningModel, dataset)
```

* * * * *

## 11. Production Deployment

```python
# Export Lightning model for production
class ProductionModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        """Define prediction logic"""
        x = batch
        return torch.softmax(self(x), dim=1)

# Load trained model
model = ProductionModel.load_from_checkpoint('best_model.ckpt')

# Convert to TorchScript
scripted_model = model.to_torchscript()
torch.jit.save(scripted_model, 'model_scripted.pt')

# ONNX export
input_sample = torch.randn(1, 784)
model.to_onnx('model.onnx', input_sample, export_params=True)

# Prediction with Lightning
predictions = trainer.predict(model, data_module)

# Create inference pipeline
class LightningInference:
    def __init__(self, checkpoint_path):
        self.model = ProductionModel.load_from_checkpoint(checkpoint_path)
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            if x.dim() == 1:
                x = x.unsqueeze(0)

            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            return {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy()
            }

# Use inference pipeline
inference = LightningInference('best_model.ckpt')
result = inference.predict(test_input)
```

* * * * *

Summary
=======

- **LightningModule** provides structured model organization with automatic training loops
- **LightningDataModule** encapsulates all data-related logic for clean separation
- **Trainer** handles complex training scenarios with minimal configuration
- **Callbacks** enable modular functionality like checkpointing and early stopping
- **Multi-GPU scaling** is seamless with automatic data and model parallelism
- **Experiment tracking** is built-in with support for multiple logging platforms
- **Hyperparameter tuning** integrates easily with Optuna, Ray Tune, and other tools
- **Production deployment** is straightforward with TorchScript and ONNX export
- **Reproducibility** is ensured through deterministic training and automatic logging