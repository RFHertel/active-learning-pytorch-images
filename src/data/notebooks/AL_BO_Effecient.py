"""
optimized_active_learning_pipeline.py

High-performance active learning pipeline with memory optimizations for limited GPU RAM.
Optimized for RTX 3050 (4GB VRAM) with timing, exception handling, and resource monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import random
import time
import psutil # psutil (process and system utilities) is a cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors) in Python. It is useful mainly for system monitoring, profiling and limiting process resources and management of running processes.
import GPUtil # GPUtil is a Python module for getting the GPU status from NVIDA GPUs using nvidia-smi. GPUtil locates all GPUs on the computer, determines their availablity.
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial.distance import cdist
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import warnings
import traceback
import gc
import sys
import os
warnings.filterwarnings('ignore')

# Development/debugging configuration
DEBUG_CONFIG = {
    'batch_size': 32,          # Smaller for faster iteration
    'initial_labeled': 500,    # Less data for quick tests
    'query_size': 250,
    'max_iterations': 2,       # Just 2 iterations for testing
    'epochs_per_iteration': 2, # Quick training
    'num_runs': 1,             # Single run
    'num_workers': 0,          # No multiprocessing in debug
    'strategies': ['random', 'least_confidence']  # Test 2 strategies
}

# Production configuration  
PROD_CONFIG = {
    'batch_size': 64,
    'initial_labeled': 1000,
    'query_size': 500,
    'max_iterations': 5,
    'epochs_per_iteration': 10,
    'num_runs': 2,
    'num_workers': 2,
    'strategies': ['random', 'least_confidence', 'entropy', 'bald', 'diverse_entropy']
}

# Auto-detect mode for debugging or production:
# import sys
# if sys.gettrace() is not None:
#     print("🐛 Debug mode detected - using debug configuration")
#     config = DEBUG_CONFIG
# else:
#     print("🚀 Production mode - using full configuration")
#     config = PROD_CONFIG

# Improved debug mode detection: Check for debugger or VS Code environment variables
def is_debug_mode():
    if sys.gettrace() is not None:
        return True
    # Check for VS Code debugger environment variables
    if os.environ.get('VSCODE_PID') or os.environ.get('TERM_PROGRAM') == 'vscode':
        if '--debug' in sys.argv or 'ptvsd' in sys.modules or 'pydevd' in sys.modules:
            return True
    return False

if is_debug_mode():
    print("🐛 Debug mode detected - using debug configuration")
    config = DEBUG_CONFIG
else:
    print("🚀 Production mode - using full configuration")
    config = PROD_CONFIG


# ============================================================================
# TIMING AND MONITORING UTILITIES
# ============================================================================

class PerformanceMonitor:
    """Monitor GPU memory, timing, and system resources."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_snapshots = []
        
    @contextmanager
    def timer(self, name):
        """Context manager for timing code sections."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)
            print(f"  ⏱️  {name}: {elapsed:.2f}s")
    
    def check_gpu_memory(self, stage=""):
        """Check current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            # For RTX 3050 with 4GB
            if allocated > 3.5:
                print(f"  ⚠️  WARNING: High GPU memory usage at {stage}: {allocated:.2f}GB / 4GB")
                gc.collect()
                torch.cuda.empty_cache()
            
            self.memory_snapshots.append({
                'stage': stage,
                'allocated': allocated,
                'reserved': reserved
            })
            
            return allocated, reserved
        return 0, 0
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        for name, times in self.timings.items():
            avg_time = np.mean(times)
            total_time = np.sum(times)
            print(f"{name:30s}: {avg_time:.2f}s (avg) | {total_time:.2f}s (total)")
        
        if self.memory_snapshots:
            max_memory = max(s['allocated'] for s in self.memory_snapshots)
            print(f"\nPeak GPU Memory: {max_memory:.2f}GB / 4.0GB")


monitor = PerformanceMonitor()


def handle_exceptions(func):
    """Decorator for exception handling with cleanup."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ GPU OUT OF MEMORY in {func.__name__}")
                print("Attempting cleanup and retry with smaller batch...")
                
                # Clear GPU memory
                gc.collect()
                torch.cuda.empty_cache()
                
                # Reduce batch size and retry
                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = kwargs['batch_size'] // 2
                    print(f"Retrying with batch_size={kwargs['batch_size']}")
                    return func(*args, **kwargs)
                else:
                    raise
            else:
                print(f"\n❌ Error in {func.__name__}: {str(e)}")
                traceback.print_exc()
                raise
        except Exception as e:
            print(f"\n❌ Unexpected error in {func.__name__}: {str(e)}")
            traceback.print_exc()
            raise
    return wrapper


# ============================================================================
# OPTIMIZED DATASET AND MODEL CLASSES
# ============================================================================

class ActiveLearningDataset(Dataset):
    """Custom dataset wrapper for active learning that tracks labeled/unlabeled indices."""
    
    def __init__(self, base_dataset, initial_labeled_indices=None):
        self.base_dataset = base_dataset
        self.indices = list(range(len(base_dataset)))
        
        if initial_labeled_indices is None:
            self.labeled_indices = set()
            self.unlabeled_indices = set(self.indices)
        else:
            self.labeled_indices = set(initial_labeled_indices)
            self.unlabeled_indices = set(self.indices) - self.labeled_indices
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]
    
    def label_indices(self, indices):
        """Move indices from unlabeled to labeled set."""
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.add(idx)
    
    def get_labeled_subset(self):
        return Subset(self.base_dataset, list(self.labeled_indices))
    
    def get_unlabeled_subset(self):
        return Subset(self.base_dataset, list(self.unlabeled_indices))


class OptimizedCNN(nn.Module):
    """
    Memory-efficient CNN for RTX 3050 (4GB VRAM).
    Uses less memory than original while maintaining performance.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(OptimizedCNN, self).__init__()
        
        # Reduced channel sizes for memory efficiency
        self.features = nn.Sequential(
            # Block 1: 32x32x3 → 16x16x32 (reduced from 64)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Block 2: 16x16x32 → 8x8x64 (reduced from 128)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Block 3: 8x8x64 → 4x4x128 (reduced from 256)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Reduced classifier size
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),  # Reduced from 512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self.dropout_rate = dropout_rate
    
    def forward(self, x, return_features=False):
        features = self.features(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        
        if return_features:
            return flattened
        
        output = self.classifier(flattened)
        return output
    
    def get_embeddings(self, x):
        """Get intermediate embeddings for diversity-based sampling."""
        with torch.no_grad():
            return self.forward(x, return_features=True)


# ============================================================================
# OPTIMIZED QUERY STRATEGIES
# ============================================================================

class QueryStrategy:
    """Base class for query strategies."""
    
    def __init__(self, model, device, n_classes=10):
        self.model = model
        self.device = device
        self.n_classes = n_classes
    
    @handle_exceptions
    def select(self, unlabeled_loader, n_samples):
        raise NotImplementedError


class FastUncertaintyStrategy(QueryStrategy):
    """Optimized base class for uncertainty-based strategies."""
    
    @handle_exceptions
    def compute_all_uncertainties(self, unlabeled_loader):
        """Compute all uncertainty metrics in a single pass."""
        self.model.eval()
        
        all_probs = []
        all_features = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                # Non-blocking GPU transfer
                data = data.to(self.device, non_blocking=True)
                
                # Get features and predictions in one forward pass
                features = self.model.get_embeddings(data)
                logits = self.model.classifier(features)
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu())  # Move to CPU immediately to save GPU memory
                all_features.append(features.cpu())
                
                batch_size = data.size(0)
                indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
                
                # Clear intermediate GPU memory
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        all_probs = torch.cat(all_probs)
        all_features = torch.cat(all_features)
        
        # Compute all metrics on CPU to save GPU memory
        metrics = {
            'probs': all_probs.numpy(),
            'features': all_features.numpy(),
            'indices': indices
        }
        
        return metrics


class LeastConfidenceStrategy(FastUncertaintyStrategy):
    """Optimized least confidence strategy."""
    
    def select(self, unlabeled_loader, n_samples):
        with monitor.timer("Least Confidence Selection"):
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            
            # Compute on CPU
            max_probs = np.max(metrics['probs'], axis=1)
            confidences = 1 - max_probs
            
            # Use argpartition for O(n) selection instead of O(n log n) sorting
            selected_idx = np.argpartition(confidences, -n_samples)[-n_samples:]
            
            return [metrics['indices'][i] for i in selected_idx], confidences[selected_idx]


class MarginSamplingStrategy(FastUncertaintyStrategy):
    """Optimized margin sampling strategy."""
    
    def select(self, unlabeled_loader, n_samples):
        with monitor.timer("Margin Sampling Selection"):
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            
            # Sort probabilities
            sorted_probs = np.sort(metrics['probs'], axis=1)[:, ::-1]
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            
            selected_idx = np.argpartition(margins, n_samples)[:n_samples]
            
            return [metrics['indices'][i] for i in selected_idx], margins[selected_idx]


class EntropyStrategy(FastUncertaintyStrategy):
    """Optimized entropy strategy."""
    
    def select(self, unlabeled_loader, n_samples):
        with monitor.timer("Entropy Selection"):
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            
            # Vectorized entropy computation
            probs = metrics['probs']
            log_probs = np.log(probs + 1e-10)
            entropy = -np.sum(probs * log_probs, axis=1)
            
            selected_idx = np.argpartition(entropy, -n_samples)[-n_samples:]
            
            return [metrics['indices'][i] for i in selected_idx], entropy[selected_idx]


class OptimizedBALDStrategy(QueryStrategy):
    """Memory-efficient BALD strategy for 4GB GPU."""
    
    def __init__(self, model, device, n_classes=10, n_dropout_samples=5):  # Reduced from 10
        super().__init__(model, device, n_classes)
        self.n_dropout_samples = n_dropout_samples
    
    def enable_dropout(self):
        """Enable dropout during inference for MC-Dropout."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @handle_exceptions
    def select(self, unlabeled_loader, n_samples):
        with monitor.timer("BALD Selection"):
            self.model.eval()
            self.enable_dropout()
            
            # Process in smaller chunks for 4GB GPU
            chunk_size = 100  # Smaller chunks for limited memory
            all_bald_scores = []
            all_indices = []
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(unlabeled_loader):
                    data = data.to(self.device, non_blocking=True)
                    batch_size = data.size(0)
                    
                    # Process batch in chunks if needed
                    for chunk_start in range(0, batch_size, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, batch_size)
                        chunk_data = data[chunk_start:chunk_end]
                        
                        # MC-Dropout forward passes
                        mc_predictions = []
                        for _ in range(self.n_dropout_samples):
                            outputs = self.model(chunk_data)
                            probs = F.softmax(outputs, dim=1)
                            mc_predictions.append(probs.unsqueeze(0))
                        
                        mc_predictions = torch.cat(mc_predictions, dim=0)
                        
                        # Compute BALD scores
                        mean_probs = mc_predictions.mean(dim=0)
                        expected_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
                        
                        entropy_of_expected = -(mc_predictions * torch.log(mc_predictions + 1e-10)).sum(dim=2).mean(dim=0)
                        
                        bald = entropy_of_expected - expected_entropy
                        all_bald_scores.extend(bald.cpu().numpy())
                    
                    # Track indices
                    start_idx = batch_idx * unlabeled_loader.batch_size
                    all_indices.extend(range(start_idx, start_idx + batch_size))
                    
                    # Clear GPU memory periodically
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
            
            all_bald_scores = np.array(all_bald_scores)
            selected_idx = np.argpartition(all_bald_scores, -n_samples)[-n_samples:]
            
            return [all_indices[i] for i in selected_idx], all_bald_scores[selected_idx]


class FastDiversityStrategy(FastUncertaintyStrategy):
    """Memory-efficient diversity-aware strategy."""
    
    def __init__(self, model, device, base_strategy, n_classes=10, diversity_weight=0.5):
        super().__init__(model, device, n_classes)
        self.base_strategy = base_strategy
        self.diversity_weight = diversity_weight
    
    @handle_exceptions
    def select(self, unlabeled_loader, n_samples):
        with monitor.timer("Diversity-Aware Selection"):
            # Get uncertainty scores
            uncertain_indices, uncertainty_scores = self.base_strategy.select(
                unlabeled_loader, min(n_samples * 3, len(unlabeled_loader.dataset))
            )
            
            if len(uncertain_indices) <= n_samples:
                return uncertain_indices, uncertainty_scores
            
            # Get features efficiently
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            features = metrics['features']
            
            # Reduce dimensionality for faster clustering
            if features.shape[1] > 50:
                reducer = GaussianRandomProjection(n_components=50, random_state=42)
                features = reducer.fit_transform(features)
            
            # Get features of uncertain samples
            uncertain_features = features[uncertain_indices]
            
            # Use MiniBatchKMeans for speed and memory efficiency
            kmeans = MiniBatchKMeans(
                n_clusters=min(n_samples, len(uncertain_indices)),
                batch_size=100,
                n_init=3,
                max_iter=50,
                random_state=42
            )
            kmeans.fit(uncertain_features)
            
            # Select most uncertain from each cluster
            selected = []
            for i in range(kmeans.n_clusters):
                cluster_mask = kmeans.labels_ == i
                if cluster_mask.any():
                    cluster_indices = np.array(uncertain_indices)[cluster_mask]
                    cluster_scores = uncertainty_scores[cluster_mask[:len(uncertainty_scores)]]
                    if len(cluster_scores) > 0:
                        best_idx = cluster_indices[np.argmax(cluster_scores)]
                        selected.append(best_idx)
            
            return selected[:n_samples], uncertainty_scores[:n_samples]


# ============================================================================
# OPTIMIZED MAIN PIPELINE
# ============================================================================

class OptimizedActiveLearningPipeline:
    """Optimized pipeline for RTX 3050 (4GB VRAM) with timing and monitoring."""
    
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if gpu_memory < 4.5:
                print("⚠️  Limited GPU memory detected. Using optimized settings.")
                self._use_limited_memory_config = True
            else:
                self._use_limited_memory_config = False
        
        # Default configuration optimized for 4GB GPU
        self.config = {
            'batch_size': 64 if self._use_limited_memory_config else 128,  # Smaller for 4GB
            'initial_labeled': 1000,
            'query_size': 500,
            'max_iterations': 10,
            'epochs_per_iteration': 10,  # Reduced for faster execution
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'num_runs': 2,  # Reduced for faster testing
            'num_workers': 2,  # Reduced to avoid memory overhead
            'persistent_workers': True,
            'pin_memory': torch.cuda.is_available(),
            'prefetch_factor': 2,
            'use_amp': torch.cuda.is_available(),  # Mixed precision
            'strategies': ['random', 'least_confidence', 'entropy', 'bald', 'diverse_entropy']
        }
        
        if config:
            self.config.update(config)
        
        # Performance monitoring
        self.monitor = monitor
        
        with self.monitor.timer("Data Setup"):
            self._setup_data()
        
        with self.monitor.timer("Strategy Setup"):
            self._setup_strategies()

        # Create persistent loaders once
        self.persistent_loaders = {}
    
    def _setup_data(self):
        """Setup CIFAR-10 datasets with proper transforms."""
        # Data augmentation for training
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Load datasets
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_train
        )
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform_test
        )
        
        # Create persistent test loader
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.config['batch_size'] * 2,  # Larger batch for inference
            shuffle=False,
            num_workers=self.config['num_workers'],
            persistent_workers=self.config['persistent_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        # For feature extraction (without augmentation)
        self.trainset_no_aug = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_test
        )
    
    def _setup_strategies(self):
        """Initialize available query strategies."""
        self.strategy_classes = {
            'random': None,
            'least_confidence': LeastConfidenceStrategy,
            'margin': MarginSamplingStrategy,
            'entropy': EntropyStrategy,
            'bald': OptimizedBALDStrategy,
            'diverse_entropy': lambda m, d: FastDiversityStrategy(m, d, EntropyStrategy(m, d))
        }
    
    def get_balanced_initial_set(self, n_samples, n_classes=10):
        """Create a class-balanced initial labeled set."""
        class_indices = defaultdict(list)
        for idx in range(len(self.trainset)):
            _, label = self.trainset[idx]
            class_indices[label].append(idx)
        
        samples_per_class = n_samples // n_classes
        selected = []
        
        for class_label in range(n_classes):
            available = class_indices[class_label]
            class_samples = random.sample(available, min(samples_per_class, len(available)))
            selected.extend(class_samples)
        
        if len(selected) < n_samples:
            all_indices = set(range(len(self.trainset)))
            remaining = list(all_indices - set(selected))
            additional = random.sample(remaining, n_samples - len(selected))
            selected.extend(additional)
        
        return selected[:n_samples]
    
    @handle_exceptions
    def train_model(self, model, train_loader, epochs):
        """Optimized training with mixed precision and monitoring."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Mixed precision training for memory efficiency
        scaler = torch.cuda.amp.GradScaler() if self.config['use_amp'] else None
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient
                
                if self.config['use_amp']:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Memory cleanup every few batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            total_loss += avg_loss
            
            epoch_time = time.perf_counter() - epoch_start
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Train Acc={accuracy:.2f}%, Time={epoch_time:.1f}s")
        
        return total_loss / epochs
    
    @handle_exceptions
    def evaluate_model(self, model, data_loader):
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total
    
    @handle_exceptions
    def run_strategy(self, strategy_name, run_id=0, seed=42):
        try:    
            """Run a single active learning experiment with monitoring."""
            # Set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
            
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_name.upper()} - Run {run_id + 1}")
            print(f"{'='*60}")
            
            # Initialize model
            model = OptimizedCNN(num_classes=10).to(self.device)
            
            # Check initial memory
            self.monitor.check_gpu_memory(f"{strategy_name}_start")
            
            # Get initial labeled set
            with self.monitor.timer("Initial Set Selection"):
                initial_indices = self.get_balanced_initial_set(self.config['initial_labeled'])
                al_dataset = ActiveLearningDataset(self.trainset, initial_indices)
            
            # Initialize strategy
            if strategy_name != 'random':
                strategy_class = self.strategy_classes[strategy_name]
                strategy = strategy_class(model, self.device)
            
            # Track results
            results = {
                'labeled_counts': [],
                'test_accuracies': [],
                'train_losses': [],
                'iteration_times': []
            }
            
            for iteration in range(self.config['max_iterations']):
                iteration_start = time.perf_counter()
                
                print(f"\nIteration {iteration + 1}/{self.config['max_iterations']}")
                print(f"Labeled: {len(al_dataset.labeled_indices)}, "
                    f"Unlabeled: {len(al_dataset.unlabeled_indices)}")
                
                # Create data loader with persistent workers
                labeled_subset = al_dataset.get_labeled_subset()
                labeled_loader = DataLoader(
                    labeled_subset,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=self.config['num_workers'],
                    persistent_workers=self.config['persistent_workers'],
                    pin_memory=self.config['pin_memory'],
                    prefetch_factor=self.config['prefetch_factor']
                )
                
                # Train model
                with self.monitor.timer(f"Training (Iter {iteration+1})"):
                    train_loss = self.train_model(model, labeled_loader, self.config['epochs_per_iteration'])
                
                # Evaluate
                with self.monitor.timer(f"Evaluation (Iter {iteration+1})"):
                    test_acc = self.evaluate_model(model, self.test_loader)
                
                # Check memory
                self.monitor.check_gpu_memory(f"{strategy_name}_iter_{iteration}")
                
                results['labeled_counts'].append(len(al_dataset.labeled_indices))
                results['test_accuracies'].append(test_acc)
                results['train_losses'].append(train_loss)
                results['iteration_times'].append(time.perf_counter() - iteration_start)
                
                print(f"Results: Train Loss={train_loss:.4f}, Test Acc={test_acc:.2f}%")
                print(f"Iteration Time: {results['iteration_times'][-1]:.1f}s")
                
                # Query new samples
                if iteration < self.config['max_iterations'] - 1 and len(al_dataset.unlabeled_indices) > 0:
                    query_size = min(self.config['query_size'], len(al_dataset.unlabeled_indices))
                    
                    with self.monitor.timer(f"Query Selection (Iter {iteration+1})"):
                        if strategy_name == 'random':
                            new_indices = random.sample(list(al_dataset.unlabeled_indices), query_size)
                        else:
                            unlabeled_subset = Subset(self.trainset_no_aug, list(al_dataset.unlabeled_indices))
                            unlabeled_loader = DataLoader(
                            unlabeled_subset,
                            batch_size=self.config['batch_size'] * 2,  # Larger for inference
                            shuffle=False,
                            num_workers=self.config['num_workers'],
                            persistent_workers=self.config['persistent_workers'],
                            pin_memory=self.config['pin_memory']
                        )
                        
                            selected_relative, _ = strategy.select(unlabeled_loader, query_size)
                        
                        # Map back to original indices
                            unlabeled_list = list(al_dataset.unlabeled_indices)
                            new_indices = [unlabeled_list[i] for i in selected_relative 
                                        if i < len(unlabeled_list)]
                
                    al_dataset.label_indices(new_indices)
                
                # Clear memory after query
                    gc.collect()
                    torch.cuda.empty_cache()
        
            return results
        finally:
            # Always cleanup workers after each strategy
            self.cleanup_workers()
   
    def run_comparison(self):
        """Run comparison of all strategies with full monitoring."""
        print("\n" + "="*80)
        print("OPTIMIZED ACTIVE LEARNING PIPELINE")
        print("="*80)
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        all_results = {}
        total_start = time.perf_counter()
        
        for strategy in self.config['strategies']:
            print(f"\n{'#'*80}")
            print(f"Testing Strategy: {strategy.upper()}")
            print(f"{'#'*80}")
            
            strategy_results = []
            for run in range(self.config['num_runs']):
                try:
                    result = self.run_strategy(strategy, run, seed=42 + run)
                    strategy_results.append(result)
                except Exception as e:
                    print(f"❌ Failed run {run+1} for {strategy}: {str(e)}")
                    continue
            
            if strategy_results:
                all_results[strategy] = strategy_results
        
        total_time = time.perf_counter() - total_start
        
        # Calculate statistics
        stats = self._calculate_statistics(all_results)
        
        # Create visualizations
        self._create_plots(stats)
        
        # Print summary
        self._print_summary(stats)
        
        # Print performance summary
        self.monitor.print_summary()
        
        print(f"\n⏱️  Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        return stats

    def _calculate_statistics(self, all_results):
        """Calculate mean and std for each strategy across runs."""
        stats = {}
        
        for strategy, runs in all_results.items():
            if not runs:
                continue
                
            # Stack results from all runs
            accuracies = np.array([run['test_accuracies'] for run in runs])
            losses = np.array([run['train_losses'] for run in runs])
            times = np.array([run['iteration_times'] for run in runs])
            
            stats[strategy] = {
                'labeled_counts': runs[0]['labeled_counts'],
                'mean_accuracy': np.mean(accuracies, axis=0),
                'std_accuracy': np.std(accuracies, axis=0),
                'mean_loss': np.mean(losses, axis=0),
                'std_loss': np.std(losses, axis=0),
                'mean_time': np.mean(times, axis=0),
                'total_time': np.sum(times)
            }
        
        return stats

    def _create_plots(self, stats):
        """Create comprehensive visualization of results."""
        # Color palette
        colors = {
            'random': '#808080',
            'least_confidence': '#1f77b4',
            'margin': '#ff7f0e',
            'entropy': '#2ca02c',
            'bald': '#d62728',
            'diverse_entropy': '#9467bd'
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Test Accuracy vs Labeled Samples',
                'Relative Improvement over Random',
                'Time per Iteration',
                'Final Accuracy Comparison'
            )
        )
        
        # Plot 1: Test Accuracy
        for strategy, data in stats.items():
            if strategy in colors:
                fig.add_trace(go.Scatter(
                    x=data['labeled_counts'],
                    y=data['mean_accuracy'],
                    error_y=dict(type='data', array=data['std_accuracy']),
                    mode='lines+markers',
                    name=strategy.replace('_', ' ').title(),
                    line=dict(color=colors.get(strategy, '#808080'), width=2),
                    marker=dict(size=8)
                ), row=1, col=1)
        
        # Plot 2: Relative Improvement
        if 'random' in stats:
            random_acc = stats['random']['mean_accuracy']
            for strategy, data in stats.items():
                if strategy != 'random' and strategy in colors:
                    improvement = ((data['mean_accuracy'] - random_acc) / random_acc) * 100
                    fig.add_trace(go.Scatter(
                        x=data['labeled_counts'],
                        y=improvement,
                        mode='lines+markers',
                        name=strategy.replace('_', ' ').title(),
                        line=dict(color=colors.get(strategy, '#808080'), width=2),
                        marker=dict(size=8)
                    ), row=1, col=2)
        
        # Plot 3: Time per Iteration
        for strategy, data in stats.items():
            if strategy in colors:
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(data['mean_time']) + 1)),
                    y=data['mean_time'],
                    mode='lines+markers',
                    name=strategy.replace('_', ' ').title(),
                    line=dict(color=colors.get(strategy, '#808080'), width=1),
                    showlegend=False
                ), row=2, col=1)
        
        # Plot 4: Final Accuracy Bar Chart
        final_accuracies = []
        final_stds = []
        strategies = []
        
        for strategy in ['random', 'least_confidence', 'margin', 'entropy', 'bald', 'diverse_entropy']:
            if strategy in stats:
                final_accuracies.append(stats[strategy]['mean_accuracy'][-1])
                final_stds.append(stats[strategy]['std_accuracy'][-1])
                strategies.append(strategy.replace('_', ' ').title())
        
        if strategies:
            fig.add_trace(go.Bar(
                x=strategies,
                y=final_accuracies,
                error_y=dict(type='data', array=final_stds),
                marker_color=[colors.get(s.lower().replace(' ', '_'), '#808080') for s in strategies],
                showlegend=False
            ), row=2, col=2)
        
        # Update layout
        fig.update_xaxes(title_text="Number of Labeled Samples", row=1, col=1)
        fig.update_xaxes(title_text="Number of Labeled Samples", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Strategy", row=2, col=2)
        
        fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Final Accuracy (%)", row=2, col=2)
        
        fig.update_layout(
            title_text='Optimized Active Learning Strategy Comparison (CIFAR-10)',
            height=800,
            width=1400,
            template='plotly_white',
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        # Save and open
        fig.write_html('optimized_active_learning_comparisons.html')
        webbrowser.open('optimized_active_learning_comparisons.html')

    def _print_summary(self, stats):
        """Print comprehensive summary of results."""
        print(f"\n{'='*80}")
        print("ACTIVE LEARNING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        if not stats:
            print("No results to display.")
            return
        
        # Get final results
        final_results = {}
        for strategy, data in stats.items():
            final_results[strategy] = {
                'accuracy': data['mean_accuracy'][-1],
                'std': data['std_accuracy'][-1],
                'time': data['total_time']
            }
        
        # Sort by accuracy
        sorted_strategies = sorted(final_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\nFinal Accuracies (after all iterations):")
        print("-" * 50)
        
        if 'random' in final_results:
            for strategy, result in sorted_strategies:
                improvement = result['accuracy'] - final_results['random']['accuracy']
                print(f"{strategy.replace('_', ' ').title():20s}: "
                    f"{result['accuracy']:.2f}% ± {result['std']:.2f}% "
                    f"(Δ: {improvement:+.2f}%) "
                    f"Time: {result['time']:.1f}s")
        
        print("\nDetailed Performance Analysis:")
        print("-" * 50)
        
        # Calculate area under curve (AUC) for each strategy
        aucs = {}
        for strategy, data in stats.items():
            x = np.array(data['labeled_counts'])
            y = np.array(data['mean_accuracy'])
            auc = np.trapz(y, x) / (x[-1] - x[0])
            aucs[strategy] = auc
        
        if 'random' in aucs:
            random_auc = aucs['random']
            for strategy in sorted(aucs.keys()):
                if strategy != 'random':
                    improvement = ((aucs[strategy] - random_auc) / random_auc) * 100
                    print(f"{strategy.replace('_', ' ').title():20s}: "
                            f"AUC = {aucs[strategy]:.2f}, "
                            f"Improvement = {improvement:+.2f}%")
        
        # Early stopping analysis
        print("\nEarly Performance (25% of data labeled):")
        print("-" * 50)
        
        if stats and 'random' in stats:
            quarter_idx = len(stats['random']['labeled_counts']) // 4
            for strategy in sorted_strategies:
                strat_name = strategy[0]
                if strat_name in stats:
                    early_acc = stats[strat_name]['mean_accuracy'][quarter_idx]
                    early_std = stats[strat_name]['std_accuracy'][quarter_idx]
                    early_improvement = early_acc - stats['random']['mean_accuracy'][quarter_idx]
                    print(f"{strat_name.replace('_', ' ').title():20s}: "
                            f"{early_acc:.2f}% ± {early_std:.2f}% "
                            f"(Δ: {early_improvement:+.2f}%)")
        
        # Best performing strategy
        if sorted_strategies and 'random' in final_results:
            best_strategy = sorted_strategies[0][0]
            best_improvement = final_results[best_strategy]['accuracy'] - final_results['random']['accuracy']
            
            print(f"\n{'='*80}")
            print(f"CONCLUSION:")
            print(f"{'='*80}")
            print(f"✅ Best Strategy: {best_strategy.replace('_', ' ').title()}")
            print(f"✅ Improvement over Random: {best_improvement:+.2f}%")
            
            if best_improvement > 2.0:
                print("✅ Active Learning is significantly outperforming random sampling!")
            elif best_improvement > 0.5:
                print("✅ Active Learning shows modest improvements over random sampling.")
            else:
                print("⚠️  Active Learning shows minimal improvement. Consider:")
                print("   - Using a larger query budget")
                print("   - Implementing more sophisticated strategies")
                print("   - Checking if the model is already near optimal performance")

    # For Threading:
    def _create_persistent_test_loader(self):
        """Create a single persistent test loader to reuse."""
        if 'test' not in self.persistent_loaders:
            self.persistent_loaders['test'] = DataLoader(
                self.testset,
                batch_size=self.config['batch_size'] * 2,
                shuffle=False,
                num_workers=self.config['num_workers'],
                persistent_workers=self.config['persistent_workers'],
                pin_memory=self.config['pin_memory']
            )
        return self.persistent_loaders['test']
    
    def cleanup_workers(self):
        """Explicitly cleanup all DataLoader workers."""
        for loader in self.persistent_loaders.values():
            if hasattr(loader, '_iterator'):
                del loader._iterator
        self.persistent_loaders.clear()
        
        # Force garbage collection
        import gc
        gc.collect()


def main():
   """Main function to run the optimized active learning pipeline."""
   
   # Configuration optimized for RTX 3050 (4GB VRAM)
   config = {
       'batch_size': 64,           # Reduced for 4GB GPU
       'initial_labeled': 1000,    # 2% of CIFAR-10 training data
       'query_size': 500,          # Query 1% at each iteration
       'max_iterations': 5,        # Reduced for faster testing
       'epochs_per_iteration': 10, # Reduced for faster execution
       'learning_rate': 0.01,
       'weight_decay': 5e-4,
       'momentum': 0.9,
       'num_runs': 2,              # Reduced for faster testing
       'strategies': [
           'random',
           'least_confidence',
           'entropy',
           'bald',
           'diverse_entropy'
       ]
   }
   
   print("="*80)
   print("OPTIMIZED ACTIVE LEARNING PIPELINE FOR CIFAR-10")
   print("Optimized for RTX 3050 (4GB VRAM)")
   print("="*80)
   
   # System information
   print("\nSystem Information:")
   print(f"  CPU: {psutil.cpu_count()} cores")
   print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
   
   if torch.cuda.is_available():
       print(f"  GPU: {torch.cuda.get_device_name(0)}")
       print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
       print(f"  CUDA Version: {torch.version.cuda}")
       print(f"  PyTorch Version: {torch.__version__}")
   
   print("\nConfiguration:")
   for key, value in config.items():
       print(f"  {key}: {value}")
   print()
   
   # Create and run pipeline
   try:
       pipeline = OptimizedActiveLearningPipeline(config)
       results = pipeline.run_comparison()
       
       print("\n" + "="*80)
       print("✅ Experiment complete! Check 'optimized_active_learning_comparisons.html' for visualizations.")
       print("="*80)
       
       return results
       
   except Exception as e:
       print(f"\n❌ Pipeline failed: {str(e)}")
       traceback.print_exc()
       return None


if __name__ == "__main__":
   results = main()