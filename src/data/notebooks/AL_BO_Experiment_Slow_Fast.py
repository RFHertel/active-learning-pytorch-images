"""
optimized_active_learning_pipeline.py

High-performance active learning pipeline with memory optimizations for limited GPU RAM.
Optimized for RTX 3050 (4GB VRAM) with timing, exception handling, and resource monitoring.

Fixed configuration management with proper debug/production and slow/fast modes.
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
import psutil
import GPUtil
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
import logging
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('ActiveLearning')


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class ConfigManager:
    """Centralized configuration management to avoid multiple detection calls."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.mode = self._detect_mode()
            self.speed = 'slow'  # Default to slow mode
            self.config = self._get_config()
            self._print_configuration()
    
    def _detect_mode(self):
        """Detect if running in debug or production mode."""
        if sys.gettrace() is not None:
            return 'debug'
        if 'pydevd' in sys.modules or 'debugpy' in sys.modules:
            return 'debug'
        if os.environ.get('VSCODE_PID') or os.environ.get('TERM_PROGRAM') == 'vscode':
            if any(arg in sys.argv for arg in ['--debug', 'debug']):
                return 'debug'
        return 'production'
    
    def set_speed(self, speed):
        """Switch between slow and fast mode."""
        if speed not in ['slow', 'fast']:
            raise ValueError("Speed must be 'slow' or 'fast'")
        self.speed = speed
        self.config = self._get_config()
        logger.info(f"Switched to {speed} mode")
        self._print_configuration()
    
    def _get_config(self):
        """Get configuration based on current mode and speed."""
        if self.mode == 'debug':
            # Debug mode - always minimal configuration
            return {
                'batch_size': 64,
                'initial_labeled': 1000,
                'query_size': 500,
                'max_iterations': 5,
                'epochs_per_iteration': 5,
                'num_runs': 1,
                'num_workers': 0,  # No multiprocessing in debug
                'persistent_workers': False,
                'pin_memory': False,
                'prefetch_factor': None,
                'use_amp': False,  # No mixed precision in debug
                'strategies': ['random', 'least_confidence', 'entropy', 'bald', 'diverse_entropy'],
                'learning_rate': 0.01,
                'weight_decay': 5e-4,
                'momentum': 0.9
            }
        else:
            # Production mode - depends on speed setting
            if self.speed == 'slow':
                return {
                    'batch_size': 64,
                    'initial_labeled': 1000,
                    'query_size': 500,
                    'max_iterations': 5,
                    'epochs_per_iteration': 10,
                    'num_runs': 2,
                    'num_workers': 0,  # No multiprocessing in slow mode
                    'persistent_workers': False,
                    'pin_memory': torch.cuda.is_available(),
                    'prefetch_factor': None,
                    'use_amp': torch.cuda.is_available(),
                    'strategies': ['random', 'least_confidence', 'entropy', 'bald', 'diverse_entropy'],
                    'learning_rate': 0.01,
                    'weight_decay': 5e-4,
                    'momentum': 0.9
                }
            else:  # fast mode
                return {
                    'batch_size': 128,
                    'initial_labeled': 1000,
                    'query_size': 500,
                    'max_iterations': 5,
                    'epochs_per_iteration': 10,
                    'num_runs': 2,
                    'num_workers': 2,  # Multiprocessing enabled
                    'persistent_workers': True,
                    'pin_memory': torch.cuda.is_available(),
                    'prefetch_factor': 2,
                    'use_amp': torch.cuda.is_available(),
                    'strategies': ['random', 'least_confidence', 'entropy', 'bald', 'diverse_entropy'],
                    'learning_rate': 0.01,
                    'weight_decay': 5e-4,
                    'momentum': 0.9
                }
    
    def _print_configuration(self):
        """Print current configuration status."""
        print("\n" + "="*60)
        print(f"🔧 MODE: {self.mode.upper()} | SPEED: {self.speed.upper()}")
        print("="*60)
        logger.info(f"Configuration: {self.mode} mode, {self.speed} speed")
        logger.debug(f"Workers: {self.config['num_workers']}, Persistent: {self.config['persistent_workers']}")


# Create singleton instance
config_manager = ConfigManager()


# ============================================================================
# TIMING AND MONITORING UTILITIES
# ============================================================================

class PerformanceMonitor:
    """Monitor GPU memory, timing, and system resources."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_snapshots = []
        self.show_progress = True  # Control progress printing
        
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
            logger.info(f"{name}: {elapsed:.2f}s")
            if elapsed > 1.0:  # Only print for operations > 1 second
                print(f"  ⏱️  {name}: {elapsed:.2f}s")
    
    def check_gpu_memory(self, stage=""):
        """Check current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            # For RTX 3050 with 4GB
            if allocated > 3.5:
                logger.warning(f"High GPU memory at {stage}: {allocated:.2f}GB / 4GB")
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
                logger.error(f"GPU OUT OF MEMORY in {func.__name__}")
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
                logger.error(f"Error in {func.__name__}: {str(e)}")
                print(f"\n❌ Error in {func.__name__}: {str(e)}")
                traceback.print_exc()
                raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            print(f"\n❌ Unexpected error in {func.__name__}: {str(e)}")
            traceback.print_exc()
            raise
    return wrapper


# ============================================================================
# DATASET AND MODEL CLASSES (unchanged from before)
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
    """Memory-efficient CNN for RTX 3050 (4GB VRAM)."""
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(OptimizedCNN, self).__init__()
        
        # Reduced channel sizes for memory efficiency
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Block 3
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
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
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
        
        total_batches = len(unlabeled_loader)
        logger.info(f"Processing {total_batches} batches for uncertainty computation")
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                # Only log first, last, and every 100th batch
                if batch_idx == 0:
                    logger.debug(f"Started processing batches (1/{total_batches})")
                elif batch_idx == total_batches - 1:
                    logger.debug(f"Processing final batch ({total_batches}/{total_batches})")
                elif batch_idx % 100 == 0:
                    logger.debug(f"Processing batch {batch_idx}/{total_batches}")
                
                data = data.to(self.device, non_blocking=True)
                
                features = self.model.get_embeddings(data)
                logits = self.model.classifier(features)
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu())
                all_features.append(features.cpu())
                
                batch_size = data.size(0)
                indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        logger.info(f"Completed uncertainty computation for {len(indices)} samples")
        
        all_probs = torch.cat(all_probs)
        all_features = torch.cat(all_features)
        
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
            logger.info(f"Starting Least Confidence selection for {n_samples} samples")
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            
            max_probs = np.max(metrics['probs'], axis=1)
            confidences = 1 - max_probs
            
            selected_idx = np.argpartition(confidences, -n_samples)[-n_samples:]
            
            logger.info(f"Selected {len(selected_idx)} samples with confidence range: "
                       f"{confidences[selected_idx].min():.3f} - {confidences[selected_idx].max():.3f}")
            
            return [metrics['indices'][i] for i in selected_idx], confidences[selected_idx]


class EntropyStrategy(FastUncertaintyStrategy):
    """Optimized entropy strategy."""
    
    def select(self, unlabeled_loader, n_samples):
        with monitor.timer("Entropy Selection"):
            logger.info(f"Starting Entropy selection for {n_samples} samples")
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            
            probs = metrics['probs']
            log_probs = np.log(probs + 1e-10)
            entropy = -np.sum(probs * log_probs, axis=1)
            
            selected_idx = np.argpartition(entropy, -n_samples)[-n_samples:]
            
            logger.info(f"Selected {len(selected_idx)} samples with entropy range: "
                       f"{entropy[selected_idx].min():.3f} - {entropy[selected_idx].max():.3f}")
            
            return [metrics['indices'][i] for i in selected_idx], entropy[selected_idx]


class OptimizedBALDStrategy(QueryStrategy):
    """Memory-efficient BALD strategy for 4GB GPU."""
    
    def __init__(self, model, device, n_classes=10, n_dropout_samples=5):
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
            logger.info(f"Starting BALD selection for {n_samples} samples")
            self.model.eval()
            self.enable_dropout()
            
            chunk_size = 100
            all_bald_scores = []
            all_indices = []
            total_batches = len(unlabeled_loader)
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(unlabeled_loader):
                    if batch_idx % 50 == 0:
                        logger.debug(f"BALD processing batch {batch_idx}/{total_batches}")
                    
                    data = data.to(self.device, non_blocking=True)
                    batch_size = data.size(0)
                    
                    for chunk_start in range(0, batch_size, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, batch_size)
                        chunk_data = data[chunk_start:chunk_end]
                        
                        mc_predictions = []
                        for _ in range(self.n_dropout_samples):
                            outputs = self.model(chunk_data)
                            probs = F.softmax(outputs, dim=1)
                            mc_predictions.append(probs.unsqueeze(0))
                        
                        mc_predictions = torch.cat(mc_predictions, dim=0)
                        
                        mean_probs = mc_predictions.mean(dim=0)
                        expected_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
                        entropy_of_expected = -(mc_predictions * torch.log(mc_predictions + 1e-10)).sum(dim=2).mean(dim=0)
                        
                        bald = entropy_of_expected - expected_entropy
                        all_bald_scores.extend(bald.cpu().numpy())
                    
                    start_idx = batch_idx * unlabeled_loader.batch_size
                    all_indices.extend(range(start_idx, start_idx + batch_size))
                    
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
            
            all_bald_scores = np.array(all_bald_scores)
            selected_idx = np.argpartition(all_bald_scores, -n_samples)[-n_samples:]
            
            logger.info(f"Selected {len(selected_idx)} samples with BALD scores: "
                       f"{all_bald_scores[selected_idx].min():.3f} - {all_bald_scores[selected_idx].max():.3f}")
            
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
            logger.info(f"Starting Diversity-Aware selection for {n_samples} samples")
            
            uncertain_indices, uncertainty_scores = self.base_strategy.select(
                unlabeled_loader, min(n_samples * 3, len(unlabeled_loader.dataset))
            )
            
            if len(uncertain_indices) <= n_samples:
                return uncertain_indices, uncertainty_scores
            
            metrics = self.compute_all_uncertainties(unlabeled_loader)
            features = metrics['features']
            
            if features.shape[1] > 50:
                reducer = GaussianRandomProjection(n_components=50, random_state=42)
                features = reducer.fit_transform(features)
            
            uncertain_features = features[uncertain_indices]
            
            kmeans = MiniBatchKMeans(
                n_clusters=min(n_samples, len(uncertain_indices)),
                batch_size=100,
                n_init=3,
                max_iter=50,
                random_state=42
            )
            kmeans.fit(uncertain_features)
            
            selected = []
            for i in range(kmeans.n_clusters):
                cluster_mask = kmeans.labels_ == i
                if cluster_mask.any():
                    cluster_indices = np.array(uncertain_indices)[cluster_mask]
                    cluster_scores = uncertainty_scores[cluster_mask[:len(uncertainty_scores)]]
                    if len(cluster_scores) > 0:
                        best_idx = cluster_indices[np.argmax(cluster_scores)]
                        selected.append(best_idx)
            
            logger.info(f"Selected {len(selected)} diverse samples from {kmeans.n_clusters} clusters")
            
            return selected[:n_samples], uncertainty_scores[:n_samples]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class OptimizedActiveLearningPipeline:
    """Optimized pipeline for RTX 3050 (4GB VRAM) with timing and monitoring."""
    
    def __init__(self, custom_config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Show current configuration
        print(f"\n🔧 MODE: {config_manager.mode.upper()} | SPEED: {config_manager.speed.upper()}")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Use configuration from manager
        self.config = config_manager.config.copy()
        
        if custom_config:
            self.config.update(custom_config)
        
        # Performance monitoring
        self.monitor = monitor
        
        with self.monitor.timer("Data Setup"):
            self._setup_data()
        
        with self.monitor.timer("Strategy Setup"):
            self._setup_strategies()
        
        self.persistent_loaders = {}
    
    def _setup_data(self):
        """Setup CIFAR-10 datasets with proper transforms."""
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
        
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_train
        )
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform_test
        )
        
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            persistent_workers=self.config['persistent_workers'] if self.config['num_workers'] > 0 else False,
            pin_memory=self.config['pin_memory']
        )
        
        self.trainset_no_aug = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_test
        )
    
    def _setup_strategies(self):
        """Initialize available query strategies."""
        self.strategy_classes = {
            'random': None,
            'least_confidence': LeastConfidenceStrategy,
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
                
                optimizer.zero_grad(set_to_none=True)
                
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
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            total_loss += avg_loss
            
            epoch_time = time.perf_counter() - epoch_start
            
            if (epoch + 1) % max(1, epochs // 2) == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
                print(f"    Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={epoch_time:.1f}s")
       
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
        """Run a single active learning experiment with monitoring."""
        try:
            # Set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.benchmark = True
            
            # Print mode status
            print(f"\n{'='*60}")
            print(f"🔧 MODE: {config_manager.mode.upper()} | SPEED: {config_manager.speed.upper()}")
            print(f"Strategy: {strategy_name.upper()} - Run {run_id + 1}")
            print(f"{'='*60}")
            
            logger.info(f"Starting {strategy_name} run {run_id + 1}")
            
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
                logger.info(f"Iteration {iteration + 1}: {len(al_dataset.labeled_indices)} labeled")
                
                # Create data loader
                labeled_subset = al_dataset.get_labeled_subset()
                labeled_loader = DataLoader(
                    labeled_subset,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=self.config['num_workers'],
                    persistent_workers=self.config['persistent_workers'] if self.config['num_workers'] > 0 else False,
                    pin_memory=self.config['pin_memory'],
                    prefetch_factor=self.config['prefetch_factor'] if self.config['num_workers'] > 0 else None
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
                logger.info(f"Iteration {iteration + 1} complete: Acc={test_acc:.2f}%, Time={results['iteration_times'][-1]:.1f}s")
                
                # Query new samples
                if iteration < self.config['max_iterations'] - 1 and len(al_dataset.unlabeled_indices) > 0:
                    query_size = min(self.config['query_size'], len(al_dataset.unlabeled_indices))
                    
                    with self.monitor.timer(f"Query Selection (Iter {iteration+1})"):
                        if strategy_name == 'random':
                            logger.info(f"Random selection of {query_size} samples")
                            new_indices = random.sample(list(al_dataset.unlabeled_indices), query_size)
                        else:
                            logger.info(f"Starting {strategy_name} selection of {query_size} samples")
                            unlabeled_subset = Subset(self.trainset_no_aug, list(al_dataset.unlabeled_indices))
                            unlabeled_loader = DataLoader(
                                unlabeled_subset,
                                batch_size=self.config['batch_size'] * 2,
                                shuffle=False,
                                num_workers=self.config['num_workers'],
                                persistent_workers=self.config['persistent_workers'] if self.config['num_workers'] > 0 else False,
                                pin_memory=self.config['pin_memory']
                            )
                            
                            selected_relative, _ = strategy.select(unlabeled_loader, query_size)
                            
                            # Map back to original indices
                            unlabeled_list = list(al_dataset.unlabeled_indices)
                            new_indices = [unlabeled_list[i] for i in selected_relative 
                                        if i < len(unlabeled_list)]
                    
                    al_dataset.label_indices(new_indices)
                    logger.info(f"Added {len(new_indices)} new samples to labeled set")
                
                # Clear memory after query
                gc.collect()
                torch.cuda.empty_cache()
            
            return results
            
        finally:
            # Always cleanup
            self.cleanup_workers()
            gc.collect()
            torch.cuda.empty_cache()

    def cleanup_workers(self):
        """Explicitly cleanup all DataLoader workers."""
        for loader in list(self.persistent_loaders.values()):
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                loader._iterator._shutdown_workers()
                del loader._iterator
        self.persistent_loaders.clear()
        gc.collect()

    def run_comparison(self):
        """Run comparison of all strategies with full monitoring."""
        print("\n" + "="*80)
        print("OPTIMIZED ACTIVE LEARNING PIPELINE")
        print(f"🔧 MODE: {config_manager.mode.upper()} | SPEED: {config_manager.speed.upper()}")
        print("="*80)
        print("\nConfiguration:")
        for key, value in self.config.items():
            if value is not None:
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
                    logger.error(f"Failed run {run+1} for {strategy}: {str(e)}")
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
        logger.info(f"Pipeline complete in {total_time:.1f}s")
        
        return stats

    def _calculate_statistics(self, all_results):
        """Calculate mean and std for each strategy across runs."""
        stats = {}
        
        for strategy, runs in all_results.items():
            if not runs:
                continue
            
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
        colors = {
            'random': '#808080',
            'least_confidence': '#1f77b4',
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
                    error_y=dict(type='data', array=data['std_accuracy']) if len(data['std_accuracy']) > 0 else None,
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
        
        for strategy in ['random', 'least_confidence', 'entropy', 'bald', 'diverse_entropy']:
            if strategy in stats:
                final_accuracies.append(stats[strategy]['mean_accuracy'][-1])
                final_stds.append(stats[strategy]['std_accuracy'][-1] if len(stats[strategy]['std_accuracy']) > 0 else 0)
                strategies.append(strategy.replace('_', ' ').title())
        
        if strategies:
            fig.add_trace(go.Bar(
                x=strategies,
                y=final_accuracies,
                error_y=dict(type='data', array=final_stds) if any(final_stds) else None,
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
        fig.write_html('optimized_active_learning_comparison_experiment.html')
        webbrowser.open('optimized_active_learning_comparison_experiment.html')

    def _print_summary(self, stats):
        """Print comprehensive summary of results."""
        print(f"\n{'='*80}")
        print("ACTIVE LEARNING RESULTS SUMMARY")
        print(f"🔧 MODE: {config_manager.mode.upper()} | SPEED: {config_manager.speed.upper()}")
        print(f"{'='*80}")
        
        if not stats:
            print("No results to display.")
            return
        
        # Get final results
        final_results = {}
        for strategy, data in stats.items():
            final_results[strategy] = {
                'accuracy': data['mean_accuracy'][-1],
                'std': data['std_accuracy'][-1] if len(data['std_accuracy']) > 0 else 0,
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


def main():
   """Main function to run the optimized active learning pipeline."""
   
   print("="*80)
   print("OPTIMIZED ACTIVE LEARNING PIPELINE FOR CIFAR-10")
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
   
   # Option to switch modes
   if config_manager.mode == 'production':
       print("\nSpeed Mode Options:")
       print("  1. Slow (no multiprocessing, safer)")
       print("  2. Fast (multiprocessing enabled)")
       
       choice = input("\nSelect speed mode (1/2) [default=1]: ").strip()
       if choice == '2':
           config_manager.set_speed('fast')
       else:
           config_manager.set_speed('slow')
   
   # Create and run pipeline
   try:
       pipeline = OptimizedActiveLearningPipeline()
       results = pipeline.run_comparison()
       
       print("\n" + "="*80)
       print("✅ Experiment complete! Check 'optimized_active_learning_comparison_experiment.html' for visualizations.")
       print("="*80)
       
       return results
       
   except Exception as e:
       logger.error(f"Pipeline failed: {str(e)}")
       print(f"\n❌ Pipeline failed: {str(e)}")
       traceback.print_exc()
       return None


if __name__ == "__main__":
   results = main()