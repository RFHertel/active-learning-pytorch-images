"""
active_learning_pipeline.py

A robust active learning pipeline for CIFAR-10 with multiple query strategies,
proper uncertainty estimation, and diversity-aware sampling.
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
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')



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


class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture with better feature extraction and uncertainty estimation.
    Includes dropout layers for MC-Dropout and intermediate feature extraction.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
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


class QueryStrategy:
    """Base class for query strategies."""
    
    def __init__(self, model, device, n_classes=10):
        self.model = model
        self.device = device
        self.n_classes = n_classes
    
    def select(self, unlabeled_loader, n_samples):
        raise NotImplementedError


class LeastConfidenceStrategy(QueryStrategy):
    """Select samples where the model is least confident about its prediction."""
    
    def select(self, unlabeled_loader, n_samples):
        self.model.eval()
        confidences = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(self.device)
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                
                # Least confidence: 1 - max(p_i)
                max_probs, _ = torch.max(probs, dim=1)
                confidences.extend((1 - max_probs).cpu().numpy())
                
                # Track original indices
                batch_size = data.size(0)
                indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        
        # Select top n_samples with lowest confidence
        confidences = np.array(confidences)
        selected_idx = np.argsort(confidences)[-n_samples:]
        
        return [indices[i] for i in selected_idx], confidences[selected_idx]


class MarginSamplingStrategy(QueryStrategy):
    """Select samples with smallest margin between top two predictions."""
    
    def select(self, unlabeled_loader, n_samples):
        self.model.eval()
        margins = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(self.device)
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                
                # Sort probabilities
                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                
                # Margin: p1 - p2 (smaller margin = more uncertain)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                margins.extend(margin.cpu().numpy())
                
                batch_size = data.size(0)
                indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        
        margins = np.array(margins)
        selected_idx = np.argsort(margins)[:n_samples]
        
        return [indices[i] for i in selected_idx], margins[selected_idx]


class EntropyStrategy(QueryStrategy):
    """Select samples with highest entropy in predictions."""
    
    def select(self, unlabeled_loader, n_samples):
        self.model.eval()
        entropies = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(self.device)
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                
                # Entropy: -sum(p_i * log(p_i))
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=1)
                entropies.extend(entropy.cpu().numpy())
                
                batch_size = data.size(0)
                indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        
        entropies = np.array(entropies)
        selected_idx = np.argsort(entropies)[-n_samples:]
        
        return [indices[i] for i in selected_idx], entropies[selected_idx]


class BALDStrategy(QueryStrategy):
    """Bayesian Active Learning by Disagreement - measures epistemic uncertainty."""
    
    def __init__(self, model, device, n_classes=10, n_dropout_samples=10):
        super().__init__(model, device, n_classes)
        self.n_dropout_samples = n_dropout_samples
    
    def enable_dropout(self):
        """Enable dropout during inference for MC-Dropout."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    def select(self, unlabeled_loader, n_samples):
        self.model.eval()
        self.enable_dropout()
        
        bald_scores = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Multiple forward passes with dropout
                all_probs = []
                for _ in range(self.n_dropout_samples):
                    outputs = self.model(data)
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs.unsqueeze(0))
                
                all_probs = torch.cat(all_probs, dim=0)  # [n_dropout, batch, n_classes]
                
                # Expected entropy (aleatoric uncertainty)
                mean_probs = all_probs.mean(dim=0)
                expected_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
                
                # Entropy of expectation (total uncertainty)
                entropy_of_expected = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2).mean(dim=0)
                
                # BALD score (epistemic uncertainty)
                bald = entropy_of_expected - expected_entropy
                bald_scores.extend(bald.cpu().numpy())
                
                indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        
        bald_scores = np.array(bald_scores)
        selected_idx = np.argsort(bald_scores)[-n_samples:]
        
        return [indices[i] for i in selected_idx], bald_scores[selected_idx]


class DiversityAwareStrategy(QueryStrategy):
    """
    Combines uncertainty sampling with diversity using clustering.
    Ensures selected samples are both uncertain and diverse.
    """
    
    def __init__(self, model, device, base_strategy, n_classes=10, diversity_weight=0.5):
        super().__init__(model, device, n_classes)
        self.base_strategy = base_strategy
        self.diversity_weight = diversity_weight
    
    def select(self, unlabeled_loader, n_samples):
        # First, get uncertainty scores from base strategy
        uncertain_indices, uncertainty_scores = self.base_strategy.select(
            unlabeled_loader, min(n_samples * 3, len(unlabeled_loader.dataset))
        )
        
        if len(uncertain_indices) <= n_samples:
            return uncertain_indices, uncertainty_scores
        
        # Extract features for diversity calculation
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for data, _ in unlabeled_loader:
                data = data.to(self.device)
                feat = self.model.get_embeddings(data)
                features.append(feat.cpu().numpy())
        
        features = np.vstack(features)
        
        # Get features of uncertain samples
        uncertain_features = features[uncertain_indices]
        
        # Use K-means to ensure diversity
        kmeans = KMeans(n_clusters=min(n_samples, len(uncertain_indices)), 
                        random_state=42, n_init=10)
        kmeans.fit(uncertain_features)
        
        # Select closest point to each cluster center
        selected = []
        for center in kmeans.cluster_centers_:
            distances = cdist([center], uncertain_features)[0]
            closest_idx = np.argmin(distances)
            if uncertain_indices[closest_idx] not in selected:
                selected.append(uncertain_indices[closest_idx])
        
        # Fill remaining slots with most uncertain samples
        remaining = [idx for idx in uncertain_indices if idx not in selected]
        remaining_scores = [uncertainty_scores[i] for i, idx in enumerate(uncertain_indices) 
                          if idx not in selected]
        
        if remaining and len(selected) < n_samples:
            sorted_idx = np.argsort(remaining_scores)[-( n_samples - len(selected)):]
            selected.extend([remaining[i] for i in sorted_idx])
        
        return selected[:n_samples], uncertainty_scores[:n_samples]


class ActiveLearningPipeline:
    """Main active learning pipeline orchestrating the entire process."""
    
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Default configuration
        self.config = {
            'batch_size': 128,
            'initial_labeled': 1000,  # Start with 2% of data
            'query_size': 500,        # Query 1% at a time
            'max_iterations': 10,
            'epochs_per_iteration': 20,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'num_runs': 3,
            'strategies': ['random', 'least_confidence', 'margin', 'entropy', 'bald', 'diverse_entropy']
        }
        
        if config:
            self.config.update(config)
        
        self._setup_data()
        self._setup_strategies()
    
    def _setup_data(self):
        """Setup CIFAR-10 datasets with proper transforms."""
        # Data augmentation for training
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        
        self.test_loader = DataLoader(
            self.testset, batch_size=self.config['batch_size'], shuffle=False
        )
        
        # For feature extraction (without augmentation)
        self.trainset_no_aug = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_test
        )
    
    def _setup_strategies(self):
        """Initialize available query strategies."""
        self.strategy_classes = {
            'random': None,  # Special case
            'least_confidence': LeastConfidenceStrategy,
            'margin': MarginSamplingStrategy,
            'entropy': EntropyStrategy,
            'bald': BALDStrategy,
            'diverse_entropy': lambda m, d: DiversityAwareStrategy(m, d, EntropyStrategy(m, d))
        }
    
    def get_balanced_initial_set(self, n_samples, n_classes=10):
        """Create a class-balanced initial labeled set."""
        # Group indices by class
        class_indices = defaultdict(list)
        for idx in range(len(self.trainset)):
            _, label = self.trainset[idx]
            class_indices[label].append(idx)
        
        # Sample equally from each class
        samples_per_class = n_samples // n_classes
        selected = []
        
        for class_label in range(n_classes):
            available = class_indices[class_label]
            class_samples = random.sample(available, min(samples_per_class, len(available)))
            selected.extend(class_samples)
        
        # Fill remaining slots if needed
        if len(selected) < n_samples:
            all_indices = set(range(len(self.trainset)))
            remaining = list(all_indices - set(selected))
            additional = random.sample(remaining, n_samples - len(selected))
            selected.extend(additional)
        
        return selected[:n_samples]
    
    def train_model(self, model, train_loader, epochs):
        """Train the model with cosine annealing and proper optimization."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            total_loss += avg_loss
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={accuracy:.2f}%")
        
        return total_loss / epochs
    
    def evaluate_model(self, model, data_loader):
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total
    
    def run_strategy(self, strategy_name, run_id=0, seed=42):
        """Run a single active learning experiment with a specific strategy."""
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name.upper()} - Run {run_id + 1}")
        print(f"{'='*60}")
        
        # Initialize model
        model = ImprovedCNN(num_classes=10).to(self.device)
        
        # Get initial labeled set
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
            'train_losses': []
        }
        
        for iteration in range(self.config['max_iterations']):
            print(f"\nIteration {iteration + 1}/{self.config['max_iterations']}")
            print(f"Labeled: {len(al_dataset.labeled_indices)}, "
                  f"Unlabeled: {len(al_dataset.unlabeled_indices)}")
            
            # Create data loader for labeled data
            labeled_subset = al_dataset.get_labeled_subset()
            labeled_loader = DataLoader(
                labeled_subset, batch_size=self.config['batch_size'],
                shuffle=True, num_workers=2
            )
            
            # Train model
            train_loss = self.train_model(model, labeled_loader, self.config['epochs_per_iteration'])
            
            # Evaluate
            test_acc = self.evaluate_model(model, self.test_loader)
            
            results['labeled_counts'].append(len(al_dataset.labeled_indices))
            results['test_accuracies'].append(test_acc)
            results['train_losses'].append(train_loss)
            
            print(f"Results: Train Loss={train_loss:.4f}, Test Acc={test_acc:.2f}%")
            
            # Query new samples
            if iteration < self.config['max_iterations'] - 1 and len(al_dataset.unlabeled_indices) > 0:
                query_size = min(self.config['query_size'], len(al_dataset.unlabeled_indices))
                
                if strategy_name == 'random':
                    # Random sampling
                    new_indices = random.sample(list(al_dataset.unlabeled_indices), query_size)
                else:
                    # Active learning strategy
                    unlabeled_subset = Subset(self.trainset_no_aug, list(al_dataset.unlabeled_indices))
                    unlabeled_loader = DataLoader(
                        unlabeled_subset, batch_size=self.config['batch_size'],
                        shuffle=False, num_workers=2
                    )
                    
                    selected_relative, _ = strategy.select(unlabeled_loader, query_size)
                    
                    # Map back to original indices
                    unlabeled_list = list(al_dataset.unlabeled_indices)
                    new_indices = [unlabeled_list[i] for i in selected_relative 
                                 if i < len(unlabeled_list)]
                
                al_dataset.label_indices(new_indices)
        
        return results
    
    def run_comparison(self):
        """Run comparison of all strategies."""
        all_results = {}
        
        for strategy in self.config['strategies']:
            print(f"\n{'#'*80}")
            print(f"Testing Strategy: {strategy.upper()}")
            print(f"{'#'*80}")
            
            strategy_results = []
            for run in range(self.config['num_runs']):
                result = self.run_strategy(strategy, run, seed=42 + run)
                strategy_results.append(result)
            
            all_results[strategy] = strategy_results
        
        # Calculate statistics
        stats = self._calculate_statistics(all_results)
        
        # Create visualizations
        self._create_plots(stats)
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def _calculate_statistics(self, all_results):
        """Calculate mean and std for each strategy across runs."""
        stats = {}
        
        for strategy, runs in all_results.items():
            n_iterations = len(runs[0]['test_accuracies'])
            
            # Stack results from all runs
            accuracies = np.array([run['test_accuracies'] for run in runs])
            losses = np.array([run['train_losses'] for run in runs])
            
            stats[strategy] = {
                'labeled_counts': runs[0]['labeled_counts'],
                'mean_accuracy': np.mean(accuracies, axis=0),
                'std_accuracy': np.std(accuracies, axis=0),
                'mean_loss': np.mean(losses, axis=0),
                'std_loss': np.std(losses, axis=0)
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
                'Training Loss vs Labeled Samples',
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
                    line=dict(color=colors[strategy], width=2),
                    marker=dict(size=8)
                ), row=1, col=1)
        
        # Plot 2: Relative Improvement
        random_acc = stats['random']['mean_accuracy']
        for strategy, data in stats.items():
            if strategy != 'random' and strategy in colors:
                improvement = ((data['mean_accuracy'] - random_acc) / random_acc) * 100
                fig.add_trace(go.Scatter(
                    x=data['labeled_counts'],
                    y=improvement,
                    mode='lines+markers',
                    name=strategy.replace('_', ' ').title(),
                    line=dict(color=colors[strategy], width=2),
                    marker=dict(size=8)
                ), row=1, col=2)
        
        # Plot 3: Training Loss
        for strategy, data in stats.items():
            if strategy in colors:
                fig.add_trace(go.Scatter(
                    x=data['labeled_counts'],
                    y=data['mean_loss'],
                    mode='lines',
                    name=strategy.replace('_', ' ').title(),
                    line=dict(color=colors[strategy], width=1, dash='dash'),
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
        fig.update_xaxes(title_text="Number of Labeled Samples", row=2, col=1)
        fig.update_xaxes(title_text="Strategy", row=2, col=2)
        
        fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
        fig.update_yaxes(title_text="Training Loss", row=2, col=1)
        fig.update_yaxes(title_text="Final Accuracy (%)", row=2, col=2)
        
        fig.update_layout(
            title_text='Active Learning Strategy Comparison (CIFAR-10)',
            height=800,
            width=1400,
            template='plotly_white',
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        # Save and open
        fig.write_html('active_learning_comparison.html')
        webbrowser.open('active_learning_comparison.html')
    
    def _print_summary(self, stats):
        """Print comprehensive summary of results."""
        print(f"\n{'='*80}")
        print("ACTIVE LEARNING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Get final results
        final_results = {}
        for strategy, data in stats.items():
            final_results[strategy] = {
                'accuracy': data['mean_accuracy'][-1],
                'std': data['std_accuracy'][-1]
            }
        
        # Sort by accuracy
        sorted_strategies = sorted(final_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\nFinal Accuracies (after all iterations):")
        print("-" * 50)
        for strategy, result in sorted_strategies:
            improvement = result['accuracy'] - final_results['random']['accuracy']
            print(f"{strategy.replace('_', ' ').title():20s}: "
                 f"{result['accuracy']:.2f}% ± {result['std']:.2f}% "
                 f"(Δ: {improvement:+.2f}%)")
            print("\nDetailed Performance Analysis:")
            print("-" * 50)
       
        # Calculate area under curve (AUC) for each strategy
        aucs = {}
        for strategy, data in stats.items():
            # Approximate AUC using trapezoidal rule
            x = np.array(data['labeled_counts'])
            y = np.array(data['mean_accuracy'])
            auc = np.trapz(y, x) / (x[-1] - x[0])
            aucs[strategy] = auc

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
        quarter_idx = len(stats['random']['labeled_counts']) // 4
        for strategy in sorted_strategies:
            strat_name = strategy[0]
            early_acc = stats[strat_name]['mean_accuracy'][quarter_idx]
            early_std = stats[strat_name]['std_accuracy'][quarter_idx]
            early_improvement = early_acc - stats['random']['mean_accuracy'][quarter_idx]
            print(f"{strat_name.replace('_', ' ').title():20s}: "
                    f"{early_acc:.2f}% ± {early_std:.2f}% "
                    f"(Δ: {early_improvement:+.2f}%)")

        # Best performing strategy
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
            print("⚠️ Active Learning shows minimal improvement. Consider:")
            print("   - Using a larger query budget")
            print("   - Implementing more sophisticated strategies")
            print("   - Checking if the model is already near optimal performance")


def main():
   """Main function to run the active learning pipeline."""
   
   # Configuration for experiments
   config = {
       'batch_size': 128,
       'initial_labeled': 1000,    # 2% of CIFAR-10 training data
       'query_size': 500,          # Query 1% at each iteration
       'max_iterations': 10,       # 10 iterations total
       'epochs_per_iteration': 20,
       'learning_rate': 0.01,
       'weight_decay': 5e-4,
       'momentum': 0.9,
       'num_runs': 3,              # 3 runs for statistical significance
       'strategies': [
           'random',
           'least_confidence',
           'margin',
           'entropy',
           'bald',
           'diverse_entropy'
       ]
   }
   
   print("="*80)
   print("ACTIVE LEARNING PIPELINE FOR CIFAR-10")
   print("="*80)
   print("\nConfiguration:")
   for key, value in config.items():
       print(f"  {key}: {value}")
   print()
   
   # Create and run pipeline
   pipeline = ActiveLearningPipeline(config)
   results = pipeline.run_comparison()
   
   print("\n" + "="*80)
   print("Experiment complete! Check 'active_learning_comparison.html' for visualizations.")
   print("="*80)
   
   return results


# Additional utility functions for advanced strategies

class CoreSetStrategy(QueryStrategy):
   """
   Core-set selection strategy that aims to select a representative subset
   of the data by maximizing the minimum distance to already selected points.
   """
   
   def __init__(self, model, device, n_classes=10):
       super().__init__(model, device, n_classes)
       self.selected_indices = []
   
   def update_distances(self, features, selected_idx, min_distances):
       """Update minimum distances after selecting a new point."""
       if len(features.shape) == 1:
           features = features.reshape(1, -1)
       
       new_distances = cdist([features[selected_idx]], features)[0]
       min_distances = np.minimum(min_distances, new_distances)
       return min_distances
   
   def select(self, unlabeled_loader, n_samples, labeled_features=None):
       self.model.eval()
       
       # Extract features
       features = []
       with torch.no_grad():
           for data, _ in unlabeled_loader:
               data = data.to(self.device)
               feat = self.model.get_embeddings(data)
               features.append(feat.cpu().numpy())
       
       features = np.vstack(features)
       n_unlabeled = features.shape[0]
       
       # Initialize distances
       if labeled_features is not None:
           # Calculate distances to labeled set
           min_distances = np.min(cdist(features, labeled_features), axis=1)
       else:
           # If no labeled features, start with infinite distances
           min_distances = np.full(n_unlabeled, np.inf)
           # Select first point randomly
           first_idx = np.random.randint(n_unlabeled)
           selected = [first_idx]
           min_distances = self.update_distances(features, first_idx, min_distances)
       
       selected = []
       
       # Greedy selection
       for _ in range(n_samples):
           if np.all(min_distances == 0):
               # All points are selected or distances are zero
               break
           
           # Select point with maximum minimum distance
           idx = np.argmax(min_distances)
           selected.append(idx)
           
           # Update distances
           min_distances = self.update_distances(features, idx, min_distances)
           min_distances[idx] = 0  # Mark as selected
       
       return selected, min_distances[selected]


class ExpectedGradientLengthStrategy(QueryStrategy):
   """
   Expected Gradient Length (EGL) strategy.
   Selects samples expected to produce the largest gradient magnitude.
   """
   
   def __init__(self, model, device, n_classes=10):
       super().__init__(model, device, n_classes)
   
   def select(self, unlabeled_loader, n_samples):
       self.model.eval()
       
       # We'll approximate EGL using the entropy-weighted prediction confidence
       egl_scores = []
       indices = []
       
       with torch.no_grad():
           for batch_idx, (data, _) in enumerate(unlabeled_loader):
               data = data.to(self.device)
               data.requires_grad = True
               
               outputs = self.model(data)
               probs = F.softmax(outputs, dim=1)
               
               # Compute pseudo-labels (most likely class)
               pseudo_labels = torch.argmax(probs, dim=1)
               
               # Compute loss with pseudo-labels
               criterion = nn.CrossEntropyLoss(reduction='none')
               losses = criterion(outputs, pseudo_labels)
               
               # Weight by entropy (uncertainty)
               log_probs = torch.log(probs + 1e-10)
               entropy = -(probs * log_probs).sum(dim=1)
               
               # EGL approximation: loss * entropy
               egl = losses * entropy
               egl_scores.extend(egl.cpu().numpy())
               
               batch_size = data.size(0)
               indices.extend(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
       
       egl_scores = np.array(egl_scores)
       selected_idx = np.argsort(egl_scores)[-n_samples:]
       
       return [indices[i] for i in selected_idx], egl_scores[selected_idx]


class BadgeStrategy(QueryStrategy):
   """
   BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy.
   Combines uncertainty and diversity using gradient embeddings.
   """
   
   def __init__(self, model, device, n_classes=10):
       super().__init__(model, device, n_classes)
   
   def get_grad_embeddings(self, unlabeled_loader):
       """Get gradient embeddings for all unlabeled samples."""
       self.model.eval()
       embeddings = []
       
       for data, _ in unlabeled_loader:
           data = data.to(self.device)
           data.requires_grad = True
           
           # Forward pass
           outputs = self.model(data)
           probs = F.softmax(outputs, dim=1)
           
           # Get predicted class
           pseudo_labels = torch.argmax(probs, dim=1)
           
           # Compute gradients for last layer
           self.model.zero_grad()
           
           # Get embeddings before final layer
           features = self.model.forward(data, return_features=True)
           
           # Compute hypothetical loss
           criterion = nn.CrossEntropyLoss()
           loss = criterion(outputs, pseudo_labels)
           
           # Get gradients with respect to features
           grad = torch.autograd.grad(loss, features, retain_graph=False)[0]
           
           # Weight by uncertainty (using max probability as inverse uncertainty)
           max_probs = torch.max(probs, dim=1)[0]
           uncertainty = 1 - max_probs
           
           # Create gradient embedding
           grad_embedding = grad * uncertainty.unsqueeze(1)
           embeddings.append(grad_embedding.cpu().detach().numpy())
       
       return np.vstack(embeddings)
   
   def select(self, unlabeled_loader, n_samples):
       # Get gradient embeddings
       grad_embeddings = self.get_grad_embeddings(unlabeled_loader)
       
       # Use k-means++ initialization for diverse selection
       from sklearn.cluster import kmeans_plusplus
       
       _, selected_idx = kmeans_plusplus(grad_embeddings, n_samples, random_state=42)
       
       return selected_idx.tolist(), np.ones(n_samples)  # Return uniform scores


# Enhanced pipeline with additional strategies
class EnhancedActiveLearningPipeline(ActiveLearningPipeline):
   """Extended pipeline with more sophisticated strategies."""
   
   def _setup_strategies(self):
       """Initialize extended set of query strategies."""
       super()._setup_strategies()
       
       # Add advanced strategies
       self.strategy_classes.update({
           'coreset': CoreSetStrategy,
           'egl': ExpectedGradientLengthStrategy,
           'badge': BadgeStrategy,
       })
   
   def run_advanced_comparison(self):
       """Run comparison including advanced strategies."""
       # Update config to include all strategies
       self.config['strategies'] = [
           'random',
           'least_confidence',
           'margin',
           'entropy',
           'bald',
           'diverse_entropy',
           'coreset',
           'egl',
           'badge'
       ]
       
       # Run standard comparison
       return self.run_comparison()


def run_advanced_experiments():
   """Run experiments with advanced active learning strategies."""
   
   config = {
       'batch_size': 128,
       'initial_labeled': 1000,
       'query_size': 500,
       'max_iterations': 10,
       'epochs_per_iteration': 20,
       'learning_rate': 0.01,
       'weight_decay': 5e-4,
       'momentum': 0.9,
       'num_runs': 3,
   }
   
   print("="*80)
   print("ADVANCED ACTIVE LEARNING EXPERIMENTS")
   print("="*80)
   
   pipeline = EnhancedActiveLearningPipeline(config)
   results = pipeline.run_advanced_comparison()
   
   return results


if __name__ == "__main__":
   # Run standard experiments
   results = main()

   print(torch.cuda.is_available())  # Should return True
   print(torch.version.cuda)  # Should show 12.6
   
   # Optionally run advanced experiments
   print("\n" + "="*80)
   user_input = input("Run advanced experiments with additional strategies? (y/n): ")
   if user_input.lower() == 'y':
       advanced_results = run_advanced_experiments()