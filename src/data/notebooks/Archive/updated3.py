import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import random
from sklearn.metrics import accuracy_score
import copy

def run_active_learning_experiment():
    # Set seeds for reproducibility
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_seed(42)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = {
        'batch_size': 128,
        'initial_labeled': 1000,  # Start smaller to see more dramatic differences
        'query_size': 200,
        'max_iterations': 8,
        'epochs_per_iteration': 15,
        'learning_rate': 0.001,
        'weight_decay': 1e-4
    }

    # Data preparation with proper normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=transform_test)
    
    # Create a validation set from training data for better evaluation
    train_size = len(trainset)
    val_size = 5000
    train_indices = list(range(train_size - val_size))
    val_indices = list(range(train_size - val_size, train_size))
    
    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)
    
    test_loader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

    # Improved CNN model with batch normalization
    class ImprovedCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(ImprovedCNN, self).__init__()
            
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x, return_features=False):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            
            x = self.global_pool(x)
            features = torch.flatten(x, 1)
            
            x = self.dropout(F.relu(self.fc1(features)))
            logits = self.fc2(x)
            
            if return_features:
                return logits, features
            return logits

    def train_model(model, train_loader, val_loader, epochs, device):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            scheduler.step()
            
            # Validate
            if epoch % 5 == 0 or epoch == epochs - 1:
                val_acc = evaluate_model(model, val_loader, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return train_loss / len(train_loader)

    def evaluate_model(model, data_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total

    # Multiple Active Learning Strategies
    def query_least_confidence(model, unlabeled_dataset, unlabeled_indices, query_size):
        model.eval()
        
        # Create dataloader for unlabeled data
        unlabeled_subset = Subset(unlabeled_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=config['batch_size'], 
                                    shuffle=False, num_workers=0)
        
        uncertainties = []
        
        with torch.no_grad():
            for data, _ in unlabeled_loader:
                data = data.to(device)
                outputs = model(data)
                probabilities = F.softmax(outputs, dim=1)
                
                # Least confidence: 1 - max(p_i)
                max_probs, _ = torch.max(probabilities, dim=1)
                uncertainty = 1.0 - max_probs
                uncertainties.extend(uncertainty.cpu().numpy())
        
        # Get indices of most uncertain samples
        uncertain_indices = np.argsort(uncertainties)[-query_size:]
        selected_indices = [unlabeled_indices[i] for i in uncertain_indices]
        
        print(f"Uncertainty range: {np.min(uncertainties):.3f} - {np.max(uncertainties):.3f}")
        return selected_indices

    def query_margin_sampling(model, unlabeled_dataset, unlabeled_indices, query_size):
        model.eval()
        
        unlabeled_subset = Subset(unlabeled_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=config['batch_size'], 
                                    shuffle=False, num_workers=0)
        
        margins = []
        
        with torch.no_grad():
            for data, _ in unlabeled_loader:
                data = data.to(device)
                outputs = model(data)
                probabilities = F.softmax(outputs, dim=1)
                
                # Margin sampling: difference between top two predictions
                sorted_probs, _ = torch.sort(probabilities, descending=True)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                margins.extend(margin.cpu().numpy())
        
        # Get indices with smallest margins (most uncertain)
        uncertain_indices = np.argsort(margins)[:query_size]
        selected_indices = [unlabeled_indices[i] for i in uncertain_indices]
        
        return selected_indices

    def query_entropy_sampling(model, unlabeled_dataset, unlabeled_indices, query_size):
        model.eval()
        
        unlabeled_subset = Subset(unlabeled_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=config['batch_size'], 
                                    shuffle=False, num_workers=0)
        
        entropies = []
        
        with torch.no_grad():
            for data, _ in unlabeled_loader:
                data = data.to(device)
                outputs = model(data)
                probabilities = F.softmax(outputs, dim=1)
                
                # Entropy: -sum(p * log(p))
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                entropies.extend(entropy.cpu().numpy())
        
        # Get indices with highest entropy
        uncertain_indices = np.argsort(entropies)[-query_size:]
        selected_indices = [unlabeled_indices[i] for i in uncertain_indices]
        
        return selected_indices

    def query_random(unlabeled_indices, query_size):
        return random.sample(unlabeled_indices, min(query_size, len(unlabeled_indices)))

    # Main experiment function
    def run_experiment(query_strategy, strategy_name):
        print(f"\n{'='*50}")
        print(f"Running {strategy_name}")
        print(f"{'='*50}")
        
        # Initialize indices
        all_train_indices = list(range(len(train_subset)))
        np.random.shuffle(all_train_indices)
        
        labeled_indices = all_train_indices[:config['initial_labeled']]
        unlabeled_indices = all_train_indices[config['initial_labeled']:]
        
        # Track results
        results = {
            'labeled_counts': [],
            'train_losses': [],
            'val_accuracies': [],
            'test_accuracies': []
        }
        
        # Initialize model
        model = ImprovedCNN().to(device)
        
        for iteration in range(config['max_iterations']):
            print(f"\nIteration {iteration + 1}/{config['max_iterations']}")
            print(f"Labeled samples: {len(labeled_indices)}")
            print(f"Unlabeled samples: {len(unlabeled_indices)}")
            
            # Create labeled dataset
            labeled_subset = Subset(train_subset, labeled_indices)
            labeled_loader = DataLoader(labeled_subset, batch_size=config['batch_size'], 
                                      shuffle=True, num_workers=0)
            
            # Train model
            train_loss = train_model(model, labeled_loader, val_loader, 
                                   config['epochs_per_iteration'], device)
            
            # Evaluate
            val_acc = evaluate_model(model, val_loader, device)
            test_acc = evaluate_model(model, test_loader, device)
            
            # Record results
            results['labeled_counts'].append(len(labeled_indices))
            results['train_losses'].append(train_loss)
            results['val_accuracies'].append(val_acc)
            results['test_accuracies'].append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.2f}%")
            print(f"Test Accuracy: {test_acc:.2f}%")
            
            # Query new samples (except in last iteration)
            if iteration < config['max_iterations'] - 1 and len(unlabeled_indices) > 0:
                if query_strategy == 'random':
                    new_indices = query_random(unlabeled_indices, config['query_size'])
                elif query_strategy == 'least_confidence':
                    new_indices = query_least_confidence(model, train_subset, 
                                                       unlabeled_indices, config['query_size'])
                elif query_strategy == 'margin':
                    new_indices = query_margin_sampling(model, train_subset, 
                                                      unlabeled_indices, config['query_size'])
                elif query_strategy == 'entropy':
                    new_indices = query_entropy_sampling(model, train_subset, 
                                                       unlabeled_indices, config['query_size'])
                
                # Update labeled/unlabeled sets
                labeled_indices.extend(new_indices)
                unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
        
        return results

    # Run experiments
    print("Starting Active Learning Comparison...")
    
    # Test multiple AL strategies
    strategies = {
        'random': 'Random Sampling',
        'least_confidence': 'Least Confidence',
        'margin': 'Margin Sampling', 
        'entropy': 'Entropy Sampling'
    }
    
    all_results = {}
    
    for strategy, name in strategies.items():
        set_seed(42)  # Reset seed for fair comparison
        all_results[strategy] = run_experiment(strategy, name)

    # Create comprehensive plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Test Accuracy vs Labeled Samples', 'Validation Accuracy vs Labeled Samples',
                       'Training Loss vs Labeled Samples', 'Accuracy Improvement over Random'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = {'random': '#1f77b4', 'least_confidence': '#ff7f0e', 
             'margin': '#2ca02c', 'entropy': '#d62728'}
    
    for strategy, name in strategies.items():
        results = all_results[strategy]
        color = colors[strategy]
        
        # Test accuracy
        fig.add_trace(go.Scatter(
            x=results['labeled_counts'], y=results['test_accuracies'],
            mode='lines+markers', name=f'{name}', 
            line=dict(color=color, width=2)
        ), row=1, col=1)
        
        # Validation accuracy  
        fig.add_trace(go.Scatter(
            x=results['labeled_counts'], y=results['val_accuracies'],
            mode='lines+markers', name=f'{name} (Val)', 
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False
        ), row=1, col=2)
        
        # Training loss
        fig.add_trace(go.Scatter(
            x=results['labeled_counts'], y=results['train_losses'],
            mode='lines+markers', name=f'{name} (Loss)', 
            line=dict(color=color, width=2),
            showlegend=False
        ), row=2, col=1)
        
        # Improvement over random
        if strategy != 'random':
            random_acc = np.array(all_results['random']['test_accuracies'])
            strategy_acc = np.array(results['test_accuracies'])
            improvement = strategy_acc - random_acc
            
            fig.add_trace(go.Scatter(
                x=results['labeled_counts'], y=improvement,
                mode='lines+markers', name=f'{name} vs Random', 
                line=dict(color=color, width=2),
                showlegend=False
            ), row=2, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Number of Labeled Samples")
    fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Validation Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Training Loss", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy Improvement (%)", row=2, col=2)
    
    fig.update_layout(
        title_text='Comprehensive Active Learning Comparison (CIFAR-10)',
        height=800, width=1200,
        template='plotly_white'
    )
    
    fig.write_html('comprehensive_al_comparison.html')
    webbrowser.open('comprehensive_al_comparison.html')
    
    # Print final results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for strategy, name in strategies.items():
        final_acc = all_results[strategy]['test_accuracies'][-1]
        print(f"{name:20}: {final_acc:.2f}%")
        
    print(f"\n{'='*60}")
    print("IMPROVEMENT OVER RANDOM SAMPLING:")
    print(f"{'='*60}")
    
    random_final = all_results['random']['test_accuracies'][-1]
    for strategy, name in strategies.items():
        if strategy != 'random':
            final_acc = all_results[strategy]['test_accuracies'][-1]
            improvement = final_acc - random_final
            print(f"{name:20}: {improvement:+.2f}%")

if __name__ == "__main__":
    run_active_learning_experiment()