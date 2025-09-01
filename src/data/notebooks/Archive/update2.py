import torch  # Import PyTorch for tensor operations and neural networks
import torch.nn as nn  # Import neural network modules from PyTorch
import torch.optim as optim  # Import optimizers like Adam from PyTorch
import torch.nn.functional as F  # Import functional operations like softmax and ReLU
from torch.utils.data import DataLoader, Subset  # Import data loading utilities for batches and subsets
import torchvision  # Import torchvision for datasets and models
import torchvision.transforms as transforms  # Import image transformations like normalization
import numpy as np  # Import NumPy for array operations
import plotly.express as px  # Import Plotly Express for easy interactive plots
import plotly.graph_objects as go  # Import Plotly Graph Objects for more customizable plots
import webbrowser  # Import webbrowser to automatically open HTML plot files
import random  # Import random for seeding and random sampling

def run_pipeline(mode='both'):  # Main function to run the pipeline; mode can be 'active', 'random', or 'both' for comparison
    # Set random seed for reproducibility across runs (ensures same initial splits and shuffles)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Make CUDA operations deterministic
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

    # Hyperparameters (adjusted for better demonstration: larger query_size and more iterations to show AL benefits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect GPU if available, else use CPU
    batch_size = 128  # Batch size for data loaders (balance between speed and memory)
    initial_labeled = 5000  # Initial number of labeled samples (10% of CIFAR-10 train set)
    query_size = 2500  # Number of samples to query/add per iteration (5% of train set)
    iterations = 10  # Number of AL iterations (total labeled ~30k by end)
    epochs_per_iter = 20  # Epochs to train per iteration (increased for better convergence)

    # Data transformations: Convert to tensor and normalize (mean/std for CIFAR-10 channels)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 datasets: Train set for labeled/unlabeled pools, test set for evaluation
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)  # DataLoader for test set (no shuffle needed)

    # Define a simple CNN model for classification (2 conv layers + 2 FC layers)
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()  # Call parent class constructor
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Conv layer: 3 input channels (RGB), 32 outputs, 3x3 kernel
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Conv layer: 32 inputs, 64 outputs, 3x3 kernel
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling: 2x2 kernel to reduce dimensions
            self.fc1 = nn.Linear(64 * 8 * 8, 128)  # FC layer: Flattened input to 128 units
            self.fc2 = nn.Linear(128, 10)  # Output layer: 128 to 10 classes (CIFAR-10)

        def forward(self, x):  # Forward pass definition
            x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
            x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
            x = torch.flatten(x, 1)  # Flatten tensor for FC layers (start from dim 1)
            x = F.relu(self.fc1(x))  # FC1 + ReLU
            x = self.fc2(x)  # FC2 (logits output)
            return x

    # Training function: Train model on labeled data and return average loss
    def train_model(model, loader, optimizer, criterion, scheduler=None):
        model.train()  # Set model to training mode (enables dropout/BatchNorm if used)
        total_loss = 0  # Accumulator for total loss
        num_batches = 0  # Counter for batches
        for epoch in range(epochs_per_iter):  # Loop over epochs
            for images, labels in loader:  # Loop over batches in loader
                images, labels = images.to(device), labels.to(device)  # Move data to device (GPU/CPU)
                optimizer.zero_grad()  # Reset gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagate gradients
                optimizer.step()  # Update weights
                total_loss += loss.item()  # Accumulate loss value
                num_batches += 1  # Increment batch count
            if scheduler:  # If scheduler provided, step it per epoch
                scheduler.step()  # Adjust learning rate
        avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute average loss
        return avg_loss

    # Evaluation function: Compute accuracy on test set
    def evaluate(model, loader):
        model.eval()  # Set model to evaluation mode (disables dropout/BatchNorm updates)
        correct, total = 0, 0  # Counters for correct predictions and total samples
        with torch.no_grad():  # Disable gradient computation for efficiency
            for images, labels in loader:  # Loop over test batches
                images, labels = images.to(device), labels.to(device)  # Move to device
                outputs = model(images)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get predicted class (argmax)
                total += labels.size(0)  # Add batch size to total
                correct += (predicted == labels).sum().item()  # Count correct predictions
        return 100 * correct / total  # Return accuracy percentage

    # Active learning query: Least confidence (select samples with lowest max softmax probability)
    def query_least_confidence(model, unlabeled_loader):
        model.eval()  # Set to eval mode
        confidences = []  # List to store max probabilities (confidences)
        indices = []  # List to store corresponding indices
        with torch.no_grad():  # No gradients needed
            for batch_idx, (images, _) in enumerate(unlabeled_loader):  # Loop over unlabeled batches (ignore dummy labels)
                images = images.to(device)  # Move to device
                outputs = model(images)  # Forward pass
                probs = F.softmax(outputs, dim=1)  # Softmax to get probabilities
                max_probs, _ = torch.max(probs, dim=1)  # Get max prob per sample
                start_idx = batch_idx * unlabeled_loader.batch_size  # Calculate starting index for batch
                batch_indices = np.arange(start_idx, start_idx + len(images))  # Generate batch indices
                confidences.extend(max_probs.cpu().numpy())  # Append confidences (to CPU for NumPy)
                indices.extend(batch_indices)  # Append indices
        sorted_indices = np.argsort(confidences)[:query_size]  # Sort by lowest confidence, take top query_size
        return [indices[i] for i in sorted_indices]  # Return selected indices

    # Random sampling query: Randomly select query_size indices from unlabeled pool
    def query_random(unlabeled_indices):
        return random.sample(unlabeled_indices, min(query_size, len(unlabeled_indices)))  # Randomly sample without replacement

    # Core loop function: Runs AL or random, collects metrics
    def run_loop(query_func, mode_name):
        # Initialize indices: Shuffle all train indices for random initial split
        all_indices = list(range(len(trainset)))
        np.random.shuffle(all_indices)  # Shuffle for randomness
        labeled_indices = all_indices[:initial_labeled]  # Initial labeled set
        unlabeled_indices = all_indices[initial_labeled:]  # Initial unlabeled pool

        # Create initial unlabeled subset
        unlabeled_set = Subset(trainset, unlabeled_indices)

        # Initialize model, loss, optimizer, and scheduler (for better training)
        model = SimpleCNN().to(device)  # Create model and move to device
        criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # LR scheduler: Reduce by 0.1 every 5 epochs

        # Lists to track metrics over iterations
        avg_losses = []  # Average training losses
        test_errors = []  # Test errors (100 - accuracy)
        label_counts = []  # Total labeled counts

        # Main iteration loop
        for iter in range(iterations + 1):  # +1 to include initial evaluation
            print(f"{mode_name} Iteration {iter}: Labeled samples = {len(labeled_indices)}")  # Log current iteration
            # Create labeled subset and loader (shuffle for training)
            labeled_set = Subset(trainset, labeled_indices)
            labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)

            # Train and get average loss
            avg_loss = train_model(model, labeled_loader, optimizer, criterion, scheduler)
            avg_losses.append(avg_loss)  # Append loss
            label_counts.append(len(labeled_indices))  # Append current label count

            # Evaluate on test set
            acc = evaluate(model, testloader)  # Get accuracy
            error = 100 - acc  # Compute error
            test_errors.append(error)  # Append error
            print(f"Test Accuracy: {acc:.2f}% (Error: {error:.2f}%)")  # Log results

            if iter == iterations:  # Stop after last iteration
                break

            # Create unlabeled loader (no shuffle needed for querying)
            unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)

            # Query new indices based on mode
            if query_func == query_least_confidence:  # For active learning
                queried_indices = query_least_confidence(model, unlabeled_loader)
            else:  # For random
                queried_indices = query_random(unlabeled_indices)

            # Add queried to labeled, remove from unlabeled
            labeled_indices.extend(queried_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in queried_indices]  # Update unlabeled list
            unlabeled_set = Subset(trainset, unlabeled_indices)  # Update unlabeled subset

        return label_counts, avg_losses, test_errors  # Return metrics for plotting

    # Run active learning if mode allows
    if mode in ['active', 'both']:
        al_label_counts, al_avg_losses, al_test_errors = run_loop(query_least_confidence, 'Active Learning')

    # Run random sampling if mode allows
    if mode in ['random', 'both']:
        rand_label_counts, rand_avg_losses, rand_test_errors = run_loop(query_random, 'Random Sampling')

    # Generate comparison plots (overlay AL and random on same figures)
    if mode == 'both':
        # Loss plot: Combined for comparison
        fig_loss = go.Figure()  # Create empty figure
        fig_loss.add_trace(go.Scatter(x=al_label_counts, y=al_avg_losses, mode='lines', name='Active Learning Loss'))  # Add AL trace
        fig_loss.add_trace(go.Scatter(x=rand_label_counts, y=rand_avg_losses, mode='lines', name='Random Sampling Loss'))  # Add random trace
        fig_loss.update_layout(title='Training Loss Reduction: Active vs Random',  # Update layout
                               xaxis_title='Total Labeled Images',
                               yaxis_title='Average Training Loss')
        fig_loss.write_html('loss_plot.html')  # Save to HTML
        webbrowser.open('loss_plot.html')  # Open in browser

        # Error plot: Combined for comparison
        fig_error = go.Figure()  # Create empty figure
        fig_error.add_trace(go.Scatter(x=al_label_counts, y=al_test_errors, mode='lines', name='Active Learning Error'))  # Add AL trace
        fig_error.add_trace(go.Scatter(x=rand_label_counts, y=rand_test_errors, mode='lines', name='Random Sampling Error'))  # Add random trace
        fig_error.update_layout(title='Test Error Reduction: Active vs Random',  # Update layout
                                xaxis_title='Total Labeled Images',
                                yaxis_title='Test Error (%)')
        fig_error.write_html('error_plot.html')  # Save to HTML
        webbrowser.open('error_plot.html')  # Open in browser

        print("Pipelines Done! Check the HTML plots for comparisons.")  # Final log
    elif mode == 'active':
        # Single plots for AL only (similar to original)
        fig_loss = px.line(x=al_label_counts, y=al_avg_losses, title='Active Learning: Training Loss Reduction')
        fig_loss.write_html('loss_plot.html')
        webbrowser.open('loss_plot.html')
        fig_error = px.line(x=al_label_counts, y=al_test_errors, title='Active Learning: Test Error Reduction')
        fig_error.write_html('error_plot.html')
        webbrowser.open('error_plot.html')
    elif mode == 'random':
        # Single plots for random only
        fig_loss = px.line(x=rand_label_counts, y=rand_avg_losses, title='Random Sampling: Training Loss Reduction')
        fig_loss.write_html('loss_plot.html')
        webbrowser.open('loss_plot.html')
        fig_error = px.line(x=rand_label_counts, y=rand_test_errors, title='Random Sampling: Test Error Reduction')
        fig_error.write_html('error_plot.html')
        webbrowser.open('error_plot.html')

if __name__ == "__main__":
    run_pipeline(mode='both')  # Run both by default for comparison; change to 'active' or 'random' if needed