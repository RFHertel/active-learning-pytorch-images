# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import torchvision.transforms as transforms
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import webbrowser  # For opening HTML if not in notebook

# def run_updated_pipeline():
#     # Hyperparams (same)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batch_size = 128
#     initial_labeled = 5000
#     query_size = 500
#     iterations = 5
#     epochs_per_iter = 10

#     # Data (same)
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
#     testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#     # Model (same)
#     class SimpleCNN(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#             self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.fc1 = nn.Linear(64 * 8 * 8, 128)
#             self.fc2 = nn.Linear(128, 10)

#         def forward(self, x):
#             x = self.pool(F.relu(self.conv1(x)))
#             x = self.pool(F.relu(self.conv2(x)))
#             x = torch.flatten(x, 1)
#             x = F.relu(self.fc1(x))
#             x = self.fc2(x)
#             return x

#     # Updated train function
#     def train_model(model, loader, optimizer, criterion):
#         model.train()
#         total_loss = 0
#         num_batches = 0
#         for _ in range(epochs_per_iter):
#             for images, labels in loader:
#                 images, labels = images.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#                 num_batches += 1
#         avg_loss = total_loss / num_batches if num_batches > 0 else 0
#         return avg_loss

#     # Evaluate (same)
#     def evaluate(model, loader):
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for images, labels in loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         return 100 * correct / total

#     # Query (same)
#     def query_least_confidence(model, unlabeled_loader):
#         model.eval()
#         confidences = []
#         indices = []
#         with torch.no_grad():
#             for batch_idx, (images, _) in enumerate(unlabeled_loader):
#                 images = images.to(device)
#                 outputs = model(images)
#                 probs = F.softmax(outputs, dim=1)
#                 max_probs, _ = torch.max(probs, dim=1)
#                 start_idx = batch_idx * unlabeled_loader.batch_size
#                 batch_indices = np.arange(start_idx, start_idx + len(images))
#                 confidences.extend(max_probs.cpu().numpy())
#                 indices.extend(batch_indices)
#         sorted_indices = np.argsort(confidences)[:query_size]
#         return [indices[i] for i in sorted_indices]

#     # Main loop with tracking
#     all_indices = list(range(len(trainset)))
#     np.random.shuffle(all_indices)
#     labeled_indices = all_indices[:initial_labeled]
#     unlabeled_indices = all_indices[initial_labeled:]

#     unlabeled_set = Subset(trainset, unlabeled_indices)

#     model = SimpleCNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     avg_losses = []
#     test_errors = []
#     label_counts = []

#     for iter in range(iterations + 1):
#         print(f"Iteration {iter}: Labeled samples = {len(labeled_indices)}")
#         labeled_set = Subset(trainset, labeled_indices)
#         labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
        
#         avg_loss = train_model(model, labeled_loader, optimizer, criterion)
#         avg_losses.append(avg_loss)
#         label_counts.append(len(labeled_indices))
        
#         acc = evaluate(model, testloader)
#         error = 100 - acc
#         test_errors.append(error)
#         print(f"Test Accuracy: {acc:.2f}% (Error: {error:.2f}%)")
        
#         if iter == iterations:
#             break
        
#         unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)
#         queried_indices = query_least_confidence(model, unlabeled_loader)
        
#         labeled_indices.extend(queried_indices)
#         unlabeled_indices = [idx for idx in unlabeled_indices if idx not in queried_indices]
#         unlabeled_set = Subset(trainset, unlabeled_indices)

#     # Generate Plotly plots (save to HTML and open)
#     fig_loss = px.line(x=label_counts, y=avg_losses, labels={'x': 'Total Labeled Images', 'y': 'Average Training Loss'},
#                        title='Training Loss Reduction with Added Labels')
#     fig_loss.write_html('loss_plot.html')
#     webbrowser.open('loss_plot.html')

#     fig_error = px.line(x=label_counts, y=test_errors, labels={'x': 'Total Labeled Images', 'y': 'Test Error (%)'},
#                         title='Test Error Reduction with Added Labels')
#     fig_error.write_html('error_plot.html')
#     webbrowser.open('error_plot.html')

#     print("Updated Pipeline Done! Check the HTML plots.")

# if __name__ == "__main__":
#     run_updated_pipeline()  # For standalone testing

#######################################################################################################

# import torch  # Import PyTorch for tensor operations and neural networks
# import torch.nn as nn  # Import neural network modules from PyTorch
# import torch.optim as optim  # Import optimizers like Adam from PyTorch
# import torch.nn.functional as F  # Import functional operations like softmax and ReLU
# from torch.utils.data import DataLoader, Subset  # Import data loading utilities for batches and subsets
# import torchvision  # Import torchvision for datasets and models
# import torchvision.transforms as transforms  # Import image transformations like normalization
# import numpy as np  # Import NumPy for array operations
# import plotly.express as px  # Import Plotly Express for easy interactive plots
# import plotly.graph_objects as go  # Import Plotly Graph Objects for more customizable plots
# import webbrowser  # Import webbrowser to automatically open HTML plot files
# import random  # Import random for seeding and random sampling

# def run_pipeline(mode='both'):  # Main function to run the pipeline; mode can be 'active', 'random', or 'both' for comparison
#     # Set random seed for reproducibility across runs (ensures same initial splits and shuffles)
#     seed = 42
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True  # Make CUDA operations deterministic
#     torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

#     # Hyperparameters (adjusted for better demonstration: larger query_size and more iterations to show AL benefits)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect GPU if available, else use CPU
#     batch_size = 128  # Batch size for data loaders (balance between speed and memory)
#     initial_labeled = 5000  # Initial number of labeled samples (10% of CIFAR-10 train set)
#     query_size = 2500  # Number of samples to query/add per iteration (5% of train set)
#     iterations = 10  # Number of AL iterations (total labeled ~30k by end)
#     epochs_per_iter = 20  # Epochs to train per iteration (increased for better convergence)

#     # Data transformations: Convert to tensor and normalize (mean/std for CIFAR-10 channels)
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     # Load CIFAR-10 datasets: Train set for labeled/unlabeled pools, test set for evaluation
#     trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
#     testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)  # DataLoader for test set (no shuffle needed)

#     # Define a simple CNN model for classification (2 conv layers + 2 FC layers)
#     class SimpleCNN(nn.Module):
#         def __init__(self):
#             super().__init__()  # Call parent class constructor
#             self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Conv layer: 3 input channels (RGB), 32 outputs, 3x3 kernel
#             self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Conv layer: 32 inputs, 64 outputs, 3x3 kernel
#             self.pool = nn.MaxPool2d(2, 2)  # Max pooling: 2x2 kernel to reduce dimensions
#             self.fc1 = nn.Linear(64 * 8 * 8, 128)  # FC layer: Flattened input to 128 units
#             self.fc2 = nn.Linear(128, 10)  # Output layer: 128 to 10 classes (CIFAR-10)

#         def forward(self, x):  # Forward pass definition
#             x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
#             x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
#             x = torch.flatten(x, 1)  # Flatten tensor for FC layers (start from dim 1)
#             x = F.relu(self.fc1(x))  # FC1 + ReLU
#             x = self.fc2(x)  # FC2 (logits output)
#             return x

#     # Training function: Train model on labeled data and return average loss
#     def train_model(model, loader, optimizer, criterion, scheduler=None):
#         model.train()  # Set model to training mode (enables dropout/BatchNorm if used)
#         total_loss = 0  # Accumulator for total loss
#         num_batches = 0  # Counter for batches
#         for epoch in range(epochs_per_iter):  # Loop over epochs
#             for images, labels in loader:  # Loop over batches in loader
#                 images, labels = images.to(device), labels.to(device)  # Move data to device (GPU/CPU)
#                 optimizer.zero_grad()  # Reset gradients
#                 outputs = model(images)  # Forward pass
#                 loss = criterion(outputs, labels)  # Compute loss
#                 loss.backward()  # Backpropagate gradients
#                 optimizer.step()  # Update weights
#                 total_loss += loss.item()  # Accumulate loss value
#                 num_batches += 1  # Increment batch count
#             if scheduler:  # If scheduler provided, step it per epoch
#                 scheduler.step()  # Adjust learning rate
#         avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute average loss
#         return avg_loss

#     # Evaluation function: Compute accuracy on test set
#     def evaluate(model, loader):
#         model.eval()  # Set model to evaluation mode (disables dropout/BatchNorm updates)
#         correct, total = 0, 0  # Counters for correct predictions and total samples
#         with torch.no_grad():  # Disable gradient computation for efficiency
#             for images, labels in loader:  # Loop over test batches
#                 images, labels = images.to(device), labels.to(device)  # Move to device
#                 outputs = model(images)  # Forward pass
#                 _, predicted = torch.max(outputs, 1)  # Get predicted class (argmax)
#                 total += labels.size(0)  # Add batch size to total
#                 correct += (predicted == labels).sum().item()  # Count correct predictions
#         return 100 * correct / total  # Return accuracy percentage

#     # Active learning query: Least confidence (select samples with lowest max softmax probability)
#     def query_least_confidence(model, unlabeled_loader):
#         model.eval()  # Set to eval mode
#         confidences = []  # List to store max probabilities (confidences)
#         indices = []  # List to store corresponding indices
#         with torch.no_grad():  # No gradients needed
#             for batch_idx, (images, _) in enumerate(unlabeled_loader):  # Loop over unlabeled batches (ignore dummy labels)
#                 images = images.to(device)  # Move to device
#                 outputs = model(images)  # Forward pass
#                 probs = F.softmax(outputs, dim=1)  # Softmax to get probabilities
#                 max_probs, _ = torch.max(probs, dim=1)  # Get max prob per sample
#                 start_idx = batch_idx * unlabeled_loader.batch_size  # Calculate starting index for batch
#                 batch_indices = np.arange(start_idx, start_idx + len(images))  # Generate batch indices
#                 confidences.extend(max_probs.cpu().numpy())  # Append confidences (to CPU for NumPy)
#                 indices.extend(batch_indices)  # Append indices
#         sorted_indices = np.argsort(confidences)[:query_size]  # Sort by lowest confidence, take top query_size
#         return [indices[i] for i in sorted_indices]  # Return selected indices

#     # Random sampling query: Randomly select query_size indices from unlabeled pool
#     def query_random(unlabeled_indices):
#         return random.sample(unlabeled_indices, min(query_size, len(unlabeled_indices)))  # Randomly sample without replacement

#     # Core loop function: Runs AL or random, collects metrics
#     def run_loop(query_func, mode_name):
#         # Initialize indices: Shuffle all train indices for random initial split
#         all_indices = list(range(len(trainset)))
#         np.random.shuffle(all_indices)  # Shuffle for randomness
#         labeled_indices = all_indices[:initial_labeled]  # Initial labeled set
#         unlabeled_indices = all_indices[initial_labeled:]  # Initial unlabeled pool

#         # Create initial unlabeled subset
#         unlabeled_set = Subset(trainset, unlabeled_indices)

#         # Initialize model, loss, optimizer, and scheduler (for better training)
#         model = SimpleCNN().to(device)  # Create model and move to device
#         criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
#         optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # LR scheduler: Reduce by 0.1 every 5 epochs

#         # Lists to track metrics over iterations
#         avg_losses = []  # Average training losses
#         test_errors = []  # Test errors (100 - accuracy)
#         label_counts = []  # Total labeled counts

#         # Main iteration loop
#         for iter in range(iterations + 1):  # +1 to include initial evaluation
#             print(f"{mode_name} Iteration {iter}: Labeled samples = {len(labeled_indices)}")  # Log current iteration
#             # Create labeled subset and loader (shuffle for training)
#             labeled_set = Subset(trainset, labeled_indices)
#             labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)

#             # Train and get average loss
#             avg_loss = train_model(model, labeled_loader, optimizer, criterion, scheduler)
#             avg_losses.append(avg_loss)  # Append loss
#             label_counts.append(len(labeled_indices))  # Append current label count

#             # Evaluate on test set
#             acc = evaluate(model, testloader)  # Get accuracy
#             error = 100 - acc  # Compute error
#             test_errors.append(error)  # Append error
#             print(f"Test Accuracy: {acc:.2f}% (Error: {error:.2f}%)")  # Log results

#             if iter == iterations:  # Stop after last iteration
#                 break

#             # Create unlabeled loader (no shuffle needed for querying)
#             unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)

#             # Query new indices based on mode
#             if query_func == query_least_confidence:  # For active learning
#                 queried_indices = query_least_confidence(model, unlabeled_loader)
#             else:  # For random
#                 queried_indices = query_random(unlabeled_indices)

#             # Add queried to labeled, remove from unlabeled
#             labeled_indices.extend(queried_indices)
#             unlabeled_indices = [idx for idx in unlabeled_indices if idx not in queried_indices]  # Update unlabeled list
#             unlabeled_set = Subset(trainset, unlabeled_indices)  # Update unlabeled subset

#         return label_counts, avg_losses, test_errors  # Return metrics for plotting

#     # Run active learning if mode allows
#     if mode in ['active', 'both']:
#         al_label_counts, al_avg_losses, al_test_errors = run_loop(query_least_confidence, 'Active Learning')

#     # Run random sampling if mode allows
#     if mode in ['random', 'both']:
#         rand_label_counts, rand_avg_losses, rand_test_errors = run_loop(query_random, 'Random Sampling')

#     # Generate comparison plots (overlay AL and random on same figures)
#     if mode == 'both':
#         # Loss plot: Combined for comparison
#         fig_loss = go.Figure()  # Create empty figure
#         fig_loss.add_trace(go.Scatter(x=al_label_counts, y=al_avg_losses, mode='lines', name='Active Learning Loss'))  # Add AL trace
#         fig_loss.add_trace(go.Scatter(x=rand_label_counts, y=rand_avg_losses, mode='lines', name='Random Sampling Loss'))  # Add random trace
#         fig_loss.update_layout(title='Training Loss Reduction: Active vs Random',  # Update layout
#                                xaxis_title='Total Labeled Images',
#                                yaxis_title='Average Training Loss')
#         fig_loss.write_html('loss_plot.html')  # Save to HTML
#         webbrowser.open('loss_plot.html')  # Open in browser

#         # Error plot: Combined for comparison
#         fig_error = go.Figure()  # Create empty figure
#         fig_error.add_trace(go.Scatter(x=al_label_counts, y=al_test_errors, mode='lines', name='Active Learning Error'))  # Add AL trace
#         fig_error.add_trace(go.Scatter(x=rand_label_counts, y=rand_test_errors, mode='lines', name='Random Sampling Error'))  # Add random trace
#         fig_error.update_layout(title='Test Error Reduction: Active vs Random',  # Update layout
#                                 xaxis_title='Total Labeled Images',
#                                 yaxis_title='Test Error (%)')
#         fig_error.write_html('error_plot.html')  # Save to HTML
#         webbrowser.open('error_plot.html')  # Open in browser

#         print("Pipelines Done! Check the HTML plots for comparisons.")  # Final log
#     elif mode == 'active':
#         # Single plots for AL only (similar to original)
#         fig_loss = px.line(x=al_label_counts, y=al_avg_losses, title='Active Learning: Training Loss Reduction')
#         fig_loss.write_html('loss_plot.html')
#         webbrowser.open('loss_plot.html')
#         fig_error = px.line(x=al_label_counts, y=al_test_errors, title='Active Learning: Test Error Reduction')
#         fig_error.write_html('error_plot.html')
#         webbrowser.open('error_plot.html')
#     elif mode == 'random':
#         # Single plots for random only
#         fig_loss = px.line(x=rand_label_counts, y=rand_avg_losses, title='Random Sampling: Training Loss Reduction')
#         fig_loss.write_html('loss_plot.html')
#         webbrowser.open('loss_plot.html')
#         fig_error = px.line(x=rand_label_counts, y=rand_test_errors, title='Random Sampling: Test Error Reduction')
#         fig_error.write_html('error_plot.html')
#         webbrowser.open('error_plot.html')

# if __name__ == "__main__":
#     run_pipeline(mode='both')  # Run both by default for comparison; change to 'active' or 'random' if needed

# import torch  # Import PyTorch for tensor operations and neural networks
# import torch.nn as nn  # Import neural network modules from PyTorch
# import torch.optim as optim  # Import optimizers like Adam from PyTorch
# import torch.nn.functional as F  # Import functional operations like softmax and ReLU
# from torch.utils.data import DataLoader, Subset  # Import data loading utilities for batches and subsets
# import torchvision  # Import torchvision for datasets and models
# import torchvision.transforms as transforms  # Import image transformations like normalization
# import numpy as np  # Import NumPy for array operations
# import plotly.graph_objects as go  # Import Plotly Graph Objects for customizable plots
# import webbrowser  # Import webbrowser to automatically open HTML plot files
# import random  # Import random for seeding and random sampling

# def run_pipeline(mode='both'):  # Main function to run the pipeline; mode can be 'active', 'random', or 'both' for comparison
#     # Set random seed for reproducibility across runs (ensures same initial splits and shuffles)
#     seed = 42
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True  # Make CUDA operations deterministic
#     torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

#     # Hyperparameters (adjusted for better demonstration: larger query_size and more iterations to show AL benefits)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect GPU if available, else use CPU
#     batch_size = 128  # Batch size for data loaders (balance between speed and memory)
#     initial_labeled = 5000  # Initial number of labeled samples (10% of CIFAR-10 train set)
#     query_size = 500  # Number of samples to query/add per iteration (5% of train set)
#     iterations = 5  # Number of AL iterations (total labeled ~30k by end)
#     epochs_per_iter = 20  # Epochs to train per iteration (increased for better convergence)

#     # Data transformations: Convert to tensor and normalize (mean/std for CIFAR-10 channels)
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     # Load CIFAR-10 datasets: Train set for labeled/unlabeled pools, test set for evaluation
#     trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
#     testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)  # DataLoader for test set (no shuffle needed)

#     # Define a simple CNN model for classification (2 conv layers + 2 FC layers)
#     class SimpleCNN(nn.Module):
#         def __init__(self):
#             super().__init__()  # Call parent class constructor
#             self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Conv layer: 3 input channels (RGB), 32 outputs, 3x3 kernel
#             self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Conv layer: 32 inputs, 64 outputs, 3x3 kernel
#             self.pool = nn.MaxPool2d(2, 2)  # Max pooling: 2x2 kernel to reduce dimensions
#             self.fc1 = nn.Linear(64 * 8 * 8, 128)  # FC layer: Flattened input to 128 units
#             self.fc2 = nn.Linear(128, 10)  # Output layer: 128 to 10 classes (CIFAR-10)

#         def forward(self, x):  # Forward pass definition
#             x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
#             x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
#             x = torch.flatten(x, 1)  # Flatten tensor for FC layers (start from dim 1)
#             x = F.relu(self.fc1(x))  # FC1 + ReLU
#             x = self.fc2(x)  # FC2 (logits output)
#             return x

#     # Training function: Train model on labeled data and return average loss
#     def train_model(model, loader, optimizer, criterion, scheduler=None):
#         model.train()  # Set model to training mode (enables dropout/BatchNorm if used)
#         total_loss = 0  # Accumulator for total loss
#         num_batches = 0  # Counter for batches
#         for epoch in range(epochs_per_iter):  # Loop over epochs
#             for images, labels in loader:  # Loop over batches in loader
#                 images, labels = images.to(device), labels.to(device)  # Move data to device (GPU/CPU)
#                 optimizer.zero_grad()  # Reset gradients
#                 outputs = model(images)  # Forward pass
#                 loss = criterion(outputs, labels)  # Compute loss
#                 loss.backward()  # Backpropagate gradients
#                 optimizer.step()  # Update weights
#                 total_loss += loss.item()  # Accumulate loss value
#                 num_batches += 1  # Increment batch count
#             if scheduler:  # If scheduler provided, step it per epoch
#                 scheduler.step()  # Adjust learning rate
#         avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute average loss
#         return avg_loss

#     # Evaluation function: Compute accuracy on test set
#     def evaluate(model, loader):
#         model.eval()  # Set model to evaluation mode (disables dropout/BatchNorm updates)
#         correct, total = 0, 0  # Counters for correct predictions and total samples
#         with torch.no_grad():  # Disable gradient computation for efficiency
#             for images, labels in loader:  # Loop over test batches
#                 images, labels = images.to(device), labels.to(device)  # Move to device
#                 outputs = model(images)  # Forward pass
#                 _, predicted = torch.max(outputs, 1)  # Get predicted class (argmax)
#                 total += labels.size(0)  # Add batch size to total
#                 correct += (predicted == labels).sum().item()  # Count correct predictions
#         return 100 * correct / total  # Return accuracy percentage

#     # Active learning query: Least confidence (select samples with lowest max softmax probability)
#     def query_least_confidence(model, unlabeled_loader):
#         model.eval()  # Set to eval mode
#         confidences = []  # List to store max probabilities (confidences)
#         indices = []  # List to store corresponding indices
#         with torch.no_grad():  # No gradients needed
#             for batch_idx, (images, _) in enumerate(unlabeled_loader):  # Loop over unlabeled batches (ignore dummy labels)
#                 images = images.to(device)  # Move to device
#                 outputs = model(images)  # Forward pass
#                 probs = F.softmax(outputs, dim=1)  # Softmax to get probabilities
#                 max_probs, _ = torch.max(probs, dim=1)  # Get max prob per sample
#                 start_idx = batch_idx * unlabeled_loader.batch_size  # Calculate starting index for batch
#                 batch_indices = np.arange(start_idx, start_idx + len(images))  # Generate batch indices
#                 confidences.extend(max_probs.cpu().numpy())  # Append confidences (to CPU for NumPy)
#                 indices.extend(batch_indices)  # Append indices
#         sorted_indices = np.argsort(confidences)[:query_size]  # Sort by lowest confidence, take top query_size
#         return [indices[i] for i in sorted_indices]  # Return selected indices

#     # Random sampling query: Randomly select query_size indices from unlabeled pool
#     def query_random(unlabeled_indices):
#         return random.sample(unlabeled_indices, min(query_size, len(unlabeled_indices)))  # Randomly sample without replacement

#     # Core loop function: Runs AL or random, collects metrics
#     def run_loop(query_func, mode_name):
#         # Initialize indices: Shuffle all train indices for random initial split
#         all_indices = list(range(len(trainset)))
#         np.random.shuffle(all_indices)  # Shuffle for randomness
#         labeled_indices = all_indices[:initial_labeled]  # Initial labeled set
#         unlabeled_indices = all_indices[initial_labeled:]  # Initial unlabeled pool

#         # Create initial unlabeled subset
#         unlabeled_set = Subset(trainset, unlabeled_indices)

#         # Initialize model, loss, optimizer, and scheduler (for better training)
#         model = SimpleCNN().to(device)  # Create model and move to device
#         criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
#         optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # LR scheduler: Reduce by 0.1 every 5 epochs

#         # Lists to track metrics over iterations
#         avg_losses = []  # Average training losses
#         test_errors = []  # Test errors (100 - accuracy)
#         label_counts = []  # Total labeled counts

#         # Main iteration loop
#         for iter in range(iterations + 1):  # +1 to include initial evaluation
#             print(f"{mode_name} Iteration {iter}: Labeled samples = {len(labeled_indices)}")  # Log current iteration
#             # Create labeled subset and loader (shuffle for training)
#             labeled_set = Subset(trainset, labeled_indices)
#             labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)

#             # Train and get average loss
#             avg_loss = train_model(model, labeled_loader, optimizer, criterion, scheduler)
#             avg_losses.append(avg_loss)  # Append loss
#             label_counts.append(len(labeled_indices))  # Append current label count

#             # Evaluate on test set
#             acc = evaluate(model, testloader)  # Get accuracy
#             error = 100 - acc  # Compute error
#             test_errors.append(error)  # Append error
#             print(f"Test Accuracy: {acc:.2f}% (Error: {error:.2f}%)")  # Log results

#             if iter == iterations:  # Stop after last iteration
#                 break

#             # Create unlabeled loader (no shuffle needed for querying)
#             unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)

#             # Query new indices based on mode
#             if query_func == query_least_confidence:  # For active learning
#                 queried_indices = query_least_confidence(model, unlabeled_loader)
#             else:  # For random
#                 queried_indices = query_random(unlabeled_indices)

#             # Add queried to labeled, remove from unlabeled
#             labeled_indices.extend(queried_indices)
#             unlabeled_indices = [idx for idx in unlabeled_indices if idx not in queried_indices]  # Update unlabeled list
#             unlabeled_set = Subset(trainset, unlabeled_indices)  # Update unlabeled subset

#         return label_counts, avg_losses, test_errors  # Return metrics for plotting

#     # Run active learning if mode allows
#     if mode in ['active', 'both']:
#         al_label_counts, al_avg_losses, al_test_errors = run_loop(query_least_confidence, 'Active Learning')

#     # Run random sampling if mode allows
#     if mode in ['random', 'both']:
#         rand_label_counts, rand_avg_losses, rand_test_errors = run_loop(query_random, 'Random Sampling')

#     # Generate combined plot (subplots for loss and error side-by-side)
#     if mode == 'both':
#         from plotly.subplots import make_subplots  # Import make_subplots for side-by-side plots
#         fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Loss Reduction', 'Test Error Reduction'))  # Create 1x2 subplot layout

#         # Add Active Learning traces
#         fig.add_trace(go.Scatter(x=al_label_counts, y=al_avg_losses, mode='lines', name='Active Learning Loss', line=dict(color='#1f77b4')), row=1, col=1)
#         fig.add_trace(go.Scatter(x=al_label_counts, y=al_test_errors, mode='lines', name='Active Learning Error', line=dict(color='#1f77b4')), row=1, col=2)

#         # Add Random Sampling traces
#         fig.add_trace(go.Scatter(x=rand_label_counts, y=rand_avg_losses, mode='lines', name='Random Sampling Loss', line=dict(color='#ff7f0e')), row=1, col=1)
#         fig.add_trace(go.Scatter(x=rand_label_counts, y=rand_test_errors, mode='lines', name='Random Sampling Error', line=dict(color='#ff7f0e')), row=1, col=2)

#         # Update layout for both subplots
#         fig.update_layout(title_text='Active Learning vs Random Sampling Comparison', height=600, width=1200,
#                           xaxis_title='Total Labeled Images', yaxis_title='Average Training Loss',
#                           xaxis2_title='Total Labeled Images', yaxis2_title='Test Error (%)')
#         fig.write_html('comparison_plot.html')  # Save to single HTML file
#         webbrowser.open('comparison_plot.html')  # Open in browser

#         print("Pipelines Done! Check the HTML plot for comparisons.")  # Final log

#     elif mode == 'active':
#         # Single plots for AL only (similar to original)
#         fig_loss = px.line(x=al_label_counts, y=al_avg_losses, title='Active Learning: Training Loss Reduction')
#         fig_loss.write_html('loss_plot.html')
#         webbrowser.open('loss_plot.html')
#         fig_error = px.line(x=al_label_counts, y=al_test_errors, title='Active Learning: Test Error Reduction')
#         fig_error.write_html('error_plot.html')
#         webbrowser.open('error_plot.html')
#     elif mode == 'random':
#         # Single plots for random only
#         fig_loss = px.line(x=rand_label_counts, y=rand_avg_losses, title='Random Sampling: Training Loss Reduction')
#         fig_loss.write_html('loss_plot.html')
#         webbrowser.open('loss_plot.html')
#         fig_error = px.line(x=rand_label_counts, y=rand_test_errors, title='Random Sampling: Test Error Reduction')
#         fig_error.write_html('error_plot.html')
#         webbrowser.open('error_plot.html')

# if __name__ == "__main__":
#     run_pipeline(mode='both')  # Run both by default for comparison; change to 'active' or 'random' if needed

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # Missing import
from plotly.subplots import make_subplots
import webbrowser
import random

def run_pipeline(mode='both'):
    # Set random seed ONCE for reproducible comparison
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    initial_labeled = 5000
    query_size = 500
    iterations = 5
    epochs_per_iter = 10  # Reduced for faster experimentation

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 proper stats
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    # Training function
    def train_model(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs_per_iter):
            epoch_loss = 0
            epoch_batches = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            
            total_loss += epoch_loss / epoch_batches if epoch_batches > 0 else 0
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0

    # Evaluation function
    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # Fixed active learning query function
    def query_least_confidence(model, unlabeled_indices, trainset, batch_size=128):
        model.eval()
        
        # Create subset and loader for unlabeled data
        unlabeled_subset = Subset(trainset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False)
        
        confidences = []
        
        with torch.no_grad():
            for images, _ in unlabeled_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                confidences.extend(max_probs.cpu().numpy())
        
        # Get indices of least confident samples
        least_confident_idx = np.argsort(confidences)[:query_size]
        
        # Map back to original dataset indices
        selected_indices = [unlabeled_indices[i] for i in least_confident_idx]
        
        return selected_indices

    # Random sampling function
    def query_random(unlabeled_indices):
        return random.sample(unlabeled_indices, min(query_size, len(unlabeled_indices)))

    # Create initial split (same for both methods)
    all_indices = list(range(len(trainset)))
    np.random.shuffle(all_indices)
    initial_labeled_indices = all_indices[:initial_labeled]
    initial_unlabeled_indices = all_indices[initial_labeled:]

    # Core loop function
    def run_single_experiment(query_strategy, strategy_name):
        print(f"\n=== Running {strategy_name} ===")
        
        # Initialize with the same starting conditions
        labeled_indices = initial_labeled_indices.copy()
        unlabeled_indices = initial_unlabeled_indices.copy()
        
        # Fresh model for each strategy
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Track metrics
        label_counts = []
        avg_losses = []
        test_accuracies = []
        test_errors = []

        # Main AL loop
        for iteration in range(iterations + 1):
            print(f"Iteration {iteration}: {len(labeled_indices)} labeled samples")
            
            # Create current labeled dataset
            labeled_subset = Subset(trainset, labeled_indices)
            labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
            
            # Train model
            if len(labeled_indices) > 0:  # Skip training if no labeled data
                avg_loss = train_model(model, labeled_loader, optimizer, criterion)
            else:
                avg_loss = 0
            
            # Evaluate
            accuracy = evaluate(model, testloader)
            error = 100 - accuracy
            
            # Record metrics
            label_counts.append(len(labeled_indices))
            avg_losses.append(avg_loss)
            test_accuracies.append(accuracy)
            test_errors.append(error)
            
            print(f"  Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            
            # Stop after final iteration
            if iteration == iterations:
                break
            
            # Query new samples
            if query_strategy == 'active':
                if len(unlabeled_indices) > 0:
                    new_indices = query_least_confidence(model, unlabeled_indices, trainset)
                else:
                    new_indices = []
            else:  # random
                if len(unlabeled_indices) > 0:
                    new_indices = query_random(unlabeled_indices)
                else:
                    new_indices = []
            
            # Update sets
            labeled_indices.extend(new_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
        
        return label_counts, avg_losses, test_errors, test_accuracies

    # Run experiments
    results = {}
    
    if mode in ['active', 'both']:
        results['active'] = run_single_experiment('active', 'Active Learning')
    
    if mode in ['random', 'both']:
        results['random'] = run_single_experiment('random', 'Random Sampling')

    # Plotting
    if mode == 'both':
        al_counts, al_losses, al_errors, al_accs = results['active']
        rand_counts, rand_losses, rand_errors, rand_accs = results['random']
        
        # Create comparison plots
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=('Test Accuracy vs Labeled Samples', 'Training Loss vs Labeled Samples')
        )
        
        # Accuracy comparison
        fig.add_trace(go.Scatter(
            x=al_counts, y=al_accs, mode='lines+markers', 
            name='Active Learning', line=dict(color='#1f77b4', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=rand_counts, y=rand_accs, mode='lines+markers',
            name='Random Sampling', line=dict(color='#ff7f0e', width=2)
        ), row=1, col=1)
        
        # Loss comparison
        fig.add_trace(go.Scatter(
            x=al_counts, y=al_losses, mode='lines+markers',
            name='Active Learning Loss', line=dict(color='#1f77b4', width=2),
            showlegend=False
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=rand_counts, y=rand_losses, mode='lines+markers',
            name='Random Sampling Loss', line=dict(color='#ff7f0e', width=2),
            showlegend=False
        ), row=1, col=2)
        
        # Update layout
        fig.update_xaxes(title_text="Number of Labeled Samples", row=1, col=1)
        fig.update_xaxes(title_text="Number of Labeled Samples", row=1, col=2)
        fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Training Loss", row=1, col=2)
        
        fig.update_layout(
            title_text='Active Learning vs Random Sampling Comparison (CIFAR-10)',
            height=500, width=1000,
            template='plotly_white'
        )
        
        fig.write_html('al_comparison.html')
        webbrowser.open('al_comparison.html')
        
        print(f"\n=== Final Results ===")
        print(f"Active Learning final accuracy: {al_accs[-1]:.2f}%")
        print(f"Random Sampling final accuracy: {rand_accs[-1]:.2f}%")
        print(f"Improvement: {al_accs[-1] - rand_accs[-1]:.2f}%")
        
    else:
        # Single strategy plots
        strategy = 'active' if mode == 'active' else 'random'
        counts, losses, errors, accs = results[strategy]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Test Accuracy', 'Training Loss'))
        
        fig.add_trace(go.Scatter(x=counts, y=accs, mode='lines+markers', name='Accuracy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=counts, y=losses, mode='lines+markers', name='Loss'), row=1, col=2)
        
        fig.update_layout(title_text=f'{strategy.title()} Learning Results')
        fig.write_html(f'{strategy}_results.html')
        webbrowser.open(f'{strategy}_results.html')

if __name__ == "__main__":
    run_pipeline(mode='both')