import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should show 12.6

def run_original_pipeline():
    # Hyperparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    initial_labeled = 5000
    query_size = 2500
    iterations = 5
    epochs_per_iter = 10

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Train function
    def train_model(model, loader, optimizer, criterion):
        model.train()
        for _ in range(epochs_per_iter):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    # Evaluate
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

    # Query
    def query_least_confidence(model, unlabeled_loader):
        model.eval()
        confidences = []
        indices = []
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(unlabeled_loader):
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                start_idx = batch_idx * unlabeled_loader.batch_size
                batch_indices = np.arange(start_idx, start_idx + len(images))
                confidences.extend(max_probs.cpu().numpy())
                indices.extend(batch_indices)
        sorted_indices = np.argsort(confidences)[:query_size]
        return [indices[i] for i in sorted_indices]

    # Main loop
    all_indices = list(range(len(trainset)))
    np.random.shuffle(all_indices)
    labeled_indices = all_indices[:initial_labeled]
    unlabeled_indices = all_indices[initial_labeled:]

    unlabeled_set = Subset(trainset, unlabeled_indices)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for iter in range(iterations + 1):
        print(f"Iteration {iter}: Labeled samples = {len(labeled_indices)}")
        labeled_set = Subset(trainset, labeled_indices)
        labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
        
        train_model(model, labeled_loader, optimizer, criterion)
        
        acc = evaluate(model, testloader)
        print(f"Test Accuracy: {acc:.2f}%")
        
        if iter == iterations:
            break
        
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)
        queried_indices = query_least_confidence(model, unlabeled_loader)
        
        labeled_indices.extend(queried_indices)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in queried_indices]
        unlabeled_set = Subset(trainset, unlabeled_indices)

    print("Original Pipeline Done!")

if __name__ == "__main__":
    run_original_pipeline()  # For standalone testing