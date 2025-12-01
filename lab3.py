import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import itertools

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10), # random rotation
    transforms.RandomAffine(0, translate=(0.1, 0.1)), # random translations
    transforms.ToTensor(), # 0-1 torch tensor
    transforms.Normalize((0.5,), (0.5,)) # normalizing pixels (-1,1)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

train_subset, val_subset = random_split(train_dataset, [50000, 10000])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# MLP Architecture
class MLP(nn.Module):
    def __init__(self, hidden1=256, hidden2=128, dropout=0.3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, 10)
        )

    def forward(self, x):
        return self.model(x)


# Training and eval loop
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (output.argmax(1) == target).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)

# Model definition
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MLP(hidden1=256, hidden2=128, dropout=0.2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
