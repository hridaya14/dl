import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


## Data augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

## Loading dataset
train_dataset = datasets.CIFAR10("./data", train=True, download=True,
                                 transform=train_transform)
test_dataset = datasets.CIFAR10("./data", train=False, download=True,
                                transform=test_transform)

train_subset, val_subset = random_split(train_dataset,
                                        [45000, 5000])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset, batch_size=64)
test_loader  = DataLoader(test_dataset, batch_size=64)

## CNN Architecture
class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


## Model definition
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(dropout=0.3).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=10,
                                      gamma=0.5)

## Training loop
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == y).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == y).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


## Training model
EPOCHS = 20

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader,
                                  criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader,
                                 criterion, device)

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
