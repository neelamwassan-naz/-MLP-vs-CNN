
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MNIST dataset
transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_data = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Training function
def train_model(model, name, epochs=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"{name} | Epoch {epoch + 1}/{epochs} "
            f"| Loss: {total_loss / len(train_loader):.4f}"
        )

    training_time = time.time() - start_time
    return model, training_time


# Testing function
def test_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100 * correct / total


# Train both models
print("\nTraining MLP...")
mlp_model, mlp_time = train_model(MLP(), "MLP")

print("\nTraining CNN...")
cnn_model, cnn_time = train_model(CNN(), "CNN")

# Calculate accuracy
mlp_accuracy = test_model(mlp_model)
cnn_accuracy = test_model(cnn_model)

# Display results
print("\n===== FINAL RESULTS =====")
print(f"MLP Accuracy: {mlp_accuracy:.2f}%")
print(f"CNN Accuracy: {cnn_accuracy:.2f}%")
print(f"MLP Training Time: {mlp_time:.2f} seconds")
print(f"CNN Training Time: {cnn_time:.2f} seconds")

# Plot accuracy comparison
models = ["MLP", "CNN"]
accuracies = [mlp_accuracy, cnn_accuracy]

plt.bar(models, accuracies)
plt.title("MLP vs CNN Accuracy Comparison")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)

for i, accuracy in enumerate(accuracies):
    plt.text(i, accuracy + 1, f"{accuracy:.2f}%", ha="center")

plt.tight_layout()
plt.show()
