import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import get_model 

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Enhanced data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True,
                        num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False,
                       num_workers=4, pin_memory=True)

# Initialize model
model = get_model()
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# Create directories
os.makedirs('/app/models', exist_ok=True)
os.makedirs('/app/results', exist_ok=True)

# Lists to store metrics
losses = []
train_accuracies = []
test_accuracies = []

def test_accuracy():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
best_acc = 0
for epoch in range(25):  # 25 epochs
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 50 == 49:
            avg_loss = running_loss / 50
            losses.append(avg_loss)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
            running_loss = 0.0
    
    train_acc = 100 * correct / total
    test_acc = test_accuracy()
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), '/app/models/cifar10_model.pth')
    
    scheduler.step()

print('Training finished!')
print(f'Best Test Accuracy: {best_acc:.2f}%')

# Plot training curves
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(losses, '-b', label='Training Loss')
plt.xlabel('Every 50 iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_accuracies, '-b', label='Train Accuracy')
plt.plot(test_accuracies, '-r', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.grid(True)

plt.savefig('/app/results/training_curves.png')
plt.close()


