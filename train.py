import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

#check if gpu is usable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Additional optimizations you could add:
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow optimizations

#cnn model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#preprocess dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#load cifar10 dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
trainloader = DataLoader(
    trainset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster data transfer to GPU
)

#initialize
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create a list to store losses
losses = []

# Create model save directory if it doesn't exist
os.makedirs('/app/models', exist_ok=True)

#train model
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 100 == 99:    # print every 100 mini-batches
            avg_loss = running_loss / 100
            losses.append(avg_loss)  # Store the average loss
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
            running_loss = 0.0

print('Training finished!')

# Save the model with full path
model_path = '/app/models/cifar10_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load the model using the same path
model.load_state_dict(torch.load(model_path))  # Use the same path when loading

#test model
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Plotting
plt.figure(figsize=(10,5))
plt.plot(losses, '-b', label='Training Loss')
plt.xlabel('Every 100 iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.grid(True)
plt.savefig('/app/results/loss_plot.png')
plt.close()


