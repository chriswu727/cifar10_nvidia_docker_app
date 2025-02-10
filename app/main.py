from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SimpleCNN
import os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model save directory if it doesn't exist
os.makedirs('/app/models', exist_ok=True)

# Load model with full path
model_path = '/app/models/cifar10_model.pth'
model = SimpleCNN()
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and transform image
    image = Image.open(file.file)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)  # Move input to GPU
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    return {"prediction": classes[predicted.item()]} 