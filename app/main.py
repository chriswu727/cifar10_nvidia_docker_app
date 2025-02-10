from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
import os
from database import get_db
from models.database import Prediction
from sqlalchemy.orm import Session

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
os.makedirs('/app/trained_models', exist_ok=True)

# Load model with full path
model_path = '/app/trained_models/cifar10_model.pth'
model = get_model()
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Read and transform image
    image = Image.open(file.file)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)  # Move input to GPU
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Save prediction to database
        db_prediction = Prediction(
            filename=file.filename,
            prediction=classes[predicted.item()],
            confidence=confidence.item()
        )
        db.add(db_prediction)
        db.commit()
        
        return {
            "prediction": classes[predicted.item()],
            "confidence": confidence.item()
        }

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).all()
    return predictions 