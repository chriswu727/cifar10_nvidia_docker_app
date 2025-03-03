from celery import Celery
import torch
from model import get_model
from database import SessionLocal
from models.database import Prediction
import base64
from PIL import Image
import io
import torchvision.transforms as transforms

# Initialize Celery
celery_app = Celery('tasks', broker='amqp://guest:guest@rabbitmq:5672//')

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model()
model.load_state_dict(torch.load('/app/trained_models/cifar10_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@celery_app.task
def process_prediction(image_data, filename):
    try:
        # Decode and process image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        prediction_class = classes[predicted.item()]
        confidence_value = float(confidence.item())
        
        # Update database
        db = SessionLocal()
        try:
            prediction = db.query(Prediction).filter_by(filename=filename).first()
            if prediction:
                prediction.prediction = prediction_class
                prediction.confidence = confidence_value
                db.commit()
        finally:
            db.close()
            
        return {"prediction": prediction_class, "confidence": confidence_value}
        
    except Exception as e:
        print(f"Error in process_prediction: {e}")
        return None 