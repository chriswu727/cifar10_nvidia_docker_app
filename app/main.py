from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models.database as models
from celery_worker import process_prediction
import base64

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Read image file
        contents = await file.read()
        
        # Convert image to base64 for queue
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Create initial database record
        prediction = models.Prediction(
            filename=file.filename,
            prediction="Processing",
            confidence=0.0
        )
        db.add(prediction)
        db.commit()
        
        # Send task to Celery
        process_prediction.delay(image_base64, file.filename)
        
        return {"status": "Processing", "message": "Prediction task queued"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(models.Prediction).all()
    return predictions 