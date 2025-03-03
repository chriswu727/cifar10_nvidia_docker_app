from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models.database as models
from celery_worker import process_prediction
import base64
import docker

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

# Create a router for system monitoring
monitor_router = APIRouter(prefix="/monitor", tags=["monitoring"])

@monitor_router.get("/containers")
def get_container_status():
    client = docker.from_env()
    containers_status = {}
    
    for container in client.containers.list(all=True):
        try:
            stats = container.stats(stream=False)
            containers_status[container.name] = {
                "status": container.status,
                "running": container.status == "running",
                "memory_usage": {
                    "used": stats["memory_stats"].get("usage", 0),
                    "limit": stats["memory_stats"].get("limit", 0)
                },
                "cpu_usage": stats["cpu_stats"].get("cpu_usage", {}).get("total_usage", 0)
            }
        except Exception as e:
            containers_status[container.name] = {
                "status": "error",
                "error": str(e)
            }
    
    return containers_status

@monitor_router.get("/gpu")
def get_gpu_status():
    client = docker.from_env()
    try:
        # More flexible container name matching
        workers = [c for c in client.containers.list() if 'celery_worker' in c.name]
        if not workers:
            return {"error": "No worker container found"}
            
        worker = workers[0]
        gpu_stats = worker.exec_run('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits')
        
        if gpu_stats.exit_code == 0:
            util, mem_used, mem_total = map(int, gpu_stats.output.decode().strip().split(','))
            return {
                "utilization": util,
                "memory": {
                    "used": mem_used,
                    "total": mem_total,
                    "unit": "MB"
                }
            }
    except Exception as e:
        return {"error": str(e)}

# Add to your main FastAPI app
app.include_router(monitor_router) 