# CIFAR-10 Image Classification with Nvidia Docker Containers

A deep learning application that trains and serves a CIFAR-10 image classifier using Docker containers with GPU support.

## Architecture & Features

The application consists of several containerized services:
1. **Model Training Service** (GPU-enabled)
   - Trains CNN model on CIFAR-10 dataset
   - Uses NVIDIA GPU acceleration

2. **FastAPI Backend**
   - Provides REST API endpoints
   - Queues prediction tasks
   - Handles image uploads
   - Monitors container health and GPU status using Docker SDK

3. **RabbitMQ Message Broker**
   - Queues prediction tasks
   - Manages task distribution
   - Provides monitoring UI

4. **Celery Worker** (GPU-enabled)
   - Processes prediction tasks
   - Runs model inference
   - Updates prediction results

5. **React Frontend**
   - User interface for image upload
   - Displays prediction results
   - Polls for prediction updates

6. **PostgreSQL Database**
   - Stores prediction history
   - Tracks task status
   - Records confidence scores

7. **Nginx Reverse Proxy**
   - Routes all traffic through port 80
   - Simplifies service access

8. **Flower Dashboard**
   - Monitors Celery tasks
   - Tracks worker status
   - Shows task success/failure rates
   - Provides real-time task metrics

## Access Points
- Frontend: http://localhost/
- API Documentation: http://localhost/docs
- RabbitMQ Monitor: http://localhost/rabbitmq/
- Flower Dashboard: http://localhost/flower/
- Prediction History: http://localhost/api/predictions

## Technologies Used

- PyTorch
- FastAPI
- React
- RabbitMQ
- Celery
- Flower
- PostgreSQL
- SQLAlchemy
- Docker
- Docker SDK for Python
- NVIDIA CUDA

## API Endpoints

- `POST /predict`: Upload an image for classification
- `GET /predictions`: Retrieve prediction history from database

## Task Processing Flow

1. User uploads image via frontend
2. FastAPI receives image and creates initial database record
3. Task is queued in RabbitMQ
4. Celery worker picks up task and runs prediction
5. Database is updated with results
6. Frontend polls and displays prediction

## Database Schema

The PostgreSQL database stores predictions with the following schema:
- `id`: Primary key
- `filename`: Name of the uploaded file
- `prediction`: Model's prediction result
- `confidence`: Prediction confidence score
- `timestamp`: When the prediction was made 