# CIFAR-10 Image Classification with Nvidia Docker Containers

A deep learning application that trains and serves a CIFAR-10 image classifier using Docker containers with GPU support.

## Architecture & Features

The application consists of several containerized services:
1. **Model Training Service** (GPU-enabled)
   - Trains CNN model on CIFAR-10 dataset
   - Uses NVIDIA GPU acceleration

2. **FastAPI Backend**
   - Provides REST API endpoints
   - Handles image classification

3. **React Frontend**
   - User interface for image upload
   - Displays prediction results

4. **PostgreSQL Database**
   - Stores prediction history
   - Tracks user activities

5. **Nginx Reverse Proxy**
   - Routes all traffic through port 80
   - Simplifies service access

## Access Points
- Frontend: http://localhost/
- API Documentation: http://localhost/docs
- Prediction History: http://localhost/api/predictions

## Technologies Used

- PyTorch
- FastAPI
- React
- PostgreSQL
- SQLAlchemy
- Docker
- NVIDIA CUDA

## API Endpoints

- `POST /predict`: Upload an image for classification
- `GET /predictions`: Retrieve prediction history from database

## Database Schema

The PostgreSQL database stores predictions with the following schema:
- `id`: Primary key
- `filename`: Name of the uploaded file
- `prediction`: Model's prediction result
- `confidence`: Prediction confidence score
- `timestamp`: When the prediction was made 