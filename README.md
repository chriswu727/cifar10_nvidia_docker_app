# CIFAR-10 Image Classification with Nvidia Docker Containers

A deep learning application that trains and serves a CIFAR-10 image classifier using Docker containers with GPU support.

## Features

- **Model Training**: Trains a CNN model on the CIFAR-10 dataset using NVIDIA GPU acceleration
- **FastAPI Backend**: Provides REST API endpoints for image classification
- **React Frontend**: User interface for uploading images and viewing predictions
- **PostgreSQL Database**: Stores prediction history with timestamps and confidence scores
- **Docker Containerization**: All components run in separate containers
- **GPU Support**: Utilizes NVIDIA GPU for both training and inference

## Architecture

The application consists of four main services:
1. **Model Training Service**: Trains the CNN model and saves it
2. **FastAPI Backend**: Loads the trained model and serves predictions
3. **React Frontend**: Provides user interface for interaction
4. **PostgreSQL Database**: Stores prediction history and metadata

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