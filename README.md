# CIFAR-10 Image Classification with Nvidia Docker Containers

A deep learning application that trains and serves a CIFAR-10 image classifier using Docker containers with GPU support.

## Features

- **Model Training**: Trains a CNN model on the CIFAR-10 dataset using NVIDIA GPU acceleration
- **FastAPI Backend**: Provides REST API endpoints for image classification
- **React Frontend**: User interface for uploading images and viewing predictions
- **Docker Containerization**: All components run in separate containers
- **GPU Support**: Utilizes NVIDIA GPU for both training and inference

## Architecture

The application consists of three main services:
1. **Model Training Service**: Trains the CNN model and saves it
2. **FastAPI Backend**: Loads the trained model and serves predictions
3. **React Frontend**: Provides user interface for interaction

## Technologies Used

- PyTorch
- FastAPI
- React
- Docker
- NVIDIA CUDA 