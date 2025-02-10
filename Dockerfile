#use base images of pytorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model.py to root directory
COPY app/model.py /app/model.py
COPY train.py /app/train.py

CMD ["python", "train.py"]