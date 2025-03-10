from torchvision.models import resnet18
import torch.nn as nn

def get_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model 