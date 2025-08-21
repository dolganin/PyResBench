import torch
import torch.nn as nn
from torchvision.models import resnet18

def make_resnet18(num_classes: int) -> nn.Module:
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
