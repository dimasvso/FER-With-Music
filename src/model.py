import torch
import torch.nn as nn
from torchvision import models
from config import *

def build_model(freeze_base=True):
    # Load pretrained EfficientNetB0
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze semua layer base
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Ganti classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, NUM_CLASSES)
    )

    return model


def unfreeze_top_layers(model, n_layers=20):
    # Unfreeze n layer terakhir di features
    children = list(model.features.children())
    for layer in children[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    print(f"Unfroze top {n_layers} layers")
    return model