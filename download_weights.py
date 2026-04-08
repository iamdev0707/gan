import torch
from torchvision.models import efficientnet_b0

print("Starting download...")

model = efficientnet_b0(weights="IMAGENET1K_V1")

print("Model loaded")

torch.save(model.state_dict(), "efficientnet_b0.pth")

print("Saved successfully!")