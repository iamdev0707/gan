import torch
import torch.nn as nn
from torchvision import models

class GANClassifier(nn.Module):
    def __init__(self, num_classes=12, model_name="efficientnet_b0"):
        super(GANClassifier, self).__init__()
        
        # Load a pretrained model for Transfer Learning
        if model_name == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(pretrained=True)
            in_features = self.base_model.classifier[1].in_features
            # Replace the top classifier layer
            self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported model.")

    def forward(self, x):
        return self.base_model(x)

def train_classifier(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
