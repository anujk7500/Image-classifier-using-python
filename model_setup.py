import torch
import torchvision.models as models

# Load a pre-trained ResNet and adjust it for CIFAR-100 (100 output classes)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 100)
