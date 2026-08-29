import torch
import torch.nn as nn
import torchvision.models as models

class AquaVisionModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = 'efficientnet_b0', pretrained: bool = True):
        super(AquaVisionModel, self).__init__()
        
        # Initialize the chosen vision backbone
        if backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            
            # Replace the classifier head for our specific disease classes
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            
            # Replace the fully connected layer
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Strictly image tensor in, logits out. No tabular routing.
        return self.backbone(x)
