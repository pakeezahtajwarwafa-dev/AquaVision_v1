import torch
import torch.nn as nn
import torchvision.models as models


class TabularMLP(nn.Module):
    """Multi-Layer Perceptron for tabular water quality parameters."""
    def __init__(self, in_features: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AquaVisionMultimodalModel(nn.Module):
    """Fuses CNN image backbone representations with tabular water quality vectors."""
    def __init__(
        self,
        num_classes: int,
        num_tabular_features: int = 0,
        backbone_name: str = "resnet18",
        pretrained: bool = True
    ):
        super().__init__()
        self.num_tabular_features = num_tabular_features

        # 1. Vision Backbone Setup
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            img_feat_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            img_feat_dim = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model

        # 2. Optional Tabular Sub-network Setup
        if num_tabular_features > 0:
            self.tabular_mlp = TabularMLP(in_features=num_tabular_features, out_dim=32)
            fusion_dim = img_feat_dim + 32
        else:
            self.tabular_mlp = None
            fusion_dim = img_feat_dim

        # 3. Final Multi-Class Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        img_feats = self.backbone(image)

        if self.tabular_mlp is not None and tabular is not None:
            tab_feats = self.tabular_mlp(tabular)
            fused = torch.cat([img_feats, tab_feats], dim=1)
        else:
            fused = img_feats

        return self.classifier(fused)
