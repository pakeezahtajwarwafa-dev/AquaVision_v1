import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:
    timm = None


def _build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """
    Builds a vision backbone, strips its classification head with nn.Identity(),
    and dynamically extracts feature_dim.
    """
    name_clean = name.lower().strip()

    # 1. MobileViT & timm-backed models
    if "mobilevit" in name_clean or (timm is not None and name_clean not in dir(models) and not any(name_clean.startswith(prefix) for prefix in ["resnet", "efficientnet", "vit", "swin"])):
        if timm is None:
            raise ImportError("`timm` package is required for MobileViT and non-torchvision backbones. Run `python -m pip install timm`.")
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        feature_dim = model.num_features
        return model, feature_dim

    # 2. ResNet family (resnet18, resnet34, resnet50, etc.)
    if name_clean.startswith("resnet"):
        weights_attr = f"ResNet{name_clean[6:]}_Weights"
        weights = getattr(models, weights_attr).DEFAULT if pretrained and hasattr(models, weights_attr) else None
        model = getattr(models, name_clean)(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    # 3. EfficientNet family (efficientnet_b0, efficientnet_b4, etc.)
    elif name_clean.startswith("efficientnet"):
        variant = name_clean.split("_")[-1].upper()
        weights_attr = f"EfficientNet_{variant}_Weights"
        weights = getattr(models, weights_attr).DEFAULT if pretrained and hasattr(models, weights_attr) else None
        model = getattr(models, name_clean)(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    # 4. ViT family (vit_b_16, etc.)
    elif name_clean.startswith("vit"):
        variant = name_clean[4:].upper()
        weights_attr = f"ViT_{variant}_Weights"
        weights = getattr(models, weights_attr).DEFAULT if pretrained and hasattr(models, weights_attr) else None
        model = getattr(models, name_clean)(weights=weights)
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, feature_dim

    # 5. Swin Transformer family (swin_t, swin_s, swin_b, etc.)
    elif name_clean.startswith("swin"):
        variant = name_clean[5:].upper()
        weights_attr = f"Swin_{variant}_Weights"
        weights = getattr(models, weights_attr).DEFAULT if pretrained and hasattr(models, weights_attr) else None
        model = getattr(models, name_clean)(weights=weights)
        feature_dim = model.head.in_features
        model.head = nn.Identity()
        return model, feature_dim

    # Fallback to timm if model name is unrecognized by torchvision
    elif timm is not None:
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        feature_dim = model.num_features
        return model, feature_dim

    else:
        raise ValueError(f"Unsupported backbone: '{name}'. Available: resnet18, efficientnet_b0/b4, vit_b_16, swin_t/s/b, mobilevit_s, etc.")


class AquaVisionMultimodalModel(nn.Module):
    """
    Multimodal Early-Fusion Network supporting arbitrary vision backbones 
    and tabular water quality features.
    """
    def __init__(self, num_classes: int, num_tabular_features: int = 0, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone, self.visual_feature_dim = _build_backbone(backbone_name, pretrained=pretrained)
        self.num_tabular_features = num_tabular_features

        fusion_dim = self.visual_feature_dim + num_tabular_features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image_tensor: torch.Tensor, tabular_tensor: torch.Tensor = None) -> torch.Tensor:
        visual_features = self.backbone(image_tensor)
        
        # Flatten feature map if backbone outputs [B, C, H, W] instead of 2D pooled vector [B, C]
        if visual_features.dim() > 2:
            visual_features = torch.flatten(visual_features, 1)

        if self.num_tabular_features > 0 and tabular_tensor is not None:
            fused_features = torch.cat([visual_features, tabular_tensor], dim=1)
        else:
            fused_features = visual_features

        return self.classifier(fused_features)
