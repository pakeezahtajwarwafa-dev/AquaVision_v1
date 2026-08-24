import torch
import torch.nn as nn
import timm


def build_model(
    model_name: str = "resnet34",
    num_classes: int = 8,
    pretrained: bool = True,
    drop_rate: float = 0.2
) -> nn.Module:
    """Factory function leveraging timm to create computer vision backbones."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    return model


if __name__ == "__main__":
    test_model = build_model("resnet34", num_classes=8, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = test_model(dummy_input)
    print(f"Model built successfully. Output batch shape: {out.shape}")
