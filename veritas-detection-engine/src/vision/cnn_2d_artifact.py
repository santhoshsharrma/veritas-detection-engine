"""Spatial-artifact detector: pretrained backbone fine-tuned for real vs AI-generated.

Training happens in notebooks (Kaggle/Colab). Requires torch/torchvision
(see requirements-train.txt); the API itself uses the exported ONNX model.
"""


def build_model(num_classes: int = 2):
    import torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
