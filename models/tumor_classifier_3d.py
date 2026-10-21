"""
Lightweight 3D MRI Residual Brain Tumor Classifier.
===================================================

Architecture for patient-level binary classification:
  - Class 0: Glioma
  - Class 1: Meningioma

Design Principles:
------------------
1. 3D Structural Feature Extraction: Operates directly on coregistered 4-channel
   structural MRI volumes (t1n, t1c, t2w, t2f).
2. Hardware Efficiency: Designed specifically for the NVIDIA GeForce RTX 3050 Laptop GPU
   (6 GB VRAM ceiling). Base channel width of 16 yields ~2.1M trainable parameters and
   allocates < 200 MB VRAM during mixed-precision forward-backward passes at batch size 2.
3. No Pretrained Weight Downloads: Completely self-contained architecture initialized
   using standard He/Kaiming normal initialization. Zero external network calls.
4. Robust Input Validation: Enforces 5-dimensional tensor structure (B, C, D, H, W) and
   channel count matching canonical modalities.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn


class ResNetBlock3D(nn.Module):
    """
    Standard 3D Residual Convolutional Block.
    Consists of two 3x3x3 3D convolutions with Batch Normalization and ReLU,
    with a 1x1x1 projection shortcut when channel count or spatial stride changes.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class TumorClassifier3D(nn.Module):
    """
    Lightweight 3D ResNet-style classifier for multimodal brain MRI tumor type identification.

    Default configuration accepts 4-channel volumes of target shape (4, 128, 128, 96)
    and predicts logits for 2 classes (Glioma vs. Meningioma).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        base_channels: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be at least 1, got {in_channels}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if base_channels < 4:
            raise ValueError(f"base_channels must be at least 4, got {base_channels}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        # Initial stem: 7x7x7 conv with stride 2 downsamples spatial dimensions
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        c = base_channels
        # Stage 1: maintains channel width c
        self.stage1 = nn.Sequential(
            ResNetBlock3D(c, c, stride=1),
            ResNetBlock3D(c, c, stride=1),
        )

        # Stage 2: downsamples to c * 2
        self.stage2 = nn.Sequential(
            ResNetBlock3D(c, c * 2, stride=2),
            ResNetBlock3D(c * 2, c * 2, stride=1),
        )

        # Stage 3: downsamples to c * 4
        self.stage3 = nn.Sequential(
            ResNetBlock3D(c * 2, c * 4, stride=2),
            ResNetBlock3D(c * 4, c * 4, stride=1),
        )

        # Stage 4: downsamples to c * 8
        self.stage4 = nn.Sequential(
            ResNetBlock3D(c * 4, c * 8, stride=2),
            ResNetBlock3D(c * 8, c * 8, stride=1),
        )

        # Classification head: GAP -> Flatten -> Dropout -> Linear
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(c * 8, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for conv layers and constant for batchnorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw class logits.

        Parameters:
            x: 5D FloatTensor of shape (B, C, D, H, W).

        Returns:
            logits: 2D FloatTensor of shape (B, num_classes).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")

        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input tensor with shape (B, C, D, H, W), got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        b, c, d, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Input channel count mismatch: expected {self.in_channels} channels, "
                f"got {c} in tensor of shape {tuple(x.shape)}"
            )

        if any(s <= 0 for s in (b, c, d, h, w)):
            raise ValueError(f"All tensor dimensions must be strictly positive, got {tuple(x.shape)}")

        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        pooled = self.gap(feat)
        flat = torch.flatten(pooled, start_dim=1)
        dropped = self.dropout(flat)
        logits = self.classifier(dropped)
        return logits

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tumor_classifier_3d(
    in_channels: int = 4,
    num_classes: int = 2,
    base_channels: int = 16,
    dropout_rate: float = 0.2,
) -> TumorClassifier3D:
    """
    Factory helper to instantiate a TumorClassifier3D model.

    Parameters:
        in_channels: Number of MRI modality channels (default 4 for t1n, t1c, t2w, t2f).
        num_classes: Number of target tumor classes (default 2: Glioma vs. Meningioma).
        base_channels: Initial channel width in the stem (default 16 for RTX 3050 6GB).
        dropout_rate: Dropout probability preceding the linear classification head.

    Returns:
        Instantiated TumorClassifier3D nn.Module.
    """
    return TumorClassifier3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
    )


create_classifier = create_tumor_classifier_3d
