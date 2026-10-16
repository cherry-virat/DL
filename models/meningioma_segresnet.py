"""
meningioma_segresnet.py
=======================
MONAI SegResNet Wrapper for 4-Channel Multi-Modal Meningioma Segmentation.

Architecture:
    Encoder-Decoder with ResNet-style residual blocks and trilinear upsampling.
    Specifically designed for 3D multi-modal brain tumor segmentation (BraTS).

Input:
    (B, 4, D, H, W)  [t1n, t1c, t2w, t2f]

Output:
    (B, 4, D, H, W)  [Logits for 0: BG, 1: NCR, 2: ED, 3: ET]

Key Features:
    - Residual connections across encoder stages to prevent gradient degradation.
    - Deep feature representations (4.7M parameters with init_filters=16).
    - Lightweight GPU memory footprint (~680 MB peak VRAM on RTX 3050 6GB with AMP).
"""

from typing import Tuple
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class MeningiomaSegResNet(nn.Module):
    """
    3D SegResNet for 4-channel multimodal meningioma segmentation.
    Wraps MONAI's SegResNet with standardized interface.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        init_filters: int = 16,
        dropout_prob: float = 0.2,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_filters = init_filters

        self.net = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=init_filters,
            dropout_prob=dropout_prob,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_segresnet(
    in_channels: int = 4,
    num_classes: int = 4,
    init_filters: int = 16,
    dropout_prob: float = 0.2,
) -> MeningiomaSegResNet:
    """Factory function for creating a multimodal meningioma SegResNet."""
    return MeningiomaSegResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        init_filters=init_filters,
        dropout_prob=dropout_prob,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Meningioma Multimodal SegResNet Self-Test")
    print("=" * 60)
    model = create_segresnet()
    print(f"Total parameters: {model.count_parameters():,}")
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (1, 4, 64, 64, 64), "Output shape mismatch!"
    print("[OK] Forward pass self-test successful.")

