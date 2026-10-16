"""
meningioma_multimodal_unet.py
=============================
Isolated 3D U-Net Architecture for 4-Channel Multi-Modal Meningioma Segmentation.

Input:
    (B, 4, D, H, W)  [t1n, t1c, t2w, t2f]

Output:
    (B, 4, D, H, W)  [Logits for 0: BG, 1: NCR, 2: ED, 3: ET]

Key Features:
    - 4-channel input encoder to process native multi-modal MRI simultaneously.
    - 4-class output logits for direct multiclass cross-entropy or Dice/Focal loss.
    - Instance Normalization for stable 3D medical batch processing (batch size 1-2).
    - Lightweight base channels (default: 16) ensuring safe GPU VRAM footprint on 6GB RTX 3050.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Two consecutive 3D conv blocks with InstanceNorm3d and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock3D(nn.Module):
    """Encoder block: DoubleConv3D followed by MaxPool3d."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock3D(nn.Module):
    """Decoder block: Trilinear upsampling, skip-connection concatenation, DoubleConv3D."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = DoubleConv3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad x if dimension slight mismatch due to odd spatial sizes
        diff_d = skip.shape[2] - x.shape[2]
        diff_h = skip.shape[3] - x.shape[3]
        diff_w = skip.shape[4] - x.shape[4]
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                    diff_d // 2,
                    diff_d - diff_d // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MeningiomaMultimodalUNet(nn.Module):
    """
    3D U-Net for 4-channel Multi-Modal Meningioma Segmentation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Encoder stages
        self.enc1 = EncoderBlock3D(in_channels, base_channels)  # 16
        self.enc2 = EncoderBlock3D(base_channels, base_channels * 2)  # 32
        self.enc3 = EncoderBlock3D(base_channels * 2, base_channels * 4)  # 64

        # Bottleneck
        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8)  # 128

        # Decoder stages
        self.dec3 = DecoderBlock3D(base_channels * 8, base_channels * 4, base_channels * 4)  # 64
        self.dec2 = DecoderBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)  # 32
        self.dec1 = DecoderBlock3D(base_channels * 2, base_channels, base_channels)  # 16

        # Final 1x1x1 segmentation projection
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoders
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoders
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Logits
        return self.final_conv(x)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_multimodal_unet(
    in_channels: int = 4,
    num_classes: int = 4,
    base_channels: int = 16,
) -> MeningiomaMultimodalUNet:
    """Factory function for creating a multimodal meningioma 3D U-Net."""
    return MeningiomaMultimodalUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Meningioma Multimodal 3D U-Net Self-Test")
    print("=" * 60)
    model = create_multimodal_unet()
    print(f"Total parameters: {model.count_parameters():,}")
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (1, 4, 64, 64, 64), "Output shape mismatch!"
    print("[OK] Forward pass self-test successful.")

