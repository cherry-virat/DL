# ============================================================
# MENINGIOMA SWIN UNETR 3D SEGMENTATION MODEL
# ============================================================
#
# Purpose:
#   Isolated Swin UNETR architecture for binary 3D segmentation
#   of meningioma GTV from T1c MRI.
#
# Input:
#   (B, 1, 64, 64, 64) - Single-channel contrast-enhanced T1c MRI
#
# Output:
#   (B, 2, 64, 64, 64) - 2-class logits
#     Class 0 = Background
#     Class 1 = Meningioma GTV
#
# Architecture:
#   MONAI SwinUNETR with memory-conservative feature_size=12
#   and gradient checkpointing enabled for 6GB VRAM targets.
# ============================================================

from pathlib import Path
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


# ============================================================
# WRAPPER / FACTORY
# ============================================================

class MeningiomaSwinUNETR(nn.Module):
    """
    Swin UNETR model wrapper for Meningioma GTV 3D segmentation.
    
    Args:
        in_channels: Number of input MRI channels (default: 1 for T1c).
        out_channels: Number of output segmentation classes (default: 2 for BG + GTV).
        feature_size: Base feature dimension (default: 12 for 6GB VRAM safety).
        use_checkpoint: Enable gradient checkpointing to save ~40-50% activation memory.
        spatial_dims: Number of spatial dimensions (3 for 3D MRI).
        drop_rate: Dropout rate for Swin Transformer blocks.
        attn_drop_rate: Attention dropout rate.
        dropout_path_rate: Drop path (stochastic depth) rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 12,
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims

        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=2,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=feature_size,
            norm_name="instance",
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=True,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_channels, D, H, W)
            
        Returns:
            Logits of shape (B, out_channels, D, H, W)
        """
        return self.swin_unetr(x)


def create_meningioma_swin_unetr(
    in_channels: int = 1,
    out_channels: int = 2,
    feature_size: int = 12,
    use_checkpoint: bool = True,
    spatial_dims: int = 3,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
) -> MeningiomaSwinUNETR:
    """
    Factory function to instantiate Meningioma Swin UNETR.
    """
    return MeningiomaSwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
    )


# ============================================================
# SELF-TEST SUITE
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MENINGIOMA SWIN UNETR MODEL SELF-TEST")
    print("=" * 80)

    model = create_meningioma_swin_unetr(
        in_channels=1,
        out_channels=2,
        feature_size=12,
        use_checkpoint=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture instantiated successfully.")
    print(f"Input channels:      {model.in_channels}")
    print(f"Output channels:     {model.out_channels}")
    print(f"Feature size:        {model.feature_size}")
    print(f"Use checkpointing:   {model.use_checkpoint}")
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable params:    {trainable_params:,}")

    # CPU Forward Test
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (1, 2, 64, 64, 64)
    assert tuple(output.shape) == expected_shape, f"CPU shape mismatch: {output.shape} vs {expected_shape}"
    assert not torch.isnan(output).any(), "NaN in CPU output"
    assert not torch.isinf(output).any(), "Inf in CPU output"
    print(f"\n[PASS] CPU Forward Pass -> Output shape: {tuple(output.shape)}")

    # GPU Test if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nTesting on GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = model.to(device)
        gpu_input = torch.randn(1, 1, 64, 64, 64, device=device)

        with torch.no_grad():
            gpu_output = model(gpu_input)
        torch.cuda.synchronize()

        assert tuple(gpu_output.shape) == expected_shape, f"GPU shape mismatch: {gpu_output.shape}"
        assert not torch.isnan(gpu_output).any(), "NaN in GPU output"
        assert not torch.isinf(gpu_output).any(), "Inf in GPU output"

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        print(f"[PASS] GPU Forward Pass -> Output shape: {tuple(gpu_output.shape)}")
        print(f"Peak VRAM allocated: {peak_vram_mb:.2f} MB")
        print(f"Peak VRAM reserved:  {reserved_vram_mb:.2f} MB")

        del gpu_input, gpu_output, model
        torch.cuda.empty_cache()
        print("[PASS] GPU cache cleared successfully.")
    else:
        print("\nCUDA not available; GPU test skipped.")

    print("\n" + "=" * 80)
    print("MENINGIOMA SWIN UNETR SELF-TEST PASSED COMPLETELY")
    print("=" * 80)

