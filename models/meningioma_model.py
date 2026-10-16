# ============================================================
# MENINGIOMA 3D U-NET
# ============================================================
#
# Purpose:
#   Binary 3D segmentation of meningioma GTV from T1c MRI.
#
# Input:
#   (B, 1, 64, 64, 64)
#
# Output:
#   (B, 2, 64, 64, 64)
#
# Output classes:
#   Class 0 = Background
#   Class 1 = Tumor
#
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolution blocks.

    Structure:

        Conv3D
          ↓
        InstanceNorm3D
          ↓
        ReLU
          ↓
        Conv3D
          ↓
        InstanceNorm3D
          ↓
        ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.block(x)


# ============================================================
# ENCODER BLOCK
# ============================================================

class Encoder3D(nn.Module):
    """
    Encoder block:

        DoubleConv
             ↓
        MaxPool

    Returns:
        features before pooling
        pooled features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.conv = DoubleConv3D(
            in_channels,
            out_channels
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

    def forward(
        self,
        x: torch.Tensor
    ):

        features = self.conv(x)

        pooled = self.pool(
            features
        )

        return features, pooled


# ============================================================
# DECODER BLOCK
# ============================================================

class Decoder3D(nn.Module):
    """
    Decoder block:

        Transposed Conv
             ↓
        Concatenate skip connection
             ↓
        DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ):

        x = self.up(x)

        # ----------------------------------------------------
        # Spatial-size protection
        # ----------------------------------------------------
        #
        # Normally the dimensions match exactly because our
        # patch size is divisible by powers of two.
        #
        # This protection handles any minor mismatch safely.
        # ----------------------------------------------------

        if x.shape[2:] != skip.shape[2:]:

            diff_d = (
                skip.shape[2] -
                x.shape[2]
            )

            diff_h = (
                skip.shape[3] -
                x.shape[3]
            )

            diff_w = (
                skip.shape[4] -
                x.shape[4]
            )

            x = nn.functional.pad(
                x,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                    diff_d // 2,
                    diff_d - diff_d // 2
                ]
            )

        x = torch.cat(
            [skip, x],
            dim=1
        )

        return self.conv(x)


# ============================================================
# MENINGIOMA 3D U-NET
# ============================================================

class MeningiomaUNet3D(nn.Module):
    """
    3D U-Net for binary meningioma segmentation.

    Input:
        1-channel T1c MRI

    Output:
        2 segmentation classes

    Architecture:

        Input
          │
          ▼
        Encoder 1
          │
          ▼
        Encoder 2
          │
          ▼
        Encoder 3
          │
          ▼
        Bottleneck
          │
          ▼
        Decoder 3
          │
          ▼
        Decoder 2
          │
          ▼
        Decoder 1
          │
          ▼
        Output
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 16
    ):
        super().__init__()

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.in_channels = in_channels

        self.num_classes = num_classes

        self.base_channels = base_channels

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder1 = Encoder3D(
            in_channels,
            base_channels
        )

        self.encoder2 = Encoder3D(
            base_channels,
            base_channels * 2
        )

        self.encoder3 = Encoder3D(
            base_channels * 2,
            base_channels * 4
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv3D(
            base_channels * 4,
            base_channels * 8
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder3 = Decoder3D(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4
        )

        self.decoder2 = Decoder3D(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2
        )

        self.decoder1 = Decoder3D(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels
        )

        # ----------------------------------------------------
        # Final segmentation layer
        # ----------------------------------------------------

        self.output = nn.Conv3d(
            in_channels=base_channels,
            out_channels=num_classes,
            kernel_size=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Encoder 1
        # ----------------------------------------------------

        skip1, x = self.encoder1(x)

        # ----------------------------------------------------
        # Encoder 2
        # ----------------------------------------------------

        skip2, x = self.encoder2(x)

        # ----------------------------------------------------
        # Encoder 3
        # ----------------------------------------------------

        skip3, x = self.encoder3(x)

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        x = self.bottleneck(x)

        # ----------------------------------------------------
        # Decoder 3
        # ----------------------------------------------------

        x = self.decoder3(
            x,
            skip3
        )

        # ----------------------------------------------------
        # Decoder 2
        # ----------------------------------------------------

        x = self.decoder2(
            x,
            skip2
        )

        # ----------------------------------------------------
        # Decoder 1
        # ----------------------------------------------------

        x = self.decoder1(
            x,
            skip1
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        x = self.output(x)

        return x


# ============================================================
# MODEL FACTORY
# ============================================================

def create_meningioma_model(
    base_channels: int = 16
) -> MeningiomaUNet3D:

    model = MeningiomaUNet3D(
        in_channels=1,
        num_classes=2,
        base_channels=base_channels
    )

    return model


# ============================================================
# SIMPLE SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("MENINGIOMA 3D U-NET MODEL TEST")
    print("=" * 100)

    model = create_meningioma_model()

    print("\nModel created successfully.")

    print(
        "\nInput channels:",
        model.in_channels
    )

    print(
        "Output classes:",
        model.num_classes
    )

    print(
        "Base channels:",
        model.base_channels
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # CPU forward test
    # --------------------------------------------------------

    model.eval()

    test_input = torch.randn(
        1,
        1,
        64,
        64,
        64
    )

    with torch.no_grad():

        output = model(
            test_input
        )

    print(
        "\nCPU input shape:",
        tuple(test_input.shape)
    )

    print(
        "CPU output shape:",
        tuple(output.shape)
    )

    expected_shape = (
        1,
        2,
        64,
        64,
        64
    )

    if tuple(output.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found: {tuple(output.shape)}"
        )

    print(
        "\nCPU forward pass: PASS"
    )

    # --------------------------------------------------------
    # NaN / Inf check
    # --------------------------------------------------------

    if torch.isnan(output).any():

        raise RuntimeError(
            "NaN detected in model output."
        )

    if torch.isinf(output).any():

        raise RuntimeError(
            "Inf detected in model output."
        )

    print(
        "CPU output finite check: PASS"
    )

    # --------------------------------------------------------
    # GPU test
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        # ----------------------------------------------------
        # Clear GPU cache
        # ----------------------------------------------------

        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Move model
        # ----------------------------------------------------

        model = model.to(
            device
        )

        # ----------------------------------------------------
        # Create GPU input
        # ----------------------------------------------------

        gpu_input = torch.randn(
            1,
            1,
            64,
            64,
            64,
            device=device
        )

        # ----------------------------------------------------
        # Reset peak memory counter
        # ----------------------------------------------------

        torch.cuda.reset_peak_memory_stats(
            device
        )

        # ----------------------------------------------------
        # GPU forward
        # ----------------------------------------------------

        with torch.no_grad():

            gpu_output = model(
                gpu_input
            )

        # ----------------------------------------------------
        # Synchronize GPU
        # ----------------------------------------------------

        torch.cuda.synchronize()

        # ----------------------------------------------------
        # Print output
        # ----------------------------------------------------

        print(
            "\nGPU input shape:",
            tuple(gpu_input.shape)
        )

        print(
            "GPU output shape:",
            tuple(gpu_output.shape)
        )

        print(
            "GPU output device:",
            gpu_output.device
        )

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        if tuple(gpu_output.shape) != expected_shape:

            raise RuntimeError(
                "Unexpected GPU output shape.\n"
                f"Expected: {expected_shape}\n"
                f"Found: {tuple(gpu_output.shape)}"
            )

        print(
            "\nGPU output shape check: PASS"
        )

        # ----------------------------------------------------
        # Device check
        # ----------------------------------------------------

        if gpu_output.device.type != "cuda":

            raise RuntimeError(
                "GPU output is not on CUDA."
            )

        print(
            "GPU output device check: PASS"
        )

        # ----------------------------------------------------
        # NaN / Inf
        # ----------------------------------------------------

        if torch.isnan(gpu_output).any():

            raise RuntimeError(
                "NaN detected in GPU model output."
            )

        if torch.isinf(gpu_output).any():

            raise RuntimeError(
                "Inf detected in GPU model output."
            )

        print(
            "GPU output finite check: PASS"
        )

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        allocated_mb = (
            torch.cuda.memory_allocated(device)
            / (1024 ** 2)
        )

        reserved_mb = (
            torch.cuda.memory_reserved(device)
            / (1024 ** 2)
        )

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device)
            / (1024 ** 2)
        )

        print(
            "\nGPU memory:"
        )

        print(
            "  Current allocated:",
            f"{allocated_mb:.2f} MB"
        )

        print(
            "  Current reserved:",
            f"{reserved_mb:.2f} MB"
        )

        print(
            "  Peak allocated:",
            f"{peak_allocated_mb:.2f} MB"
        )

        print(
            "  Peak reserved:",
            f"{peak_reserved_mb:.2f} MB"
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del gpu_input
        del gpu_output
        del model

        torch.cuda.empty_cache()

        print(
            "\nGPU cache cleared."
        )

        print(
            "\nGPU forward pass: PASS"
        )

    else:

        print(
            "\nCUDA is not available."
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("MENINGIOMA 3D U-NET MODEL TEST COMPLETE")
    print("=" * 100)