import torch


def apply_air_masking(prediction: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    Set predicted class to background (0) for empty-air voxels.

    Empty-air voxels are defined as voxels where all four input channels
    are zero, matching normalize_nonzero_zscore preprocessing.

    Args:
        prediction: Integer class prediction tensor of shape
                    [D, H, W] or [B, D, H, W].
        image: Normalized 4-channel input tensor of shape
               [4, D, H, W] or [B, 4, D, H, W].

    Returns:
        A tensor with empty-air voxels set to class 0.
    """
    if image.ndim == 4:
        is_air = torch.all(image == 0, dim=0)
    elif image.ndim == 5:
        is_air = torch.all(image == 0, dim=1)
    else:
        raise ValueError(
            f"Expected image tensor with 4 or 5 dims, got shape {tuple(image.shape)}"
        )

    is_air = is_air.to(prediction.device)
    masked_pred = prediction.clone()
    masked_pred[is_air] = 0

    return masked_pred
