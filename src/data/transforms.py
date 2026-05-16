"""
MONAI transform pipelines for ACDC.

We always use dictionary-style transforms ("...d" suffix) so we can keep
patient / phase / group metadata alongside the image and label arrays.
"""
from __future__ import annotations


def build_train_transforms(
    spacing=(1.25, 1.25, 10.0),
    spatial_size=(224, 224, 16),
    noise_std: float = 0.05,
    rot_deg: float = 15.0,
):
    """Training-time augmentation pipeline."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRangePercentilesd, CropForegroundd, SpatialPadd,
        RandSpatialCropd, RandAffined, RandGaussianNoised, RandFlipd,
        EnsureTyped,
    )
    import math

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=1.0, upper=99.0,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandSpatialCropd(keys=["image", "label"], roi_size=spatial_size, random_size=False),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0, 0, math.radians(rot_deg)),
            scale_range=(0.1, 0.1, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys="image", prob=0.3, std=noise_std),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_val_transforms(
    spacing=(1.25, 1.25, 10.0),
    spatial_size=(224, 224, 16),
):
    """Validation pipeline: deterministic preprocessing only — no augmentation."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRangePercentilesd, CropForegroundd, SpatialPadd,
        CenterSpatialCropd, EnsureTyped,
    )

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=1.0, upper=99.0,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        EnsureTyped(keys=["image", "label"]),
    ])
