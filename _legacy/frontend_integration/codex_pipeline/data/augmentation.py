"""
Data augmentation pipeline for codex element images.

Tier 1: Geometric + photometric transforms (Albumentations v2.0+)
Tier 2: Copy-paste onto codex page backgrounds
Tier 3: Mixup / CutMix within the same class
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(cfg: dict, image_size: int = 224) -> A.Compose:
    """
    Build training augmentation pipeline from config.
    Uses Albumentations v2.0 API.
    """
    aug_cfg = cfg.get("augmentation", {}).get("tier1", {})

    scale_lo = aug_cfg.get("scale_range", [0.85, 1.15])[0]
    scale_hi = aug_cfg.get("scale_range", [0.85, 1.15])[1]

    transforms = [
        # Resize to model input size
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(255, 255, 255),
        ),

        # --- Tier 1: Geometric ---
        A.Affine(
            scale=(scale_lo, scale_hi),
            translate_percent=(-0.05, 0.05),
            rotate=(-aug_cfg.get("rotation_limit", 15), aug_cfg.get("rotation_limit", 15)),
            border_mode=0,
            fill=(255, 255, 255),
            p=0.7,
        ),
        A.ElasticTransform(
            alpha=aug_cfg.get("elastic_alpha", 30),
            sigma=aug_cfg.get("elastic_sigma", 5),
            border_mode=0,
            fill=(255, 255, 255),
            p=0.3,
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.HorizontalFlip(p=0.3),

        # --- Tier 1: Photometric ---
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_limit", 0.15),
                contrast_limit=aug_cfg.get("contrast_limit", 0.15),
                p=1.0,
            ),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ], p=0.5),

        # Simulated aging / noise
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),

        A.OneOf([
            A.GaussianBlur(
                blur_limit=aug_cfg.get("blur_limit", 3),
                p=1.0,
            ),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        # Color jitter for simulating different scanning conditions
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3,
        ),

        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_train_transform_numpy(cfg: dict, image_size: int = 224) -> A.Compose:
    """
    Training augmentation that returns numpy array (no ToTensor/Normalize).
    Used for augmented precomputation where we handle normalization separately.
    """
    aug_cfg = cfg.get("augmentation", {}).get("tier1", {})

    scale_lo = aug_cfg.get("scale_range", [0.85, 1.15])[0]
    scale_hi = aug_cfg.get("scale_range", [0.85, 1.15])[1]

    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(255, 255, 255),
        ),
        A.Affine(
            scale=(scale_lo, scale_hi),
            translate_percent=(-0.05, 0.05),
            rotate=(-aug_cfg.get("rotation_limit", 15), aug_cfg.get("rotation_limit", 15)),
            border_mode=0,
            fill=(255, 255, 255),
            p=0.7,
        ),
        A.ElasticTransform(
            alpha=aug_cfg.get("elastic_alpha", 30),
            sigma=aug_cfg.get("elastic_sigma", 5),
            border_mode=0,
            fill=(255, 255, 255),
            p=0.3,
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.HorizontalFlip(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_limit", 0.15),
                contrast_limit=aug_cfg.get("contrast_limit", 0.15),
                p=1.0,
            ),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=aug_cfg.get("blur_limit", 3), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3,
        ),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    """Minimal transforms for validation / inference."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(255, 255, 255),
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def mixup(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Blend two images of the same class.
    Both images should be the same shape (post-augmentation).
    """
    lam = np.random.beta(alpha, alpha)
    return (lam * img1 + (1 - lam) * img2).astype(img1.dtype)


def cutmix(img1: np.ndarray, img2: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    CutMix: paste a random rectangle from img2 onto img1.
    Both images should be the same shape.
    """
    h, w = img1.shape[:2]
    lam = np.random.beta(alpha, alpha)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    result = img1.copy()
    result[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    return result
