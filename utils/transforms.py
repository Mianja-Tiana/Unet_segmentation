import cv2
import functools
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def resize_input(x, size, interpolation, **kwargs):
    """Redimensionne une image."""
    return cv2.resize(x, size, interpolation=interpolation)


def process_mask(mask, **kwargs):
    """Convertit le masque en valeurs 0 et 1."""
    return (mask / 255).astype(np.int64)


def get_train_transforms(image_size=(572, 572), mask_size=(388, 388)):
    """
    Returns the transformations for training.
    
    Args:
        image_size (tuple): Input image size
        mask_size (tuple): Output mask size
    """
    resize_image = functools.partial(
        resize_input, size=image_size, interpolation=cv2.INTER_CUBIC
    )
    resize_mask = functools.partial(
        resize_input, size=mask_size, interpolation=cv2.INTER_NEAREST
    )
        
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5,
        ),
        A.Affine(
            scale=(0.8, 1.2),
            rotate=(-30, 30),
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST,
            fill=0,
            fill_mask=0,
            p=0.5
        ),
        A.Lambda(image=resize_image, mask=resize_mask),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.1, 0.25),
            hole_width_range=(0.1, 0.25),
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(blur_limit=3),
        ], p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225)),
        A.Lambda(mask=process_mask),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=(572, 572), mask_size=(388, 388)):
    """Returns the transformations for validation."""
    resize_image = functools.partial(
        resize_input, size=image_size, interpolation=cv2.INTER_CUBIC
    )
    resize_mask = functools.partial(
        resize_input, size=mask_size, interpolation=cv2.INTER_NEAREST
    )
        
    return A.Compose([
        A.Lambda(image=resize_image, mask=resize_mask),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225)),
        A.Lambda(mask=process_mask),
        ToTensorV2(),
    ])