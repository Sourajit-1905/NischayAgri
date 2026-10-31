"""Transform pipelines for training and evaluation, including the
with/without-color-augmentation and with/without-GAN-data ablation flags.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms

# ImageNet stats - required since all backbones (step 4) use ImageNet-
# pretrained timm weights. Mismatched normalization silently degrades
# transfer learning without raising any error, so this is not optional.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class AugmentationConfigError(ValueError):
    """The augmentation config asks for something not yet supported."""


@dataclass(frozen=True)
class AugmentationConfig:
    use_color_augmentation: bool = True
    use_gan_data: bool = False

    # Geometric augmentation - always on for training regardless of the
    # color-augmentation flag, since the ablation is specifically about color.
    random_crop_scale_min: float = 0.8
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 15.0

    # Color augmentation - only applied when use_color_augmentation is True.
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05


def build_train_transform(
    config: AugmentationConfig, input_size: int
) -> transforms.Compose:
    if config.use_gan_data:
        raise AugmentationConfigError(
            "use_gan_data=True was requested, but no GAN-based synthetic data "
            "pipeline exists in this project yet. This flag is a placeholder "
            "for a future GAN augmentation sub-task - building and validating "
            "a GAN is substantial work of its own, separate from this module. "
            "Set use_gan_data=False to proceed with standard augmentation only."
        )

    pipeline: list[object] = [
        transforms.RandomResizedCrop(
            input_size, scale=(config.random_crop_scale_min, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        transforms.RandomRotation(degrees=config.rotation_degrees),
    ]

    if config.use_color_augmentation:
        pipeline.append(
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            )
        )

    pipeline.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return transforms.Compose(pipeline)


def build_eval_transform(input_size: int) -> transforms.Compose:
    """Deterministic - no randomness, so val/test metrics are reproducible
    run-to-run. Used for val, test, and all target-domain evaluation sets
    (PlantDoc, FieldPlant, OOD, corruption suite)."""
    # Resize slightly larger than the crop, then center-crop, matching
    # standard ImageNet-pretrained-model evaluation convention.
    resize_size = int(input_size * 1.14)  # e.g. 224 -> 256, the usual ratio
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )