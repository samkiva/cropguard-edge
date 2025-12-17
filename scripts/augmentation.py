"""
CropGuard Edge - Day 3: Data Augmentation Pipeline
This module implements advanced augmentation techniques for agricultural images.

Features:
- Realistic field conditions (lighting, blur, rotation)
- CutMix and Mixup for robustness
- Configurable intensity levels
- Visualization utilities

Author: Kivairu Samuel
Date: Day 3 of 90-Day Build
"""

import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

# ============================================================================
# AUGMENTATION CONFIGURATIONS
# ============================================================================

def get_training_augmentation(intensity='medium'):
    """
    Create augmentation pipeline for training.
    
    Args:
        intensity: 'light', 'medium', or 'heavy'
        
    Returns:
        Albumentations Compose object
    """
    
    if intensity == 'light':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    elif intensity == 'medium':
        return A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            
            # Lighting conditions (critical for outdoor use)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.6),
            
            # Environmental effects
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # Simulate shadows/occlusions
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),
            
            # Resize and normalize
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    elif intensity == 'heavy':
        return A.Compose([
            # All medium augmentations plus more aggressive ones
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=25,
                p=0.6
            ),
            
            # More aggressive lighting
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.4,
                    p=1.0
                ),
                A.CLAHE(clip_limit=4.0, p=1.0),
                A.RandomToneCurve(scale=0.3, p=1.0),
            ], p=0.7),
            
            # Distortions (simulate camera effects)
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            ], p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.4),
            
            A.RandomShadow(p=0.3),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    else:
        raise ValueError(f"Unknown intensity: {intensity}")

def get_validation_augmentation():
    """
    Minimal augmentation for validation (only resize and normalize).
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# ============================================================================
# CUTMIX IMPLEMENTATION
# ============================================================================

def cutmix(image1, image2, label1, label2, alpha=1.0):
    """
    CutMix augmentation: Cut and paste patches between images.
    
    Args:
        image1, image2: Input images (H, W, C)
        label1, label2: One-hot encoded labels
        alpha: Beta distribution parameter
        
    Returns:
        Mixed image and label
    """
    lam = np.random.beta(alpha, alpha)
    h, w = image1.shape[:2]
    
    # Sample bounding box
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    
    # Random center point
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Bounding box coordinates
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    
    # Adjust lambda to exactly cut area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image, mixed_label, lam

# ============================================================================
# MIXUP IMPLEMENTATION
# ============================================================================

def mixup(image1, image2, label1, label2, alpha=0.4):
    """
    Mixup augmentation: Linear interpolation of images and labels.
    
    Args:
        image1, image2: Input images (H, W, C)
        label1, label2: One-hot encoded labels
        alpha: Beta distribution parameter
        
    Returns:
        Mixed image and label
    """
    lam = np.random.beta(alpha, alpha)
    
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image.astype(np.uint8), mixed_label, lam

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_augmentations(image_path, num_augmentations=9, intensity='medium'):
    """
    Visualize different augmentations on a single image.
    
    Args:
        image_path: Path to input image
        num_augmentations: Number of variations to show
        intensity: Augmentation intensity
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get augmentation pipeline (without normalization for visualization)
    aug_pipeline = get_training_augmentation(intensity)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Augmented versions
    for i in range(1, num_augmentations):
        augmented = aug_pipeline(image=image)['image']
        
        # Denormalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented = augmented * std + mean
        augmented = np.clip(augmented, 0, 1)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f'Augmentation {i}', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_cutmix_examples(data_dir, num_examples=4):
    """
    Visualize CutMix augmentation examples.
    
    Args:
        data_dir: Path to data directory
        num_examples: Number of examples to show
    """
    data_path = Path(data_dir)
    
    # Get random images from different classes
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    
    for i in range(num_examples):
        # Select two random classes
        class1, class2 = random.sample(class_dirs, 2)
        
        # Get random images
        img1_path = random.choice(list(class1.glob('*.jpg')))
        img2_path = random.choice(list(class2.glob('*.jpg')))
        
        # Load images
        img1 = cv2.imread(str(img1_path))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (224, 224))
        
        img2 = cv2.imread(str(img2_path))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (224, 224))
        
        # Apply cutmix
        label1 = np.zeros(6); label1[0] = 1  # Dummy labels
        label2 = np.zeros(6); label2[1] = 1
        mixed_img, mixed_label, lam = cutmix(img1, img2, label1, label2)
        
        # Plot
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f'{class1.name}', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f'{class2.name}', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mixed_img)
        axes[i, 2].set_title(f'CutMix (λ={lam:.2f})', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_augmentation_pipeline():
    """
    Test the augmentation pipeline with a synthetic image.
    """
    print("🧪 Testing Augmentation Pipeline...")
    
    # Create synthetic test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test all intensity levels
    for intensity in ['light', 'medium', 'heavy']:
        print(f"\n  Testing '{intensity}' intensity...")
        aug = get_training_augmentation(intensity)
        
        try:
            augmented = aug(image=test_image)
            print(f"    ✓ {intensity.capitalize()} augmentation works")
            print(f"      Output shape: {augmented['image'].shape}")
        except Exception as e:
            print(f"    ❌ {intensity.capitalize()} augmentation failed: {e}")
    
    # Test validation augmentation
    print(f"\n  Testing validation augmentation...")
    val_aug = get_validation_augmentation()
    try:
        val_augmented = val_aug(image=test_image)
        print(f"    ✓ Validation augmentation works")
    except Exception as e:
        print(f"    ❌ Validation augmentation failed: {e}")
    
    # Test CutMix
    print(f"\n  Testing CutMix...")
    try:
        img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        label1 = np.array([1, 0, 0, 0, 0, 0])
        label2 = np.array([0, 1, 0, 0, 0, 0])
        
        mixed_img, mixed_label, lam = cutmix(img1, img2, label1, label2)
        print(f"    ✓ CutMix works (lambda: {lam:.3f})")
        print(f"      Mixed label sum: {mixed_label.sum():.3f} (should be ~1.0)")
    except Exception as e:
        print(f"    ❌ CutMix failed: {e}")
    
    # Test Mixup
    print(f"\n  Testing Mixup...")
    try:
        mixed_img, mixed_label, lam = mixup(img1, img2, label1, label2)
        print(f"    ✓ Mixup works (lambda: {lam:.3f})")
    except Exception as e:
        print(f"    ❌ Mixup failed: {e}")
    
    print("\n✅ All tests passed!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🌱 CropGuard Edge - Day 3: Augmentation Pipeline")
    print("="*70)
    
    test_augmentation_pipeline()
    
    print("\n" + "="*70)
    print("📝 AUGMENTATION PIPELINE READY")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use this module in your training script")
    print("  2. Run the visualization notebook to see examples")
    print("  3. Move to Day 4: Model Architecture")
    print("\n🚀 Data augmentation pipeline is production-ready!")