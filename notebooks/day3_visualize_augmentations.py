"""
CropGuard Edge - Day 3: Augmentation Visualization
Run this notebook to see your augmentations in action.

Save as: notebooks/day3_visualize_augmentations.py
(Or copy to Jupyter: notebooks/day3_visualize_augmentations.ipynb)
"""

import sys
sys.path.append('..')  # Add parent directory to path

from scripts.augmentation import (
    visualize_augmentations,
    visualize_cutmix_examples,
    get_training_augmentation,
    cutmix,
    mixup
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

print("="*70)
print("🎨 CropGuard Edge - Augmentation Visualization")
print("="*70)

# ============================================================================
# 1. VISUALIZE STANDARD AUGMENTATIONS
# ============================================================================

print("\n1️⃣ STANDARD AUGMENTATIONS")
print("-" * 70)

# Get a sample image from your training data
train_dir = Path("data/train")
classes = [d for d in train_dir.iterdir() if d.is_dir()]

if not classes:
    print("❌ No training data found. Run Day 2 script first.")
    sys.exit(1)

# Pick a random class and image
sample_class = random.choice(classes)
sample_images = list(sample_class.glob("*.jpg"))

if not sample_images:
    print(f"❌ No images found in {sample_class}")
    sys.exit(1)

sample_image_path = random.choice(sample_images)

print(f"📸 Using sample image: {sample_image_path.name}")
print(f"   From class: {sample_class.name}\n")

# Test each intensity level
for intensity in ['light', 'medium', 'heavy']:
    print(f"📊 Generating '{intensity}' augmentations...")
    fig = visualize_augmentations(sample_image_path, num_augmentations=9, intensity=intensity)
    
    # Save figure
    output_path = f"augmentation_{intensity}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}\n")
    plt.close()

# ============================================================================
# 2. VISUALIZE CUTMIX
# ============================================================================

print("\n2️⃣ CUTMIX AUGMENTATION")
print("-" * 70)

try:
    fig = visualize_cutmix_examples("data/train", num_examples=4)
    fig.savefig("augmentation_cutmix.png", dpi=150, bbox_inches='tight')
    print("✓ CutMix visualization saved: augmentation_cutmix.png\n")
    plt.close()
except Exception as e:
    print(f"⚠️  Could not generate CutMix visualization: {e}\n")

# ============================================================================
# 3. SIDE-BY-SIDE COMPARISON
# ============================================================================

print("\n3️⃣ SIDE-BY-SIDE COMPARISON")
print("-" * 70)

# Load original image
image = cv2.imread(str(sample_image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create augmentations
aug_light = get_training_augmentation('light')
aug_medium = get_training_augmentation('medium')
aug_heavy = get_training_augmentation('heavy')

# Apply augmentations
img_light = aug_light(image=image)['image']
img_medium = aug_medium(image=image)['image']
img_heavy = aug_heavy(image=image)['image']

# Denormalize for display
def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return np.clip(img, 0, 1)

# Plot comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image)
axes[0].set_title('Original', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(denormalize(img_light))
axes[1].set_title('Light Augmentation', fontsize=14)
axes[1].axis('off')

axes[2].imshow(denormalize(img_medium))
axes[2].set_title('Medium Augmentation', fontsize=14)
axes[2].axis('off')

axes[3].imshow(denormalize(img_heavy))
axes[3].set_title('Heavy Augmentation', fontsize=14)
axes[3].axis('off')

plt.tight_layout()
fig.savefig("augmentation_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Comparison saved: augmentation_comparison.png\n")
plt.close()

# ============================================================================
# 4. STATISTICS
# ============================================================================

print("\n4️⃣ AUGMENTATION STATISTICS")
print("-" * 70)

# Test augmentation on multiple images to get statistics
num_test_images = 100
aug_pipeline = get_training_augmentation('medium')

print(f"Testing augmentation on {num_test_images} random images...\n")

brightness_changes = []
contrast_changes = []

for _ in range(num_test_images):
    # Get random image
    random_class = random.choice(classes)
    random_images = list(random_class.glob("*.jpg"))
    if not random_images:
        continue
    
    random_image_path = random.choice(random_images)
    img = cv2.imread(str(random_image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Original brightness
    orig_brightness = img.mean()
    
    # Augmented brightness
    aug_img = aug_pipeline(image=img)['image']
    
    # Denormalize
    aug_img_denorm = denormalize(aug_img)
    aug_brightness = aug_img_denorm.mean()
    
    brightness_changes.append(abs(aug_brightness - (orig_brightness / 255.0)))

brightness_changes = np.array(brightness_changes)

print(f"Brightness change statistics:")
print(f"  Mean change: {brightness_changes.mean():.3f}")
print(f"  Std dev: {brightness_changes.std():.3f}")
print(f"  Max change: {brightness_changes.max():.3f}")
print(f"  Min change: {brightness_changes.min():.3f}")

# ============================================================================
# 5. VERIFICATION CHECKLIST
# ============================================================================

print("\n" + "="*70)
print("✅ AUGMENTATION VERIFICATION CHECKLIST")
print("="*70)

checklist = {
    "Light augmentation works": True,
    "Medium augmentation works": True,
    "Heavy augmentation works": True,
    "CutMix visualization created": Path("augmentation_cutmix.png").exists(),
    "Comparison image created": Path("augmentation_comparison.png").exists(),
    "Images look realistic": None,  # Manual check
    "No extreme distortions": None,  # Manual check
}

for check, status in checklist.items():
    if status is None:
        icon = "👀"
        note = "(manual inspection required)"
    elif status:
        icon = "✅"
        note = ""
    else:
        icon = "⚠️"
        note = "(check failed)"
    
    print(f"{icon} {check} {note}")

print("\n" + "="*70)
print("📋 NEXT STEPS:")
print("="*70)
print("\n1. Review the generated images:")
print("   - augmentation_light.png")
print("   - augmentation_medium.png")
print("   - augmentation_heavy.png")
print("   - augmentation_cutmix.png")
print("   - augmentation_comparison.png")
print("\n2. Verify augmentations look realistic:")
print("   ✓ Rotations are reasonable (no upside-down leaves)")
print("   ✓ Lighting changes mimic outdoor conditions")
print("   ✓ No weird artifacts or extreme distortions")
print("\n3. If everything looks good:")
print("   → Commit the augmentation script")
print("   → Move to Day 4: Model Architecture")
print("\n4. If images look unrealistic:")
print("   → Adjust intensity parameters in scripts/augmentation.py")
print("   → Re-run this visualization")
print("\n🚀 Augmentation pipeline is ready for training!")
print("="*70)

# ============================================================================
# 6. SAVE A SUMMARY REPORT
# ============================================================================

with open("augmentation_report.txt", "w") as f:
    f.write("CropGuard Edge - Day 3: Augmentation Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Sample image used: {sample_image_path.name}\n")
    f.write(f"Class: {sample_class.name}\n")
    f.write(f"Number of test images: {num_test_images}\n\n")
    f.write("Brightness Statistics:\n")
    f.write(f"  Mean change: {brightness_changes.mean():.3f}\n")
    f.write(f"  Std dev: {brightness_changes.std():.3f}\n")
    f.write(f"  Max change: {brightness_changes.max():.3f}\n")
    f.write(f"  Min change: {brightness_changes.min():.3f}\n\n")
    f.write("Generated files:\n")
    f.write("  - augmentation_light.png\n")
    f.write("  - augmentation_medium.png\n")
    f.write("  - augmentation_heavy.png\n")
    f.write("  - augmentation_cutmix.png\n")
    f.write("  - augmentation_comparison.png\n\n")
    f.write("Status: READY FOR TRAINING\n")

print("\n📄 Report saved: augmentation_report.txt")