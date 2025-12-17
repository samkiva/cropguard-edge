"""
CropGuard Edge - Day 2: Auto-Detecting Data Cleaning Script
This version automatically finds the correct dataset path.

Author: Kivairu Samuel
Date: Day 2
"""

import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import random

# Configuration
BASE_DIR = Path(".")
POSSIBLE_RAW_PATHS = [
    "plantvillage dataset",
    "plantvillage dataset/color",
    "plantvillage dataset/raw/color",
    "data/raw/plantvillage",
    "data/raw/color",
    "data/raw",
]

OUTPUT_DIR = "data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Priority classes for CropGuard v1
PRIORITY_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Potato___Early_blight",
    "Potato___Late_blight",
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def find_dataset_path():
    """Auto-detect the correct dataset path"""
    print("🔍 Searching for dataset...")
    
    for path in POSSIBLE_RAW_PATHS:
        full_path = BASE_DIR / path
        if full_path.exists():
            # Check if it contains any of our target classes
            subdirs = [d.name for d in full_path.iterdir() if d.is_dir()]
            
            # Look for any Tomato or Potato directories
            matching = [d for d in subdirs if 'Tomato' in d or 'Potato' in d]
            
            if matching:
                print(f"✓ Found dataset at: {full_path}")
                print(f"  Sample classes found: {matching[:3]}")
                return full_path
    
    # If not found, list what we have
    print("\n❌ Could not auto-detect dataset path.")
    print("\nSearched in:")
    for path in POSSIBLE_RAW_PATHS:
        print(f"  - {path}")
    
    print("\n📁 Current directory structure:")
    for item in BASE_DIR.iterdir():
        if item.is_dir():
            print(f"  - {item.name}/")
            # Show subdirs
            try:
                for subitem in list(item.iterdir())[:3]:
                    if subitem.is_dir():
                        print(f"    - {subitem.name}/")
            except:
                pass
    
    return None

def create_directory_structure():
    """Create train/val/test directories for each class"""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for class_name in PRIORITY_CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    print("✓ Directory structure created")

def is_valid_image(image_path):
    """Check if image is valid"""
    try:
        img = Image.open(image_path)
        img.verify()
        
        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            return False, "OpenCV failed to load"
        
        height, width = img_cv.shape[:2]
        if height < 50 or width < 50:
            return False, f"Image too small: {width}x{height}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def clean_and_organize_dataset(raw_data_path):
    """Main cleaning function"""
    
    stats = {
        "total_found": 0,
        "valid": 0,
        "corrupted": 0,
        "too_small": 0,
        "by_class": {}
    }
    
    print("\n🔍 Scanning dataset...")
    
    for class_name in PRIORITY_CLASSES:
        print(f"\n📂 Processing: {class_name}")
        
        # Try to find the class directory with flexible matching
        class_dirs = list(raw_data_path.glob(f"*{class_name}*"))
        
        # Try alternative patterns
        if not class_dirs:
            # Try with spaces instead of underscores
            alt_name = class_name.replace("___", " ").replace("_", " ")
            class_dirs = [d for d in raw_data_path.iterdir() 
                         if d.is_dir() and alt_name.lower() in d.name.lower()]
        
        if not class_dirs:
            print(f"  ⚠️  Class directory not found")
            print(f"     Available directories:")
            for d in list(raw_data_path.iterdir())[:5]:
                if d.is_dir():
                    print(f"     - {d.name}")
            continue
        
        class_dir = class_dirs[0]
        print(f"  Found: {class_dir.name}")
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(class_dir.glob(ext))
        
        # Also check subdirectories
        for subdir in class_dir.iterdir():
            if subdir.is_dir():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(subdir.glob(ext))
        
        stats["total_found"] += len(image_files)
        
        if len(image_files) == 0:
            print(f"  ⚠️  No images found in {class_dir}")
            continue
        
        # Validate images
        valid_images = []
        print(f"  Validating {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="  Validating"):
            is_valid, error = is_valid_image(img_path)
            
            if is_valid:
                valid_images.append(img_path)
                stats["valid"] += 1
            else:
                if "too small" in str(error).lower():
                    stats["too_small"] += 1
                else:
                    stats["corrupted"] += 1
        
        print(f"  ✓ Valid images: {len(valid_images)}")
        
        if len(valid_images) < 100:
            print(f"  ⚠️  WARNING: Only {len(valid_images)} valid images")
            print(f"     Recommended minimum: 100 per class")
        
        # Split into train/val/test
        random.shuffle(valid_images)
        
        n_train = int(len(valid_images) * TRAIN_RATIO)
        n_val = int(len(valid_images) * VAL_RATIO)
        
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train + n_val]
        test_images = valid_images[n_train + n_val:]
        
        stats["by_class"][class_name] = {
            "total": len(image_files),
            "valid": len(valid_images),
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        }
        
        # Copy to organized structure
        print(f"  Copying to train/val/test...")
        
        for split_name, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images)
        ]:
            split_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            
            for i, img_path in enumerate(tqdm(split_images, desc=f"  {split_name}")):
                new_filename = f"{class_name}_{split_name}_{i:04d}{img_path.suffix}"
                dest_path = os.path.join(split_dir, new_filename)
                
                try:
                    shutil.copy2(img_path, dest_path)
                except Exception as e:
                    print(f"    ⚠️  Failed to copy {img_path.name}: {e}")
    
    return stats

def print_statistics(stats):
    """Print dataset statistics"""
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    
    print(f"\n🔍 Total images scanned: {stats['total_found']}")
    print(f"✅ Valid images: {stats['valid']}")
    print(f"❌ Corrupted images: {stats['corrupted']}")
    print(f"⚠️  Too small images: {stats['too_small']}")
    
    if not stats["by_class"]:
        print("\n❌ No classes were processed successfully!")
        print("   Please check the dataset path and class names.")
        return
    
    print("\n📦 CLASS DISTRIBUTION:")
    print("-" * 70)
    print(f"{'Class':<35} {'Train':>8} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-" * 70)
    
    for class_name, counts in stats["by_class"].items():
        print(f"{class_name:<35} {counts['train']:>8} {counts['val']:>6} "
              f"{counts['test']:>6} {counts['valid']:>7}")
    
    print("-" * 70)
    total_train = sum(c['train'] for c in stats['by_class'].values())
    total_val = sum(c['val'] for c in stats['by_class'].values())
    total_test = sum(c['test'] for c in stats['by_class'].values())
    total_all = sum(c['valid'] for c in stats['by_class'].values())
    
    print(f"{'TOTAL':<35} {total_train:>8} {total_val:>6} {total_test:>6} {total_all:>7}")
    print("="*70)
    
    # Class balance check
    if stats['by_class']:
        counts = [c['valid'] for c in stats['by_class'].values()]
        min_count = min(counts)
        max_count = max(counts)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print("\n⚖️  CLASS BALANCE CHECK:")
        if ratio > 3:
            print(f"⚠️  WARNING: Significant class imbalance detected!")
            print(f"   Largest class: {max_count} images")
            print(f"   Smallest class: {min_count} images")
            print(f"   Ratio: {ratio:.1f}x")
            print(f"   → Use class weights during training")
        else:
            print(f"✓ Classes are balanced (ratio: {ratio:.2f}x)")

def save_class_labels():
    """Save class labels for mobile app"""
    labels_path = os.path.join(OUTPUT_DIR, "labels.txt")
    
    with open(labels_path, 'w') as f:
        for class_name in PRIORITY_CLASSES:
            readable = class_name.replace("___", ": ").replace("_", " ")
            f.write(f"{readable}\n")
    
    print(f"\n✓ Class labels saved to: {labels_path}")

def main():
    print("🌱 CropGuard Edge - Day 2: Data Cleaning")
    print("="*70)
    
    # Auto-detect dataset path
    raw_data_path = find_dataset_path()
    
    if raw_data_path is None:
        print("\n" + "="*70)
        print("❌ DATASET NOT FOUND")
        print("="*70)
        print("\nPlease check:")
        print("1. Did the dataset download completely?")
        print("2. Is it in the project directory?")
        print("3. Try: ls -la 'plantvillage dataset'/")
        return
    
    # Create directories
    create_directory_structure()
    
    # Clean and organize
    print("\nStarting data cleaning...")
    stats = clean_and_organize_dataset(raw_data_path)
    
    # Print stats
    print_statistics(stats)
    
    # Save labels
    if stats["by_class"]:
        save_class_labels()
        
        print("\n" + "="*70)
        print("✅ DATA CLEANING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review the statistics above")
        print("  2. Verify: ls data/train/")
        print("  3. Move to Day 3: Data Augmentation")
    else:
        print("\n" + "="*70)
        print("⚠️  DATA CLEANING INCOMPLETE")
        print("="*70)
        print("\nNo classes were processed. Please check the dataset structure.")

if __name__ == "__main__":
    random.seed(42)
    main()