"""
CropGuard Edge - Improved Training Script
With enhanced augmentation, regularization, and fine-tuning strategy
Compatible with TensorFlow 2.x (all versions)

Author: Kivairu Samuel
Enhanced Version
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import TensorFlow (used directly) and pick a Keras backend robustly
import tensorflow as tf

try:
    # Prefer standalone Keras if available (TF 2.16+)
    import keras as keras_pkg
    from keras import layers, Model, regularizers
    from keras.applications import MobileNetV3Small
    keras = keras_pkg
    print("✓ Using standalone Keras package")
except Exception:
    # Fallback to tf.keras for older TF versions
    import tensorflow.keras as keras  # type: ignore
    from tensorflow.keras import layers, Model, regularizers  # type: ignore
    from tensorflow.keras.applications import MobileNetV3Small  # type: ignore
    print("✓ Using TensorFlow's built-in Keras (tf.keras)")

print("="*70)
print("🌱 CropGuard Edge - Enhanced Training")
print("="*70)

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class Config:
    # Data paths
    DATA_DIR = Path("data")
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    
    # Model architecture
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 6
    DROPOUT_RATE = 0.5  # Increased from 0.3
    L2_REG = 0.001  # Reduced from 0.01
    
    # Training - Two phases
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 30  # Frozen base
    EPOCHS_PHASE2 = 20  # Fine-tuning
    
    # Phase 1: Train classifier only
    LEARNING_RATE_PHASE1 = 1e-3
    
    # Phase 2: Fine-tune entire model
    LEARNING_RATE_PHASE2 = 1e-5  # Much lower for fine-tuning
    UNFREEZE_LAYERS = 50  # Unfreeze last 50 layers
    
    # Callbacks
    PATIENCE_EARLY_STOP = 10  # Increased patience
    PATIENCE_REDUCE_LR = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Output
    MODEL_PATH = "models/cropguard_v2.h5"
    LOGS_DIR = "models/training_logs"
    
    # Class weights (if dataset is imbalanced)
    USE_CLASS_WEIGHTS = True

# ============================================================================
# ENHANCED DATA AUGMENTATION
# ============================================================================

def get_augmentation_layer():
    """Enhanced augmentation for better generalization (robust to Keras versions)"""
    ops = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),
    ]
    # RandomBrightness may not exist in all Keras builds -> fallback to a lambda
    if hasattr(layers, "RandomBrightness"):
        ops.append(layers.RandomBrightness(0.2))
    else:
        # brightness in range [-0.2, 0.2] scaled to image values (images are uint8 -> normalized later)
        ops.append(layers.Lambda(lambda x: x + tf.cast(tf.random.uniform(tf.shape(x), minval=-20, maxval=20, dtype=tf.int32), tf.float32) / 100.0))
    # RandomTranslation exists in most versions
    if hasattr(layers, "RandomTranslation"):
        ops.append(layers.RandomTranslation(0.1, 0.1))
    return keras.Sequential(ops, name="augmentation")

# ============================================================================
# DATA LOADING WITH CLASS WEIGHTS
# ============================================================================

def calculate_class_weights(directory):
    """Calculate class weights for imbalanced datasets"""
    from collections import Counter
    
    # Count samples per class
    class_counts = {}
    for class_dir in Path(directory).iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.*')))
            class_counts[class_dir.name] = count
    
    # Calculate weights
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    
    class_weights = {}
    for idx, (class_name, count) in enumerate(sorted(class_counts.items())):
        weight = total / (n_classes * count)
        class_weights[idx] = weight
        print(f"   Class {idx} ({class_name}): {count} samples, weight: {weight:.2f}")
    
    return class_weights

def create_dataset(directory, is_training=True, batch_size=32):
    """Create optimized dataset"""
    
    print(f"\n📂 Loading: {directory}")
    
    # Use image_dataset_from_directory (works in all TF 2.x versions)
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='int',
        shuffle=is_training,
        seed=42
    )
    
    # Ensure dataset is a tf.data.Dataset object
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
    
    class_names = sorted([d.name for d in Path(directory).iterdir() if d.is_dir()])
    print(f"   Classes: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    # Normalization (ImageNet stats)
    normalization = layers.Normalization(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        variance=[(0.229 * 255)**2, (0.224 * 255)**2, (0.225 * 255)**2]
    )
    
    # Enhanced augmentation for training
    if is_training:
        augmentation = get_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Apply normalization
    dataset = dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Cache and prefetch for performance
    if not is_training:
        dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names

# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

def create_model(num_classes=6, dropout_rate=0.5, l2_reg=0.001):
    """Create improved CropGuard model"""
    
    print("\n🧠 Creating model...")
    
    # Load MobileNetV3Small
    base_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=False
    )
    
    # Initially freeze
    base_model.trainable = False
    print(f"   ✓ Base model loaded (frozen)")
    
    # Build model with improved head
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    
    # Improved classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)  # NEW: batch normalization
    x = layers.Dense(
        256,  # Increased from 128
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128,  # NEW: additional layer
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = Model(inputs, outputs)
    
    print(f"   ✓ Model built")
    print(f"   ✓ Total parameters: {model.count_params():,}")
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    return model, base_model

# ============================================================================
# CALLBACKS WITH BETTER MONITORING
# ============================================================================

def get_callbacks(config, phase_name):
    """Create callbacks for each training phase"""
    
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    model_path = config.MODEL_PATH.replace('.h5', f'_{phase_name}.h5')
    
    return [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.PATIENCE_REDUCE_LR,
            min_lr=config.MIN_LR,
            verbose=1,
            min_delta=0.001
        ),
        keras.callbacks.CSVLogger(
            os.path.join(config.LOGS_DIR, f'training_log_{phase_name}.csv')
        ),
        # NEW: Learning rate schedule visualization
        keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.95 if epoch > 0 else lr,
            verbose=0
        )
    ]

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(histories, save_dir):
    """Plot combined training curves from both phases"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Combine histories
    all_acc = histories['phase1'].history['accuracy'] + histories['phase2'].history['accuracy']
    all_val_acc = histories['phase1'].history['val_accuracy'] + histories['phase2'].history['val_accuracy']
    all_loss = histories['phase1'].history['loss'] + histories['phase2'].history['loss']
    all_val_loss = histories['phase1'].history['val_loss'] + histories['phase2'].history['val_loss']
    
    phase1_epochs = len(histories['phase1'].history['accuracy'])
    total_epochs = len(all_acc)
    
    # Accuracy
    axes[0, 0].plot(all_acc, label='Train', linewidth=2)
    axes[0, 0].plot(all_val_acc, label='Val', linewidth=2)
    axes[0, 0].axvline(x=phase1_epochs, color='r', linestyle='--', label='Fine-tuning starts')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(all_loss, label='Train', linewidth=2)
    axes[0, 1].plot(all_val_loss, label='Val', linewidth=2)
    axes[0, 1].axvline(x=phase1_epochs, color='r', linestyle='--', label='Fine-tuning starts')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-2 Accuracy
    all_top2 = histories['phase1'].history['top2_acc'] + histories['phase2'].history['top2_acc']
    all_val_top2 = histories['phase1'].history['val_top2_acc'] + histories['phase2'].history['val_top2_acc']
    
    axes[1, 0].plot(all_top2, label='Train', linewidth=2)
    axes[1, 0].plot(all_val_top2, label='Val', linewidth=2)
    axes[1, 0].axvline(x=phase1_epochs, color='r', linestyle='--', label='Fine-tuning starts')
    axes[1, 0].set_title('Top-2 Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-2 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    all_lr = histories['phase1'].history['learning_rate'] + histories['phase2'].history['learning_rate']
    axes[1, 1].plot(all_lr, linewidth=2, color='green')
    axes[1, 1].axvline(x=phase1_epochs, color='r', linestyle='--', label='Fine-tuning starts')
    axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves_enhanced.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved: {save_path}")
    plt.close()

# ============================================================================
# TWO-PHASE TRAINING
# ============================================================================

def train():
    """Two-phase training: frozen base → fine-tuning"""
    
    config = Config()
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"   Warning: {e}")
    else:
        print("\nℹ️  Training on CPU")
    
    # Load data
    print("\n" + "="*70)
    print("1️⃣ LOADING DATA")
    print("="*70)
    
    train_ds, class_names = create_dataset(
        config.TRAIN_DIR, is_training=True, batch_size=config.BATCH_SIZE
    )
    val_ds, _ = create_dataset(
        config.VAL_DIR, is_training=False, batch_size=config.BATCH_SIZE
    )
    test_ds, _ = create_dataset(
        config.TEST_DIR, is_training=False, batch_size=config.BATCH_SIZE
    )
    
    # Calculate class weights
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        print("\n📊 Calculating class weights:")
        class_weights = calculate_class_weights(config.TRAIN_DIR)
    
    # Create model
    print("\n" + "="*70)
    print("2️⃣ BUILDING MODEL")
    print("="*70)
    
    model, base_model = create_model(
        num_classes=len(class_names),
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REG
    )
    
    # ========================================================================
    # PHASE 1: Train classifier head only
    # ========================================================================
    
    print("\n" + "="*70)
    print("3️⃣ PHASE 1: TRAINING CLASSIFIER HEAD")
    print("="*70)
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 Epochs: {config.EPOCHS_PHASE1}")
    print(f"📚 Learning Rate: {config.LEARNING_RATE_PHASE1}")
    print(f"❄️  Base model: FROZEN")
    print("="*70 + "\n")
    
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE_PHASE1),  # type: ignore
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')
        ]
    )
    
    callbacks_phase1 = get_callbacks(config, 'phase1')
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_PHASE1,
        callbacks=callbacks_phase1,
        class_weight=class_weights,
        verbose='auto'
    )
    
    print(f"\n⏰ Phase 1 finished: {datetime.now().strftime('%H:%M:%S')}")
    
    # ========================================================================
    # PHASE 2: Fine-tune entire model
    # ========================================================================
    
    print("\n" + "="*70)
    print("4️⃣ PHASE 2: FINE-TUNING ENTIRE MODEL")
    print("="*70)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:-config.UNFREEZE_LAYERS]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"🔓 Unfreezing last {trainable_count} layers of base model")
    print(f"📚 Learning Rate: {config.LEARNING_RATE_PHASE2} (much lower)")
    print("="*70 + "\n")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE_PHASE2),  # type: ignore
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')
        ]
    )
    
    callbacks_phase2 = get_callbacks(config, 'phase2')
    
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_PHASE2,
        callbacks=callbacks_phase2,
        class_weight=class_weights,
        verbose='auto'
    )
    
    print(f"\n⏰ Phase 2 finished: {datetime.now().strftime('%H:%M:%S')}")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("5️⃣ FINAL EVALUATION")
    print("="*70)
    
    test_loss, test_acc, test_top2 = model.evaluate(test_ds, verbose='auto')
    
    print(f"\n📊 TEST SET RESULTS:")
    print(f"   Loss:        {test_loss:.4f}")
    print(f"   Accuracy:    {test_acc:.2%}")
    print(f"   Top-2 Acc:   {test_top2:.2%}")
    
    # Save final model
    final_model_path = config.MODEL_PATH.replace('.h5', '_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Save results
    results = {
        'final_metrics': {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'test_top2_accuracy': float(test_top2)
        },
        'phase1': {
            'epochs': len(history_phase1.history['accuracy']),
            'final_train_acc': float(history_phase1.history['accuracy'][-1]),
            'final_val_acc': float(history_phase1.history['val_accuracy'][-1]),
            'best_val_acc': float(max(history_phase1.history['val_accuracy']))
        },
        'phase2': {
            'epochs': len(history_phase2.history['accuracy']),
            'final_train_acc': float(history_phase2.history['accuracy'][-1]),
            'final_val_acc': float(history_phase2.history['val_accuracy'][-1]),
            'best_val_acc': float(max(history_phase2.history['val_accuracy']))
        },
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs_phase1': config.EPOCHS_PHASE1,
            'epochs_phase2': config.EPOCHS_PHASE2,
            'learning_rate_phase1': config.LEARNING_RATE_PHASE1,
            'learning_rate_phase2': config.LEARNING_RATE_PHASE2,
            'dropout_rate': config.DROPOUT_RATE,
            'l2_reg': config.L2_REG
        },
        'classes': class_names
    }
    
    results_path = os.path.join(config.LOGS_DIR, 'results_enhanced.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    # Plot
    plot_training_history(
        {'phase1': history_phase1, 'phase2': history_phase2},
        config.LOGS_DIR
    )
    
    # Final report
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    
    print(f"\n📈 PHASE 1 RESULTS:")
    print(f"   Best Val Accuracy: {max(history_phase1.history['val_accuracy']):.2%}")
    
    print(f"\n📈 PHASE 2 RESULTS:")
    print(f"   Best Val Accuracy: {max(history_phase2.history['val_accuracy']):.2%}")
    
    print(f"\n📈 FINAL TEST RESULTS:")
    print(f"   Test Accuracy: {test_acc:.2%}")
    print(f"   Top-2 Accuracy: {test_top2:.2%}")
    
    if test_acc >= 0.85:
        print(f"\n🎉 EXCELLENT! Model is production-ready!")
    elif test_acc >= 0.75:
        print(f"\n✓ GOOD! Model performs well.")
    else:
        print(f"\n⚠️  Consider collecting more data or adjusting hyperparameters.")
    
    print(f"\n📁 SAVED FILES:")
    print(f"   Final model: {final_model_path}")
    print(f"   Phase 1 model: {config.MODEL_PATH.replace('.h5', '_phase1.h5')}")
    print(f"   Phase 2 model: {config.MODEL_PATH.replace('.h5', '_phase2.h5')}")
    print(f"   Logs: {config.LOGS_DIR}")
    
    print(f"\n🎯 NEXT STEP: Convert to TFLite")
    print("="*70)
    
    return model, history_phase1, history_phase2

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    model, hist1, hist2 = train()