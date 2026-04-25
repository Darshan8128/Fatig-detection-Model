"""
════════════════════════════════════════════════════════════════
            STEP 3: TRAIN FATIGUE CNN MODEL
════════════════════════════════════════════════════════════════
MobileNetV2 transfer learning for fatigue vs non_fatigue detection.
Two-phase training: freeze → fine-tune with RTX 2050 optimization.
"""
import os
import sys
import json
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = Path(__file__).parent
FINAL_DATASET_DIR = PROJECT_ROOT / "final_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# GPU config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FatigueModelTrainer:
    """Train MobileNetV2 for fatigue detection."""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.phase2_history = None
    
    def log(self, message):
        """Print message."""
        print(message)
    
    def setup_gpu(self):
        """Configure GPU for RTX 2050."""
        self.log("\n" + "=" * 70)
        self.log("GPU CONFIGURATION")
        self.log("=" * 70)
        
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        self.log("✅ Mixed precision (float16) enabled")
        
        # Set memory growth
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            self.log(f"✅ Memory growth enabled for {gpu}")
        
        # Limit GPU memory to 3500MB (safe for RTX 2050)
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
            )
            self.log("✅ GPU memory limited to 3500MB")
        except:
            self.log("⚠️  Could not limit GPU memory (non-critical)")
    
    def create_model(self):
        """Create MobileNetV2 model."""
        self.log("\n" + "=" * 70)
        self.log("BUILDING MODEL ARCHITECTURE")
        self.log("=" * 70)
        
        # Load MobileNetV2 with ImageNet weights
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        self.log("✅ Base model layers frozen (224x224 input, ImageNet weights)")
        
        # Build head
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs, outputs)
        
        self.log("\n✅ Model architecture:")
        self.log("   GlobalAveragePooling2D")
        self.log("   → Dense(256, relu) + BatchNorm + Dropout(0.4)")
        self.log("   → Dense(128, relu) + BatchNorm + Dropout(0.3)")
        self.log("   → Dense(1, sigmoid)")
        
        return self.model
    
    def get_data_generators(self):
        """Create data generators for train/val."""
        self.log("\n" + "=" * 70)
        self.log("DATA GENERATORS")
        self.log("=" * 70)
        
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation (no augmentation, just rescale)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        
        # Dataset paths - map to fatigue and non_fatigue
        train_dir = FINAL_DATASET_DIR / "train"
        val_dir = FINAL_DATASET_DIR / "val"
        
        # Check if we need to create binary classification structure
        # (merge eye_open/eye_closed if present)
        train_fatigue_dir = train_dir / "fatigue"
        train_nonfatigue_dir = train_dir / "non_fatigue"
        
        if not train_fatigue_dir.exists():
            self.log(f"❌ {train_fatigue_dir} not found!")
            return None, None
        
        if not train_nonfatigue_dir.exists():
            self.log(f"❌ {train_nonfatigue_dir} not found!")
            return None, None
        
        self.log(f"✅ Train data: {train_dir}")
        self.log(f"✅ Val data: {val_dir}")
        
        # Load batches
        train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=(224, 224),
            batch_size=8,
            class_mode='binary',
            classes={'fatigue': 1, 'non_fatigue': 0},
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=(224, 224),
            batch_size=8,
            class_mode='binary',
            classes={'fatigue': 1, 'non_fatigue': 0},
            seed=42
        )
        
        self.log(f"\n📊 Data loaded:")
        self.log(f"   • Train batches: {len(train_generator)}")
        self.log(f"   • Val batches: {len(val_generator)}")
        
        return train_generator, val_generator
    
    def compile_model(self, learning_rate):
        """Compile model."""
        optimizer = Adam(learning_rate=learning_rate)
        
        # Cast optimizer to float32 for mixed precision
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.log(f"✅ Model compiled (Adam, lr={learning_rate})")
    
    def train_phase1(self, train_gen, val_gen):
        """Phase 1: Train head only (10 epochs)."""
        self.log("\n" + "=" * 70)
        self.log("PHASE 1: TRAIN HEAD ONLY (10 epochs)")
        self.log("=" * 70)
        
        self.compile_model(1e-4)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / "fatigue_model_phase1.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(LOGS_DIR / "tb_fatigue_phase1"),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        self.log("\n📚 Training...")
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )
        
        self.log("\n✅ Phase 1 complete.")
    
    def train_phase2(self, train_gen, val_gen):
        """Phase 2: Fine-tune last 30 layers (20 epochs)."""
        self.log("\n" + "=" * 70)
        self.log("PHASE 2: FINE-TUNE LAST 30 LAYERS (20 epochs)")
        self.log("=" * 70)
        
        # Unfreeze last 30 layers
        base_model = self.model.layers[1]  # MobileNetV2 is second layer (after inputs)
        total_layers = len(base_model.layers)
        unfreeze_from = total_layers - 30
        
        for layer in base_model.layers[unfreeze_from:]:
            layer.trainable = True
        
        self.log(f"✅ Unfroze layers {unfreeze_from}-{total_layers}")
        
        # Compile with lower learning rate
        self.compile_model(1e-5)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / "fatigue_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(LOGS_DIR / "tb_fatigue_phase2"),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        self.log("\n📚 Fine-tuning...")
        self.phase2_history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=callbacks,
            verbose=1
        )
        
        self.log("\n✅ Phase 2 complete.")
    
    def save_models(self):
        """Save models in H5 and TFLite formats."""
        self.log("\n" + "=" * 70)
        self.log("SAVING MODELS")
        self.log("=" * 70)
        
        # H5 (already saved by checkpoint, but save final)
        h5_path = MODELS_DIR / "fatigue_model.h5"
        self.model.save(h5_path)
        self.log(f"✅ Saved: {h5_path}")
        
        # TFLite
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            tflite_model = converter.convert()
            
            tflite_path = MODELS_DIR / "fatigue_model.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            self.log(f"✅ Saved: {tflite_path}")
        except Exception as e:
            self.log(f"⚠️  Failed to save TFLite: {e}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_dict = {
            'phase1': {
                'loss': [float(x) for x in self.history.history['loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
            },
            'phase2': {
                'loss': [float(x) for x in self.phase2_history.history['loss']],
                'accuracy': [float(x) for x in self.phase2_history.history['accuracy']],
                'val_loss': [float(x) for x in self.phase2_history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.phase2_history.history['val_accuracy']]
            }
        }
        
        history_file = MODELS_DIR / "fatigue_history.json"
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.log(f"✅ Saved: {history_file}")
    
    def run(self):
        """Main pipeline."""
        self.log("\n" + "=" * 70)
        self.log("TRAINING FATIGUE DETECTION MODEL")
        self.log("=" * 70)
        
        # Setup
        self.setup_gpu()
        
        # Create model
        self.create_model()
        
        # Get data
        train_gen, val_gen = self.get_data_generators()
        if train_gen is None:
            self.log("❌ Failed to load data generators!")
            return False
        
        # Train
        self.train_phase1(train_gen, val_gen)
        self.train_phase2(train_gen, val_gen)
        
        # Save
        self.save_models()
        self.save_history()
        
        # Summary
        self.log("\n" + "=" * 70)
        self.log("TRAINING COMPLETE")
        self.log("=" * 70)
        self.log(f"\n✅ Fatigue model saved to {MODELS_DIR}/")
        self.log(f"\nNext steps:")
        self.log(f"  • STEP 4: Train Eye Model")
        self.log(f"  • Logs: {LOGS_DIR}/tb_fatigue_*/\n")
        
        return True


def main():
    """Main entry point."""
    try:
        trainer = FatigueModelTrainer()
        trainer.run()
        print("\n✅ [train_fatigue_model.py] complete. Type NEXT for next file.\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
