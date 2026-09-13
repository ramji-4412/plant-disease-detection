# =============================================================================
# Plant Disease Detection - Model Training for Google Colab
# =============================================================================
# Updated for kagglehub + vipoooool/new-plant-diseases-dataset
# Copy each CELL section into a separate Google Colab cell and run sequentially
# =============================================================================

# =============================================================================
# CELL 1: Setup (GPU Check) - ALREADY DONE BY YOU
# =============================================================================
# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# print("GPU Available:", tf.config.list_physical_devices('GPU'))

# =============================================================================
# CELL 2: Download Dataset - ALREADY DONE BY YOU
# =============================================================================
# import kagglehub
# path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
# print("Path to dataset files:", path)

# =============================================================================
# CELL 3: Mount Google Drive
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# CELL 4: Explore Dataset Structure
# =============================================================================
import os

# Set the path from kagglehub output - ADJUST THIS TO YOUR ACTUAL PATH
# Usually it's something like: /root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2
DATASET_PATH = '/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2'

print("Dataset path:", DATASET_PATH)
print("\nTop-level contents:")
for item in os.listdir(DATASET_PATH):
    print(f"  {item}")

# Check if there's a subfolder (some versions have one extra folder level)
sub_path = os.path.join(DATASET_PATH, 'new plant diseases dataset(augmented)')
if os.path.exists(sub_path):
    DATASET_PATH = sub_path
    print(f"\nAdjusted path: {DATASET_PATH}")

print("\nDataset folders (classes):")
class_folders = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])
print(f"Found {len(class_folders)} class folders")
for folder in class_folders[:10]:
    print(f"  {folder}")
if len(class_folders) > 10:
    print(f"  ... and {len(class_folders) - 10} more")

# =============================================================================
# CELL 5: Import Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3

print("All libraries imported successfully!")

# =============================================================================
# CELL 6: Configuration
# =============================================================================
IMG_SIZE = 224          # Input image size (matches your app.py)
BATCH_SIZE = 32         # Batch size for training
EPOCHS = 50             # Maximum number of epochs
LEARNING_RATE = 0.001   # Initial learning rate
FINE_TUNE_LR = 0.0001   # Learning rate for fine-tuning

# Count actual number of classes from the dataset
NUM_CLASSES = len(class_folders)
print(f"Number of classes detected: {NUM_CLASSES}")

# Path to save the best model in Google Drive
MODEL_SAVE_PATH = '/content/drive/MyDrive/plant_disease_model.keras'

print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Dataset path: {DATASET_PATH}")
print(f"Model save path: {MODEL_SAVE_PATH}")

# =============================================================================
# CELL 7: Data Augmentation and Generators
# =============================================================================
# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  # 80-20 train-validation split
)

# Validation data generator (only rescaling)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Creating training generator...")
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

print("Creating validation generator...")
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")


# =============================================================================
# CELL 8: Build the Model (Transfer Learning with EfficientNetB3)
# =============================================================================
def build_model(num_classes, img_size, fine_tune=False):
    """Build EfficientNetB3 model with custom classification head"""
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = fine_tune
    
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Build model for initial training (base model frozen)
model = build_model(NUM_CLASSES, IMG_SIZE, fine_tune=False)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# Count trainable and non-trainable parameters
trainable = sum([np.prod(w.shape) for w in model.trainable_weights])
non_trainable = sum([np.prod(w.shape) for w in model.non_trainable_weights])
print(f"\nTrainable parameters: {int(trainable):,}")
print(f"Non-trainable parameters: {int(non_trainable):,}")

# =============================================================================
# CELL 9: Define Callbacks
# =============================================================================
callbacks = [
    # Save the best model based on validation accuracy
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Stop training if validation loss doesn't improve for 10 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("Callbacks configured!")

# =============================================================================
# CELL 10: Phase 1 - Initial Training (Base Model Frozen)
# =============================================================================
print("=" * 60)
print("PHASE 1: Initial Training (Base Model Frozen)")
print("=" * 60)

history1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nPhase 1 training complete!")
print(f"Best validation accuracy so far: {max(history1.history['val_accuracy']):.4f}")


# =============================================================================
# CELL 11: Plot Phase 1 Training History
# =============================================================================
def plot_history(history, title="Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_history(history1, "Phase 1 Training History (Frozen Base)")

# =============================================================================
# CELL 12: Phase 2 - Fine-Tuning (Unfreeze Base Model)
# =============================================================================
print("=" * 60)
print("PHASE 2: Fine-Tuning (Unfreezing Base Model)")
print("=" * 60)

# Load the best model from Phase 1
model = load_model(MODEL_SAVE_PATH)

# Unfreeze the base model layers
base_model = model.layers[0]
base_model.trainable = True

# Freeze first 100 layers to prevent catastrophic forgetting
FROZEN_LAYERS = 100
for layer in base_model.layers[:FROZEN_LAYERS]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Count trainable and non-trainable parameters
trainable = sum([np.prod(w.shape) for w in model.trainable_weights])
non_trainable = sum([np.prod(w.shape) for w in model.non_trainable_weights])
print(f"\nTrainable parameters: {int(trainable):,}")
print(f"Non-trainable parameters: {int(non_trainable):,}")

# Fine-tune the model
history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=len(history1.history['loss']),
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nPhase 2 training complete!")


# =============================================================================
# CELL 13: Plot Complete Training History
# =============================================================================
def plot_complete_history(h1, h2, title="Complete Training History"):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)
    
    epochs_range = range(1, len(acc) + 1)
    phase1_epochs = len(h1.history['accuracy'])
    
    axes[0].plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].axvline(x=phase1_epochs, color='g', linestyle='--', label='Fine-tune Start')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1].axvline(x=phase1_epochs, color='g', linestyle='--', label='Fine-tune Start')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_complete_history(history1, history2, "Complete Training History")

# =============================================================================
# CELL 14: Model Evaluation
# =============================================================================
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

best_model = load_model(MODEL_SAVE_PATH)

val_loss, val_accuracy, val_precision, val_recall = best_model.evaluate(val_generator)

print(f"\nValidation Results:")
print(f"  Loss:      {val_loss:.4f}")
print(f"  Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
print(f"  F1-Score:  {f1:.4f}")

# =============================================================================
# CELL 15: Classification Report
# =============================================================================
val_generator.reset()
predictions = best_model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print("\nClassification Report:")
print("=" * 80)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# =============================================================================
# CELL 16: Save and Download Model
# =============================================================================
from datetime import datetime
import shutil

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_model_path = f'/content/plant_disease_model_{timestamp}.keras'
shutil.copy(MODEL_SAVE_PATH, final_model_path)

print(f"Best model saved at: {MODEL_SAVE_PATH}")
print(f"Timestamped copy: {final_model_path}")

# Save class names for reference
class_names_path = '/content/class_names.json'
with open(class_names_path, 'w') as f:
    json.dump(class_labels, f, indent=2)

# Download files
from google.colab import files
print("\nDownloading model to your local machine...")
files.download(MODEL_SAVE_PATH)
files.download(class_names_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFinal Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Model input size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"\nDownload the model and replace your 'plant_disease_model.keras'")
print("Your app.py will work without any changes!")
