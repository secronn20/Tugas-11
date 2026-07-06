import os
# Disable oneDNN custom operations to prevent CPU memory allocation bugs on Windows
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure Matplotlib to use the headless 'Agg' backend to save RAM
import matplotlib
matplotlib.use('Agg')

import shutil
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Setup random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
DATASET_DIR = "Surface Crack Detection"
SUBSET_DIR = "dataset_subset"
TRAIN_RATIO = 0.8
SAMPLES_PER_CLASS = 40   # 32 for training, 8 for validation (80 images total - adjusted for small dataset)
BATCH_SIZE = 1           # Batch size 1 to minimize memory usage
EPOCHS = 10              # Trains instantly on features
IMAGE_SIZE = (224, 224)

def prepare_dataset_subset():
    print("--- 1. Preparing Dataset Subset ---")
    train_dir = os.path.join(SUBSET_DIR, "train")
    val_dir = os.path.join(SUBSET_DIR, "val")
    
    classes = ["Negative", "Positive"]
    
    # Create folder structure
    for c in classes:
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)
        
    for c in classes:
        src_class_dir = os.path.join(DATASET_DIR, c)
        all_images = os.listdir(src_class_dir)
        all_images = [img for img in all_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sample files
        sampled_images = random.sample(all_images, SAMPLES_PER_CLASS)
        
        # Split into train and val
        split_idx = int(SAMPLES_PER_CLASS * TRAIN_RATIO)
        train_images = sampled_images[:split_idx]
        val_images = sampled_images[split_idx:]
        
        print(f"Class '{c}': Copying {len(train_images)} train images and {len(val_images)} val images...")
        
        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(src_class_dir, img), os.path.join(train_dir, c, img))
        for img in val_images:
            shutil.copy(os.path.join(src_class_dir, img), os.path.join(val_dir, c, img))
            
    print("Dataset subset ready!\n")

def extract_features(directory, base_model, tf_module):
    """Run VGG16 inference to pre-extract features with batch size 1 and aggressive memory cleanup."""
    features = []
    labels = []
    classes = ["Negative", "Positive"]
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        filenames = os.listdir(class_dir)
        
        print(f"Extracting features for '{class_name}' in '{directory}'...")
        
        for idx, fname in enumerate(filenames):
            img_path = os.path.join(class_dir, fname)
            
            # Load and preprocess a single image
            img = tf_module.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
            img_arr = tf_module.preprocessing.image.img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # Run prediction (use direct functional call in eager mode to prevent memory leaks)
            batch_features = base_model(img_arr, training=False).numpy()
            features.append(batch_features)
            labels.append(class_idx)
            
            # Garbage collect periodically to keep memory footprint low
            if idx % 10 == 0:
                gc.collect()
                
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    gc.collect()
    return features, labels

def train_transfer_learning():
    # Import TensorFlow only when needed to keep memory usage low at start
    print("Importing TensorFlow...")
    import tensorflow as tf
    
    # Configure TensorFlow to use only 1 thread to minimize memory buffers
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    tf.keras.backend.clear_session()
    
    print("\n--- 2. Loading VGG16 Pre-trained Base Model ---")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    print("\n--- 3. Pre-computing Features (Offline Feature Extraction) ---")
    train_features, train_labels = extract_features(os.path.join(SUBSET_DIR, "train"), base_model, tf.keras)
    val_features, val_labels = extract_features(os.path.join(SUBSET_DIR, "val"), base_model, tf.keras)
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    
    # Free base model to reclaim memory since features are extracted
    del base_model
    gc.collect()
    tf.keras.backend.clear_session()
    
    print("\n--- 4. Building Head Classifier Model ---")
    classifier_model = Sequential([
        Flatten(input_shape=train_features.shape[1:]),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    classifier_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    classifier_model.summary()
    
    print("\n--- 5. Training Classifier Model ---")
    # Training the classifier is instant and uses very little memory
    history = classifier_model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=EPOCHS,
        batch_size=16
    )
    
    print("\n--- 6. Assembling and Compiling Full Model ---")
    # Reload base model to reconstruct full model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    for layer in base_model.layers:
        layer.trainable = False
        
    full_model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Copy the trained weights
    print("Transferring trained weights to combined model...")
    full_model.layers[2].set_weights(classifier_model.layers[1].get_weights())
    full_model.layers[4].set_weights(classifier_model.layers[3].get_weights())
    
    full_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n--- 7. Saving Combined Model ---")
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "road_crack_vgg16.h5")
    full_model.save(model_path)
    print(f"Full VGG16 model saved to '{model_path}'")
    
    # Save training curves
    os.makedirs(os.path.join("static", "plots"), exist_ok=True)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#6366f1', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#06b6d4', marker='o')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#ef4444', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', marker='o')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join("static", "plots", "training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to '{plot_path}'")
    
    # Cleanup subset directory
    print("\n--- 8. Cleaning Up Dataset Subset ---")
    try:
        shutil.rmtree(SUBSET_DIR)
        print("Dataset subset directory cleaned up successfully.")
    except Exception as e:
        print(f"Warning: Could not clean up dataset subset directory: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found!")
    else:
        prepare_dataset_subset()
        train_transfer_learning()
        print("\nAll done!")
