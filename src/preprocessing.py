import tensorflow as tf
import os
import numpy as np
from sklearn.utils import class_weight

def get_datasets(dataset_dir, batch_size=32, img_size=(224, 224)):
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    # Apply Augmentation to training dataset map
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=tf.data.AUTOTUNE)
                            
    # Normalize inputs specifically for MobileNetV2 requirements within the model, 
    # but image_dataset_from_directory loads as 0-255. 
    # MobileNetV2 needs preprocess_input, we will include it in the model architecture or preprocessing layer.
                            
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def get_class_weights(dataset_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    
    y = []
    for cls_idx, cls_name in enumerate(classes):
        cls_path = os.path.join(train_dir, cls_name)
        num_images = len(os.listdir(cls_path))
        y.extend([cls_idx] * num_images)
        
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    return {i: weight for i, weight in enumerate(weights)}
