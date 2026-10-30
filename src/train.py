import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
from preprocessing import get_datasets, get_class_weights

def build_model(num_classes=5, input_shape=(224, 224, 3)):
    # MobileNetV2 needs [-1, 1] input range. Rescaling layer handles this.
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False # Freeze the base model
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def plot_training_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training plot to {save_path}")

def train():
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(dataset_dir)
    class_weights = get_class_weights(dataset_dir)
    print(f"Class weights applied for imbalance: {class_weights}")
    
    print("Building model (MobileNetV2)...")
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(models_dir, 'best_model.keras'), save_best_only=True, monitor='val_accuracy')
    ]
    
    print("Starting training...")
    epochs = 10 # 10 epochs for quick execution (can be increased later)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    plot_training_history(history, os.path.join(results_dir, 'training_history.png'))
    print("Training complete. Best model saved to models/best_model.keras")

if __name__ == "__main__":
    train()
