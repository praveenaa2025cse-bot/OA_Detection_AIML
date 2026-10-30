import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import get_datasets

def evaluate():
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    model_path = os.path.join(models_dir, 'best_model.keras')
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return
        
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test dataset...")
    _, _, test_ds = get_datasets(dataset_dir)
    
    print("Evaluating model...")
    results = model.evaluate(test_ds)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    print("Generating predictions...")
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
    classes = ['0-Healthy', '1-Doubtful', '2-Minimal', '3-Moderate', '4-Severe']
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:\n", report)
    
    # Save Report
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    print(f"Saved confusion matrix to {results_dir}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
