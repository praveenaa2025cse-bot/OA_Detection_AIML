import tensorflow as tf
import numpy as np
from PIL import Image
import os

CLASSES = ['0: Normal / Healthy', '1: Doubtful OA', '2: Minimal / Mild OA', '3: Moderate OA', '4: Severe OA']

def load_prediction_model():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'best_model.keras')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def predict_image(model, image, img_size=(224, 224)):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
        
    image = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASSES[predicted_class_idx], float(confidence), predictions[0].tolist()
