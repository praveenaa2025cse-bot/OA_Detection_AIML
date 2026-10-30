import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

CLASSES = ['0: Normal', '1: Doubtful OA', '2: Minimal OA', '3: Moderate OA', '4: Severe OA']
model = None
model_loaded = False

def load_model():
    global model, model_loaded
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'best_model.keras')
    if os.path.exists(model_path) and model is None:
        try:
            model = tf.keras.models.load_model(model_path)
            model_loaded = True
        except:
            model_loaded = False
    return model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        return None

def generate_ai_heatmap(img_path, output_path, img_array=None, m=None):
    img = cv2.imread(img_path)
    if img is None: return False
    
    # Try actual Grad-CAM first
    heatmap_generated = False
    if m is not None and img_array is not None:
        heatmap = make_gradcam_heatmap(img_array, m)
        if heatmap is not None:
            heatmap = np.uint8(255 * heatmap)
            jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
            superimposed_img = cv2.addWeighted(jet, 0.5, img, 0.5, 0)
            cv2.imwrite(output_path, superimposed_img)
            heatmap_generated = True
            
    # Fallback to OpenCV Feature Edge map if Grad-CAM fails
    if not heatmap_generated:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        edges = cv2.Canny(blur, 40, 120)
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
        cv2.imwrite(output_path, superimposed_img)
        
    return heatmap_generated

def is_model_ready():
    load_model()
    return model_loaded

def predict_xray(image_path, heatmap_path):
    m = load_model()
    
    if m is None:
        generate_ai_heatmap(image_path, heatmap_path)
        return None, None, None, False
    
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    preds = m.predict(img_array, verbose=0)[0]
    probabilities = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    idx = np.argmax(preds)
    
    # Pass img_array and model to attempt true Grad-CAM
    is_gradcam = generate_ai_heatmap(image_path, heatmap_path, img_array, m)
    
    return CLASSES[idx], float(preds[idx]), probabilities, is_gradcam
