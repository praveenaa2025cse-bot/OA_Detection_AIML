import streamlit as st
import os
import sys
from PIL import Image
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import load_prediction_model, predict_image, CLASSES

st.set_page_config(page_title="OA Early Detection AI", layout="wide", page_icon="🦴")

# Load model (cached to avoid reloading)
@st.cache_resource
def get_model():
    return load_prediction_model()

model = get_model()

# Custom CSS for clean, professional look
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1e3a8a; text-align: center; font-weight: bold; margin-bottom: 2rem; }
    .info-card { padding: 20px; border-radius: 10px; background-color: #f1f5f9; border-left: 6px solid #2563eb; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Dataset Overview", "Model Performance", "Prediction Dashboard", "About the Project"])

st.sidebar.markdown("---")
st.sidebar.info("**Medical Context:** This is an AI-assisted screening tool. It does not provide definitive medical diagnoses. Predictions support further clinical evaluation.")

if menu == "Home":
    st.markdown('<div class="main-header">🦴 AI-Assisted Early Detection System for Osteoarthritis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>Welcome to the OA Risk Assessment Dashboard</h3>
        <p>This system uses a Deep Learning architecture (MobileNetV2) to analyze knee X-ray images and estimate the severity of Osteoarthritis based on the Kellgren-Lawrence grading scale.</p>
        <p><strong>Focus:</strong> Designed to support early identification and preliminary assessment where specialized orthopedic care may be limited.</p>
    </div>
    """, unsafe_allow_html=True)
    
elif menu == "Dataset Overview":
    st.header("Dataset Analytics")
    st.write("Using the standard Osteoarthritis Knee X-ray dataset, containing 9,786 images across 5 severity classes.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", "9,786")
    col2.metric("Classes", "5 (KL Grades)")
    col3.metric("Image Dimensions", "224x224")
    col4.metric("Format", "PNG")
    
    st.subheader("Class Distribution (Train Split)")
    dist_data = {
        'Class': ['0 (Healthy)', '1 (Doubtful)', '2 (Minimal)', '3 (Moderate)', '4 (Severe)'],
        'Count': [2286, 1046, 1516, 757, 173]
    }
    df = pd.DataFrame(dist_data)
    st.bar_chart(df.set_index('Class'))
    st.caption("*Class imbalance is handled via dynamic class weighting during the ML training phase.*")

elif menu == "Model Performance":
    st.header("Model Training & Evaluation Results")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    history_path = os.path.join(results_dir, 'training_history.png')
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    report_path = os.path.join(results_dir, 'classification_report.txt')
    
    if os.path.exists(history_path):
        st.subheader("Training vs Validation Graph")
        st.image(history_path, use_column_width=True)
    else:
        st.info("Train the model first to generate training graphs.")
        
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix")
        st.image(cm_path, width=700)
    
    if os.path.exists(report_path):
        st.subheader("Classification Report (Precision, Recall, F1)")
        with open(report_path, 'r') as f:
            st.code(f.read(), language='text')
            
elif menu == "Prediction Dashboard":
    st.header("Upload Knee X-Ray for AI Prediction")
    
    if model is None:
        st.error("Model not found! The backend system needs to complete the training phase first (`python src/train.py`).")
    else:
        st.markdown("Upload a standard AP view Knee X-Ray (PNG/JPG format).")
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption='Uploaded X-Ray', use_column_width=True)
                
            with col2:
                with st.spinner('AI is analyzing the X-ray...'):
                    pred_class, confidence, probabilities = predict_image(model, image)
                
                st.success("Analysis Complete!")
                st.markdown(f"### Assessment: **{pred_class}**")
                st.markdown(f"### AI Confidence: **{confidence*100:.1f}%**")
                
                st.progress(float(confidence))
                
                st.write("#### Probability Breakdown:")
                prob_df = pd.DataFrame({
                    'Class': CLASSES,
                    'Probability': probabilities
                })
                st.bar_chart(prob_df.set_index('Class'))
                
elif menu == "About the Project":
    st.header("About This Project")
    st.write("""
    **Problem Statement ID:** 26004
    **Project Title:** AI-Assisted Early Detection System for Osteoarthritis (OA) Risk Markers in North Eastern Region (NER)
    
    **Project Objective:**
    Develop a clean and user-friendly AI/ML application that can analyze an Osteoarthritis dataset, train suitable deep learning models, predict OA-related classes, and visually present the results to support early identification.
    
    **Technology Stack:**
    - Python (Backend)
    - TensorFlow/Keras (MobileNetV2 Transfer Learning Architecture)
    - Pandas, NumPy, Scikit-learn (Data Processing & Evaluation)
    - Streamlit (Frontend Dashboard Interface)
    """)
