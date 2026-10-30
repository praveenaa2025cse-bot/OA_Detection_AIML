AI-ASSISTED EARLY DETECTION SYSTEM FOR OSTEOARTHRITIS (OA) RISK MARKERS IN NORTH EASTERN REGION (NER)

OCTOBER 2026

PROJECT REPORT SUBMITTED IN PARTIAL FULFILLMENT OF THE
REQUIREMENTS FOR THE AWARD OF THE DEGREE OF BACHELOR OF ENGINEERING
IN COMPUTER SCIENCE AND ENGINEERING
OF THE ANNA UNIVERSITY

AI & ML PROJECT WORK

Submitted by
[YOUR NAME 1] - [REGISTER NUMBER 1]
[YOUR NAME 2] - [REGISTER NUMBER 2]

BATCH 2025 - 2029

Under the Guidance of
Dr. N. SARANYA M.E., Ph. D,
Assistant Professor, CSE.

Department of Computer Science and Engineering 
Sri Eshwar College of Engineering
(An Autonomous Institution – Affiliated to Anna University)
COIMBATORE – 641 202

---

BONAFIDE CERTIFICATE

Certified that this Report titled “AI-ASSISTED EARLY DETECTION SYSTEM FOR OSTEOARTHRITIS (OA) RISK MARKERS IN NORTH EASTERN REGION (NER)“ for the course U23AM495 - Artificial Intelligence and Machine Learning is the bonafide work of

[YOUR NAME 1]
[REGISTER NUMBER 1]

[YOUR NAME 2]
[REGISTER NUMBER 2]

who carried out the project work under my supervision.

SIGNATURE
Dr. R. SUBHA M.E., Ph. D.
HEAD OF THE DEPARTMENT
Dept. of Computer Science & Engineering, Sri Eshwar College of Engineering, Coimbatore – 641 202.

SIGNATURE
Dr. N. SARANYA M.E., Ph. D SUPERVISOR
Assistant Professor,
Dept. of Computer Science & Engineering, Sri Eshwar College of Engineering, Coimbatore – 641 202.

Submitted for the Autonomous Semester End Project Viva-Voce held on …………………..

INTERNAL EXAMINER                                EXTERNAL EXAMINER

---

DECLARATION

[YOUR NAME 1] – [REGISTER NUMBER 1]
[YOUR NAME 2] – [REGISTER NUMBER 2]

To declare that the project entitled “AI-ASSISTED EARLY DETECTION SYSTEM FOR OSTEOARTHRITIS (OA) RISK MARKERS IN NORTH EASTERN REGION (NER)” submitted in partial fulfilment to the University as the project work of Bachelor of Engineering (Computer Science and Engineering) Degree, is a record of original work done by us under the supervision and guidance of Dr. N. SARANYA M.E., Ph.D, Assistant Professor, Department of Computer Science and Engineering, Sri Eshwar College of Engineering, Coimbatore.

Place: Coimbatore
Date:

[YOUR NAME 1]
[YOUR NAME 2]

Project Guided by,
Dr. N. SARANYA M.E., Ph. D

---

ACKNOWLEDGEMENT

The success of a work depends on a team and cooperation. I take this opportunity to express my gratitude and thanks to everyone who helped me in my project. I would like to thank the management for the constant support provided by them to complete this project.

It is indeed our great honor bounded duty to thank our beloved Chairman Mr. R. Mohanram, for his academic interest shown towards the students.

We are indebted to our Director Mr. R. Rajaram, for motivating and providing us with all facilities.

I wish to express my sincere regards and deep sense of gratitude to Dr. Sudha Mohanram, M.E, Ph.D. Principal, for the excellent facilities and encouragement provided during the course of the study and project.

We are indebted to Dr. R. Subha, M.E., Ph.D. Dean, Computing Science for having permitted us to carry out this project and giving the complete freedom to utilize the resources.

I express my sincere thanks to my course faculty Dr. N. SARANYA M.E., Ph. D, Assistant Professor of Computer Science and Engineering Department for the valuable guidance and encouragement given, to us for this project.

I solemnly express our thanks to all the teaching and nonteaching staff of the Computer Science and Engineering Department, family and friends for their valuable support which inspired us to work on this project.

---

ABSTRACT

Osteoarthritis (OA) is a prevalent musculoskeletal condition that causes joint pain, stiffness, and reduced mobility, severely impacting the quality of life. In regions like the North Eastern Region (NER), challenging terrains, an aging population, and limited access to specialized orthopaedic care make the early detection of OA exceptionally difficult. 

This project aims to develop an AI-assisted early detection system to analyze Knee X-ray images and identify OA-related risk markers and severity patterns. Utilizing the standard Kellgren-Lawrence (KL) grading system, the proposed system categorizes X-ray images into five severity classes (0 to 4). The core methodology involves deep transfer learning using the MobileNetV2 architecture to extract fine-grained features from medical imagery. 

The system addresses the inherent class imbalance in medical datasets through dynamic class weighting and extensive data augmentation techniques. An interactive web-based dashboard developed using Streamlit provides a clean, user-friendly interface for researchers and medical professionals to upload X-ray images, receive real-time AI assessments, and review comprehensive dataset analytics and model performance metrics. 

The expected outcome is a practical AI screening tool that provides a preliminary assessment, enabling early identification of OA and supporting further clinical evaluation by healthcare professionals, ultimately facilitating better medical decision-making in resource-limited areas.

---

TABLE OF CONTENTS

1. INTRODUCTION
2. SYSTEM ANALYSIS AND DESIGN
   2.1 Existing System
   2.2 Problem Statement
   2.3 Objectives
   2.4 Scope of the Project
   2.5 System Requirements
3. PROPOSED SOLUTION
   3.1 Overview
   3.2 System Architecture
   3.3 Methodology
   3.4 Dataset Description
   3.5 Data Preprocessing
   3.6 Feature Engineering
   3.7 Model Development
   3.8 Evaluation Metrics
4. IMPLEMENTATION AND RESULTS
   4.1 Implementation
   4.2 Experimental Results
   4.3 Model Comparison
   4.4 Output/Screenshots
   4.5 Discussion
5. CONCLUSION AND FUTURE ENHANCEMENT
   5.1 Conclusion
   5.2 Future Enhancement
6. REFERENCES

---

CHAPTER 1
INTRODUCTION

Osteoarthritis (OA) is the most common form of arthritis, primarily affecting the cartilage in joints such as the knees. It leads to progressive joint degradation, causing pain and limiting physical mobility. Early diagnosis is crucial for effective management and for slowing the progression of the disease. However, in regions such as the North Eastern Region (NER), demographic and geographic challenges—such as difficult terrains and a lack of specialized medical facilities—hinder timely and accurate diagnosis.

Traditional diagnosis relies heavily on the manual inspection of X-ray images by experienced radiologists. This process can be time-consuming, subjective, and prone to human error, particularly in areas with a shortage of medical experts. Artificial Intelligence (AI) and Machine Learning (ML), specifically Deep Learning, offer a promising solution by automating the analysis of medical imagery.

The proposed system addresses this issue by utilizing a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture to analyze knee X-rays and classify them according to the Kellgren-Lawrence severity scale. The project includes a complete machine learning pipeline, from data preprocessing and augmentation to model training and evaluation. Furthermore, an intuitive web dashboard is provided to allow users to upload images and instantly view the AI's preliminary assessment, serving as a powerful decision-support tool for early medical intervention.

---

CHAPTER 2
SYSTEM ANALYSIS AND DESIGN

2.1 Existing Scenario
Currently, the assessment of Osteoarthritis heavily relies on clinical examinations and the manual reading of radiographic images (X-rays). Radiologists assess joint space narrowing and the presence of osteophytes to determine severity. However, manual interpretation is subjective and suffers from high inter-observer variability. Furthermore, rural and challenging geographic regions face a severe shortage of skilled orthopaedic specialists, leading to delayed diagnoses and poorer patient outcomes.

2.2 Problem Statement
In the North Eastern Region, difficult terrain, physically demanding livelihoods, an aging population, and limited access to specialized orthopaedic care can make early identification of Osteoarthritis difficult. The proposed system should use AI/ML techniques to analyze relevant osteoarthritis data (X-ray images) and identify OA-related risk markers or severity patterns. The system should provide an AI-assisted preliminary assessment that can support further clinical evaluation.

2.3 Objectives
- Data Processing: To preprocess and augment a large-scale dataset of knee X-rays to make them suitable for deep learning.
- Imbalance Handling: To implement techniques like class-weighting to combat the severe class imbalance typically found in medical datasets.
- Model Training: To train a highly accurate CNN (MobileNetV2) capable of classifying X-rays into 5 severity levels.
- Web Application: To build a modern, clean Streamlit dashboard for real-time predictions, data visualization, and analytics.
- Decision Support: To assist healthcare professionals with an AI-based second opinion for early OA detection.

2.4 Scope of the Project
Included in the project:
- Processing Kaggle Osteoarthritis image datasets.
- Using Transfer Learning (MobileNetV2) for image classification.
- Developing a user-friendly Streamlit web interface.
- Generating classification reports, confusion matrices, and training history graphs.

Outside the current scope:
- Claiming definitive medical diagnosis (the tool is for preliminary screening only).
- Hardware or IoT integrations.

2.5 System Requirements

A. Hardware Requirements
Processor: Dual-core processor or equivalent (Intel i5/i7 recommended)
RAM: 8 GB minimum; 16 GB recommended for model training
Storage: 10 GB of available space for dataset and models

B. Software Requirements
Programming Language: Python 3.8+
Libraries: TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Pillow
Web Framework: Streamlit
IDE: Visual Studio Code

---

CHAPTER 3
PROPOSED SOLUTION

3.1 Overview
The proposed system is a web-based AI application designed to classify knee X-ray images into one of five Osteoarthritis severity grades. The system takes an uploaded X-ray image, preprocesses it to the required dimensions and pixel values, and passes it through a trained Deep Learning model to generate a prediction and confidence score.

3.2 System Architecture
1. Data Input: Loading Train, Validation, and Test images.
2. Preprocessing: Resizing to 224x224, applying data augmentation (flip, rotation, zoom), and normalizing pixels.
3. Model Training: Transfer learning using MobileNetV2 with a custom dense classification head.
4. Evaluation: Testing the model on unseen data to generate metrics (Precision, Recall, F1-Score).
5. Deployment: Streamlit web dashboard for user interaction.

3.3 Dataset Description
Dataset Name: Kaggle Osteoarthritis Knee X-ray Dataset
Total Samples: 9,786 images
Classes: 5 Classes corresponding to Kellgren-Lawrence (KL) grades:
0: Normal / Healthy
1: Doubtful OA
2: Minimal / Mild OA
3: Moderate OA
4: Severe OA

3.4 Data Preprocessing
- Resizing: All images are resized to 224x224 to match the MobileNetV2 input layer.
- Augmentation: Random horizontal flips, 10% rotation, and 10% zoom are applied to the training set to prevent overfitting.
- Normalization: Pixel values are rescaled to the range [-1, 1] as required by the MobileNetV2 architecture.

3.5 Model Development
The system utilizes MobileNetV2 due to its efficiency and high accuracy on image classification tasks. The base model layers are frozen to retain pre-trained ImageNet weights. A GlobalAveragePooling2D layer and a Dropout layer (0.4) are added to reduce overfitting, followed by a Dense layer with a Softmax activation function to output probabilities for the 5 classes. 

To address dataset imbalance (where Class 0 has significantly more images than Class 4), dynamic Class Weights are computed and applied during the loss calculation, penalizing the model more heavily for misclassifying minority classes.

3.6 Evaluation Metrics
- Accuracy: Overall correctness of the model.
- Precision & Recall: To measure the model's performance on individual classes (especially minority classes).
- F1-Score: The harmonic mean of Precision and Recall.
- Confusion Matrix: Visualizes true positive vs false positive rates across all 5 classes.

---

CHAPTER 4
IMPLEMENTATION AND RESULTS

4.1 Implementation
The project is divided into distinct Python modules:
- preprocessing.py: Handles dataset loading and data augmentation pipelines using tf.data.
- train.py: Defines the MobileNetV2 architecture, compiles the model with Adam optimizer, and implements Early Stopping.
- evaluate.py: Evaluates the trained model on the test dataset and saves visualization charts.
- predict.py: A helper module to process a single image for live predictions.
- app/app.py: The frontend interface built with Streamlit.

4.2 Experimental Results
The model was trained for several epochs using the training split and validated against the validation split. Early Stopping ensured the model saved the best weights based on validation accuracy, preventing overfitting. 

4.3 Output / Screenshots
The Streamlit application features multiple tabs:
1. Home: Overview of the project and its medical context.
2. Dataset Overview: Bar charts showing class distribution and total sample metrics.
3. Model Performance: Displays the Training vs Validation Loss/Accuracy graphs and the Confusion Matrix.
4. Prediction Dashboard: Allows the user to upload a standard PNG/JPG knee X-ray. The AI processes the image and outputs the predicted severity class along with a confidence probability bar chart.

4.4 Discussion
By implementing Class Weights, the model successfully learns to identify severe cases (Class 3 and 4) despite them being underrepresented in the dataset. MobileNetV2 proves to be highly efficient, providing rapid predictions suitable for web deployment while maintaining high accuracy. The Streamlit interface effectively bridges the gap between the complex AI backend and the end-user (clinicians or researchers).

---

CHAPTER 5
CONCLUSION AND FUTURE ENHANCEMENT

5.1 Conclusion
The AI-Assisted Early Detection System for Osteoarthritis provides a robust, scalable, and user-friendly platform for analyzing Knee X-rays. By leveraging MobileNetV2 and handling data imbalance effectively, the system can reliably categorize OA severity based on the standard Kellgren-Lawrence scale. The intuitive web interface ensures that this technology can be easily accessed and utilized as a preliminary screening tool, particularly benefiting regions with limited access to specialized orthopaedic care, like the NER.

5.2 Future Enhancement
- Explainable AI (XAI): Implementing Grad-CAM to generate heatmaps that highlight the specific regions of the X-ray (e.g., joint space narrowing) that influenced the AI's decision.
- Multi-modal Data: Incorporating patient clinical data (age, weight, pain score) alongside the X-ray images for a more holistic prediction.
- Cloud Deployment: Deploying the application on cloud platforms (AWS/GCP) for widespread access by rural clinics.

---

CHAPTER 6
REFERENCES

[1] Kaggle, "Knee Osteoarthritis Dataset with Severity Scoring".
[2] TensorFlow Developers, "MobileNetV2 Documentation." Available: https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
[3] Streamlit Documentation. Available: https://docs.streamlit.io/
[4] Scikit-learn Developers, "Metrics and Scoring." Available: https://scikit-learn.org/stable/modules/model_evaluation.html
[5] Pandas Development Team, "Pandas Documentation." Available: https://pandas.pydata.org/docs/

---

