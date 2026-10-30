# AI-Assisted Early Detection System for Osteoarthritis (OA) Risk Markers

This project uses AI/ML to detect Osteoarthritis (OA) severity from knee X-ray images, using the Kellgren-Lawrence grading system (0-4). Built to satisfy Problem Statement ID 26004.

## Features
- Deep Learning model utilizing **MobileNetV2** (Transfer Learning)
- Handles Class Imbalance via dynamic Weighted Loss functions
- Full Data Pipeline: Preprocessing -> Augmentation -> Training -> Evaluation
- Interactive **Streamlit Dashboard** for dataset analytics and live model predictions.

## Project Structure
```
oa-risk-detection/
├── dataset/         # X-Ray datasets (Train/Val/Test)
├── src/             # Core ML source code
│   ├── preprocessing.py 
│   ├── train.py     
│   ├── evaluate.py  
│   └── predict.py   
├── models/          # Saved trained model artifacts (.keras)
├── results/         # Evaluation graphs and confusion matrices
├── app/             # Streamlit application UI
│   └── app.py       
└── requirements.txt # Python dependencies
```

## Data and model files

Medical images, uploaded scans, local databases, generated results, and trained model files are excluded from Git. Add the dataset locally under `dataset/` before training; `src/train.py` creates the model and evaluation outputs.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python src/train.py
   ```
   *This trains the CNN and saves the best model to the `models/` directory.*

3. **Evaluate the Model:**
   ```bash
   python src/evaluate.py
   ```
   *This evaluates the model on the test dataset and generates a confusion matrix and classification report in `results/`.*

4. **Launch the Web Interface:**
   ```bash
   streamlit run app/app.py
   ```
   *This opens the web dashboard where you can view data analytics and test the prediction UI with sample X-rays.*
