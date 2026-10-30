import os
import random
import json
from shutil import copy2
from app import app, db
from models import Prediction, User
import ml

def seed_database():
    dataset_test_path = os.path.join(os.path.dirname(__file__), "dataset", "test")
    upload_folder = os.path.join(os.path.dirname(__file__), "static", "uploads")
    
    if not os.path.exists(dataset_test_path):
        print(f"Error: Dataset not found at {dataset_test_path}")
        return

    os.makedirs(upload_folder, exist_ok=True)
    
    with app.app_context():
        # Get doctor user
        doctor = User.query.filter_by(role="Doctor").first()
        if not doctor:
            # Fallback to any user
            doctor = User.query.first()
            if not doctor:
                print("Error: No users in DB. Run app and create a user first.")
                return

        print(f"Seeding for user: {doctor.name}")
        
        classes = os.listdir(dataset_test_path)
        count = 0
        
        for cls_name in classes:
            cls_path = os.path.join(dataset_test_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
                
            images = os.listdir(cls_path)
            # Pick up to 7 images per class
            selected = random.sample(images, min(7, len(images)))
            
            for img_name in selected:
                source_path = os.path.join(cls_path, img_name)
                dest_path = os.path.join(upload_folder, img_name)
                
                # Copy image to uploads
                copy2(source_path, dest_path)
                
                heatmap_name = f"heatmap_{img_name}"
                heatmap_path = os.path.join(upload_folder, heatmap_name)
                
                # Run AI Prediction
                pred_cls, conf, probs, is_gradcam = ml.predict_xray(dest_path, heatmap_path)
                
                if pred_cls is None:
                    print(f"Failed to predict {img_name}")
                    continue
                
                # Generate random plausible clinical data based on predicted class severity
                if "Severe" in pred_cls or "Moderate" in pred_cls:
                    age = random.randint(60, 85)
                    weight = random.uniform(75, 105)
                    pain = random.randint(6, 10)
                    gait = "Yes"
                    stiff = True
                else:
                    age = random.randint(45, 65)
                    weight = random.uniform(60, 85)
                    pain = random.randint(1, 5)
                    gait = "No"
                    stiff = False
                    
                height = random.uniform(155, 185)
                bmi = weight / ((height / 100) ** 2)
                notes = f"Auto-generated record for testing {cls_name} cases."
                
                p = Prediction(
                    user_id=doctor.id,
                    filename=img_name,
                    predicted_class=pred_cls,
                    confidence=conf,
                    age=age,
                    height=round(height, 1),
                    weight=round(weight, 1),
                    bmi=round(bmi, 1),
                    pain_level=pain,
                    gait_issues=gait,
                    stiffness=stiff,
                    notes=notes,
                    probabilities=json.dumps(probs),
                    is_demo=False # IMPORTANT: Show up in main dashboard
                )
                db.session.add(p)
                count += 1
                print(f"Added {img_name} -> {pred_cls}")
                
        db.session.commit()
        print(f"Successfully generated {count} test records using real predictions!")

if __name__ == "__main__":
    seed_database()
