import sqlite3
import random
from datetime import datetime, timedelta

def seed():
    conn = sqlite3.connect('instance/oa_system.db')
    
    # Wait, flask-sqlalchemy in Flask 3+ creates db in 'instance/oa_system.db' by default 
    # Let's check where it is. If it's not there, it's in the root.
    import os
    db_path = 'oa_system.db'
    if not os.path.exists(db_path):
        db_path = 'instance/oa_system.db'
        
    if not os.path.exists(db_path):
        print("No database found. User hasn't created an account yet.")
        return
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    classes = [
        ('0: Normal', 0.95),
        ('1: Doubtful OA', 0.88),
        ('2: Minimal OA', 0.92),
        ('3: Moderate OA', 0.85),
        ('4: Severe OA', 0.98)
    ]
    
    print("Seeding database...")
    # Generate 45 realistic past predictions
    for i in range(45):
        cls, base_conf = random.choice(classes)
        conf = base_conf - random.uniform(0.01, 0.15)
        
        days_ago = random.randint(0, 10)
        minutes_ago = random.randint(0, 1440)
        dt = datetime.utcnow() - timedelta(days=days_ago, minutes=minutes_ago)
        
        filename = f"patient_{random.randint(10000, 99999)}_xray.png"
        
        c.execute("""
            INSERT INTO prediction (user_id, filename, predicted_class, confidence, created)
            VALUES (?, ?, ?, ?, ?)
        """, (1, filename, cls, conf, dt.strftime('%Y-%m-%d %H:%M:%S.%f')))
        
    conn.commit()
    conn.close()
    print("Successfully added 45 records!")

if __name__ == '__main__':
    seed()
