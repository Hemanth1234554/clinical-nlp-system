import sqlite3
from datetime import datetime, timedelta
import random

DB_NAME = "medical_summaries.db"

def add_fake_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    print("Injecting fake history...")
    
    # We will add data for the last 5 days
    for i in range(5):
        # Calculate a date in the past (e.g., 1 day ago, 2 days ago)
        date_in_past = datetime.now() - timedelta(days=i)
        date_str = date_in_past.strftime("%Y-%m-%d %H:%M:%S")
        
        # Randomly decide how many patients to add for that day (between 2 and 8)
        num_patients = random.randint(2, 8)
        
        for _ in range(num_patients):
            c.execute('INSERT INTO summaries (original_text, generated_summary, created_at) VALUES (?, ?, ?)',
                      ("Fake medical note for testing graph.", "Fake summary.", date_str))
            
    conn.commit()
    conn.close()
    print("Success! Fake data added. Now refresh your website.")

if __name__ == "__main__":
    add_fake_history()