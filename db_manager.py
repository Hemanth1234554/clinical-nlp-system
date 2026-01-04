import sqlite3
from datetime import datetime
from collections import Counter

# Name of our database file
DB_NAME = "medical_summaries.db"

def init_db():
    """
    Creates the database and the table if they don't exist.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT,
            generated_summary TEXT,
            created_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_summary(text, summary):
    """
    Saves a new record into the database.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('INSERT INTO summaries (original_text, generated_summary, created_at) VALUES (?, ?, ?)',
              (text, summary, current_time))
    
    conn.commit()
    conn.close()

def get_all_summaries():
    """
    Retrieves all records from the database to show history.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM summaries ORDER BY created_at DESC')
    data = c.fetchall()
    conn.close()
    return data

def get_entity_stats():
    """
    Analyzes patient traffic for the dashboard.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT created_at FROM summaries')
    data = c.fetchall()
    conn.close()

    # Extract just the date part (YYYY-MM-DD)
    dates = [row[0].split(" ")[0] for row in data]
    return dict(Counter(dates))

# Initialize on import
if __name__ == "__main__":
    init_db()
    print("Database initialized.")