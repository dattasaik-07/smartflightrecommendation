"""
upload_to_db.py
Database connection and initialization script for Airline Recommendation System.
Handles creating the MySQL database `airline_db` and `feedback` table,
with an automatic fallback to local SQLite if MySQL server is not currently running.
"""

import os
import sys

# MySQL Configuration (customize according to your local MySQL / XAMPP setup)
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', ''),
    'port': int(os.environ.get('MYSQL_PORT', 3306)),
    'database': 'airline_db'
}

def init_mysql():
    try:
        import mysql.connector
        from mysql.connector import errorcode
        
        print(f"Connecting to MySQL server at {DB_CONFIG['host']}:{DB_CONFIG['port']} with user '{DB_CONFIG['user']}'...")
        
        # Connect to MySQL Server (without selecting database yet)
        cnx = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port']
        )
        cursor = cnx.cursor()
        
        # Create database if not exists
        db_name = DB_CONFIG['database']
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        print(f"Database '{db_name}' verified/created successfully.")
        
        # Use the database
        cursor.execute(f"USE {db_name};")
        
        # Create feedback table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            source VARCHAR(50) NOT NULL,
            destination VARCHAR(50) NOT NULL,
            budget INT NOT NULL,
            travel_class VARCHAR(50) NOT NULL,
            recommended_airline VARCHAR(50) NOT NULL,
            rating INT NOT NULL,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(create_table_query)
        cnx.commit()
        print("Table 'feedback' verified/created successfully in MySQL.")
        
        # Insert a sample feedback record if table is empty
        cursor.execute("SELECT COUNT(*) FROM feedback;")
        count = cursor.fetchone()[0]
        if count == 0:
            sample_query = """
            INSERT INTO feedback (source, destination, budget, travel_class, recommended_airline, rating, comment)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cursor.execute(sample_query, ('Chennai', 'Delhi', 5000, 'Economy', 'IndiGo', 5, 'Great recommendation, very accurate!'))
            cnx.commit()
            print("Sample feedback record inserted into MySQL.")
            
        cursor.close()
        cnx.close()
        print("MySQL database setup completed successfully!")
        return True
        
    except Exception as e:
        print("\n" + "!"*60)
        print("MySQL Connection Notice:")
        print(f"Could not connect to MySQL server ({e}).")
        print("If you have MySQL or XAMPP installed, make sure the MySQL service is started.")
        print("Default connection parameters: host=localhost, user=root, password='', port=3306")
        print("!"*60 + "\n")
        return False

def init_sqlite_fallback():
    import sqlite3
    db_file = 'feedback.db'
    print(f"Setting up SQLite fallback database '{db_file}'...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        destination TEXT NOT NULL,
        budget INTEGER NOT NULL,
        travel_class TEXT NOT NULL,
        recommended_airline TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    
    # Check count
    cursor.execute("SELECT COUNT(*) FROM feedback;")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.execute("""
        INSERT INTO feedback (source, destination, budget, travel_class, recommended_airline, rating, comment)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, ('Chennai', 'Delhi', 5000, 'Economy', 'IndiGo', 5, 'Great recommendation, matches my preferences!'))
        conn.commit()
    conn.close()
    print(f"SQLite fallback database '{db_file}' ready with feedback table!")

if __name__ == '__main__':
    print("="*60)
    print("AIRLINE RECOMMENDATION SYSTEM - DATABASE SETUP")
    print("="*60)
    mysql_ok = init_mysql()
    if not mysql_ok:
        print("Initializing local fallback SQLite database so the app runs smoothly...")
        init_sqlite_fallback()
    print("Database initialization finished.")
