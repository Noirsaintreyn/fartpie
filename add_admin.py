#!/usr/bin/env python3
"""
Script to add an admin account to the database
Usage: python add_admin.py <username> <email> <password>
"""

import sqlite3
import hashlib
import sys
import os

# Use the same DB path logic as backend.py
render_disk_path = os.getenv('RENDER_DISK_PATH')
if render_disk_path:
    os.makedirs(render_disk_path, exist_ok=True)
    DB_PATH = os.path.join(render_disk_path, 'users.db')
else:
    DB_PATH = 'users.db'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_admin(username, email, password):
    """Add an admin account to the database"""
    hashed_password = hash_password(password)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if user already exists
        c.execute("SELECT username FROM users WHERE username = ? OR email = ?", (username, email))
        existing = c.fetchone()
        if existing:
            print(f"❌ User '{username}' or email '{email}' already exists!")
            conn.close()
            return False
        
        # Insert admin account
        c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                  (username, email, hashed_password, 1))
        conn.commit()
        conn.close()
        
        print(f"✅ Admin account '{username}' added successfully!")
        print(f"   Email: {email}")
        print(f"   Password: {'*' * len(password)}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding admin account: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python add_admin.py <username> <email> <password>")
        print("\nExample:")
        print("  python add_admin.py myadmin admin@example.com mypassword123")
        sys.exit(1)
    
    username = sys.argv[1]
    email = sys.argv[2]
    password = sys.argv[3]
    
    print(f"Adding admin account...")
    print(f"Database path: {DB_PATH}")
    success = add_admin(username, email, password)
    
    if not success:
        sys.exit(1)








