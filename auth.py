import sqlite3
import hashlib
from functools import wraps
from flask import session, redirect, url_for

DB_PATH = 'users.db'

def init_db():
    """Initialize database with default users"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    
    # Add default users if they don't exist
    try:
        hashed_admin = hashlib.sha256(b'admin').hexdigest()
        hashed_user = hashlib.sha256(b'pw').hexdigest()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('rey', hashed_admin))
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('user1', hashed_user))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

def verify_login(username, password):
    """Verify username and password"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hashed_pwd))
    result = c.fetchone()
    conn.close()
    return result is not None

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function
