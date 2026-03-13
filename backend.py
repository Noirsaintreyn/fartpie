from flask import Flask, jsonify, request, session
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import hdbscan
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    ripser = None
    plot_diagrams = None

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    GaussianHMM = None

from scipy.signal import find_peaks, savgol_filter, argrelextrema
from scipy.stats import norm, kurtosis, skew, gaussian_kde, kendalltau
from datetime import datetime, timedelta
import sqlite3
import json
import uuid
from typing import Optional, Dict, Any, List, Tuple

from theoretical_bounds import (
    TheoreticalBoundsPipeline,
    PipelineConfig,
    LivePredictor,
    LiveForecast,
    validate_ohlcv,
    add_mid_range,
)

# ============================================================================
# YFINANCE INTERVAL FIX - Handles all timeframes correctly (including 4h)
# ============================================================================

def get_valid_yfinance_interval(timeframe: str) -> str:
    """
    Convert user-friendly timeframe to valid yfinance interval
    
    yfinance valid intervals:
    - Minutes: 1m, 2m, 5m, 15m, 30m, 60m, 90m
    - Hours: 1h (only this one!)
    - Days: 1d, 5d, 1wk, 1mo, 3mo
    
    Common issues:
    - 4h is NOT valid → use 1h and resample
    - 2h, 3h, 6h are NOT valid → use 1h and resample
    """
    interval_map = {
        '1m': '1m',
        '2m': '2m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',     # Valid
        '60m': '60m',   # Alternative to 1h
        '2h': '1h',     # Download 1h, resample to 2h
        '4h': '1h',     # Download 1h, resample to 4h
        '6h': '1h',     # Download 1h, resample to 6h
        '1d': '1d',
        '1wk': '1wk',
        '1mo': '1mo'
    }
    return interval_map.get(timeframe, '1d')

def needs_resampling(timeframe: str) -> bool:
    """Check if this timeframe requires resampling"""
    return timeframe in ['2h', '4h', '6h', '8h', '12h']

def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample 1h data to higher timeframes (2h, 4h, etc.)
    
    Properly aggregates OHLCV:
    - Open: first value
    - High: maximum value
    - Low: minimum value
    - Close: last value
    - Volume: sum
    """
    if df.empty:
        return df
    
    timeframe_map = {
        '2h': '2H',
        '4h': '4H',
        '6h': '6H',
        '8h': '8H',
        '12h': '12H'
    }
    
    resample_rule = timeframe_map.get(target_timeframe)
    if not resample_rule:
        return df
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Resample with proper aggregation
    resampled = df.resample(resample_rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled

def fetch_historical_data_with_resampling(
    ticker: str, 
    timeframe: str, 
    period: str = None,
    start_date: str = None,
    end_date: str = None,
    is_futures: bool = False
) -> pd.DataFrame:
    """
    Fetch historical data with proper interval handling and resampling
    
    This function handles ALL timeframes correctly, including 4h
    """
    stock = yf.Ticker(ticker)
    
    # Get valid yfinance interval
    yf_interval = get_valid_yfinance_interval(timeframe)
    
    # For futures, prefer minute-based intervals
    if is_futures and timeframe == '1h':
        yf_interval = '60m'  # Futures prefer 60m over 1h
    elif is_futures and timeframe == '4h':
        yf_interval = '60m'  # Will resample from 60m to 4h
    
    # Adjust period if needed (4h needs more data to resample properly)
    if period and needs_resampling(timeframe):
        # Multiply period to get enough data
        period_multiplier = {
            '2h': 2,
            '4h': 4,
            '6h': 6,
            '8h': 8,
            '12h': 12
        }
        mult = period_multiplier.get(timeframe, 1)
        
        # Adjust period string
        if period.endswith('d'):
            days = int(period[:-1])
            adjusted_days = min(days * mult, 730)  # Cap at yfinance limit
            period = f"{adjusted_days}d"
        elif period.endswith('mo'):
            months = int(period[:-2])
            adjusted_months = min(months * mult, 24)  # Cap at 2 years
            period = f"{adjusted_months}mo"
    
    # Fetch data with retry logic
    hist = None
    attempts = []
    
    if is_futures and timeframe in ['1h', '4h']:
        # For futures 1h/4h, try multiple combinations
        if timeframe == '1h':
            attempts = [
                ('60m', period if period else '5d'),
                ('60m', '5d'), ('60m', '3d'), ('60m', '2d'), ('60m', '1d'),
                ('1h', '5d'), ('1h', '3d'), ('1h', '2d'), ('1h', '1d'),
            ]
        else:  # 4h
            attempts = [
                ('60m', period if period else '10d'),
                ('60m', '10d'), ('60m', '7d'), ('60m', '5d'), ('60m', '3d'), ('60m', '2d'), ('60m', '1d'),
                ('1h', '10d'), ('1h', '7d'), ('1h', '5d'), ('1h', '3d'), ('1h', '2d'), ('1h', '1d'),
            ]
    else:
        attempts = [(yf_interval, period if period else '1mo')]
    
    for attempt_interval, attempt_period in attempts:
        try:
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval=attempt_interval)
            else:
                hist = stock.history(period=attempt_period, interval=attempt_interval)
            
            if hist is not None and len(hist) > 0:
                yf_interval = attempt_interval  # Update for resampling
                print(f"✓ Fetched {len(hist)} bars for {ticker} @ {timeframe} using interval={attempt_interval}, period={attempt_period}")
                break
        except Exception as e:
            error_msg = str(e)
            if "pattern" not in error_msg.lower() and "expected" not in error_msg.lower():
                print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:100]}")
            continue
    
    if hist is None or len(hist) == 0:
        raise ValueError(
            f"Could not fetch data for {ticker} at {timeframe}. "
            f"yfinance may not support this combination."
        )
    
    # Resample if needed
    if needs_resampling(timeframe) and not hist.empty:
        print(f"Resampling {yf_interval} → {timeframe}...")
        hist = resample_ohlcv(hist, timeframe)
        print(f"✓ Resampled to {len(hist)} bars")
    
    return hist
import hashlib
import warnings
import requests
import os
import pickle
from arch import arch_model
warnings.filterwarnings('ignore')

# Custom JSON encoder to handle numpy/pandas types
try:
    from flask.json.provider import DefaultJSONProvider
    class NumpyJSONProvider(DefaultJSONProvider):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif pd.isna(obj):
                return None
            return super().default(obj)
    
    app = Flask(__name__)
    app.json = NumpyJSONProvider(app)
except (ImportError, AttributeError):
    # Fallback for older Flask versions
    from flask.json import JSONEncoder
    class NumpyJSONEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif pd.isna(obj):
                return None
            return super().default(obj)
    
    app = Flask(__name__)
    app.json_encoder = NumpyJSONEncoder

app.secret_key = "degen-discovery-secret-key-2024"

# Session cookie configuration - different for production vs development
IS_PROD = os.getenv("ENV") == "production" or os.getenv("FLASK_ENV") == "production"

# session cookies for cross-domain login
# Secure cookies (HTTPS only) + SameSite=None required for cross-origin in production
# In development (localhost), use Lax + non-secure for local testing
if IS_PROD:
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True
else:
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = False

# CORS configuration - browsers require explicit origins when using credentials
# Allow common frontend domains (add your production domain here)
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://degencap.uk,https://www.degencap.uk').split(',')
CORS(app, supports_credentials=True, origins=CORS_ORIGINS)

@app.route("/")
def health():
    return {"status": "backend live"}
  
FRED_API_KEY = '024452292701539abb68abc50276eb70'

# Simple password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed):
    return hashlib.sha256(password.encode()).hexdigest() == hashed

# Initialize database
# Use persistent path for Render (persistent disk)
render_disk_path = os.getenv('RENDER_DISK_PATH')
if render_disk_path:
    # Ensure directory exists
    os.makedirs(render_disk_path, exist_ok=True)
    DB_PATH = os.path.join(render_disk_path, 'users.db')
else:
    # Fallback to current directory
    DB_PATH = 'users.db'

def init_db():
    """Initialize database - creates table and admin user if needed"""
    global DB_PATH
    try:
        # Use context manager to ensure connection is closed even on error
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          email TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL,
                          is_active INTEGER DEFAULT 1,
                          is_admin INTEGER DEFAULT 0,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          last_login TIMESTAMP)''')
            conn.commit()
            
            # Admin account: rey / flood
            rey_password = hash_password('flood')
            try:
                c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                          ('rey', 'rey@degendiscovery.com', rey_password, 1))
                conn.commit()
                print("✓ Admin account (rey) created")
            except sqlite3.IntegrityError:
                # Update existing admin password if it exists
                c.execute("UPDATE users SET password = ? WHERE username = 'rey' AND is_admin = 1", (rey_password,))
                conn.commit()
                print("✓ Admin account (rey) already exists, password updated")
            
            # Remove old admin account if it exists
            c.execute("DELETE FROM users WHERE username = 'admin' AND is_admin = 1")
            conn.commit()
            
            # Test account 1: test1 / pw
            test1_password = hash_password('pw')
            try:
                c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                          ('test1', 'test1@degendiscovery.com', test1_password, 0))
                conn.commit()
                print("✓ Test account 1 (test1) created")
            except sqlite3.IntegrityError:
                c.execute("UPDATE users SET password = ? WHERE username = 'test1'", (test1_password,))
                conn.commit()
                print("✓ Test account 1 (test1) already exists, password updated")
            
            # Test account 2: test2 / pw
            test2_password = hash_password('pw')
            try:
                c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                          ('test2', 'test2@degendiscovery.com', test2_password, 0))
                conn.commit()
                print("✓ Test account 2 (test2) created")
            except sqlite3.IntegrityError:
                c.execute("UPDATE users SET password = ? WHERE username = 'test2'", (test2_password,))
                conn.commit()
                print("✓ Test account 2 (test2) already exists, password updated")
        # Connection auto-closes here via context manager
        
        conn.close()
        print(f"✓ Database initialized at: {DB_PATH}")
    except Exception as e:
        print(f"⚠ Database initialization error: {e}")
        # Try fallback to current directory
        if DB_PATH != 'users.db':
            DB_PATH = 'users.db'
            init_db()

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'success': True, 'message': 'Backend is running!'})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    hashed_password = hash_password(password)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                  (username, email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'User registered successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username or email already exists'}), 400

@app.route('/api/login', methods=['POST'])
def login():
    global DB_PATH
    print(f"Login attempt received: {request.json}")
    
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Missing credentials'}), 400
    
    try:
        # Ensure database exists before trying to connect
        try:
            conn = sqlite3.connect(DB_PATH)
        except Exception as db_error:
            print(f"⚠ Database connection failed, trying to initialize: {db_error}")
            # Try to initialize database if connection fails
            try:
                init_db()
                conn = sqlite3.connect(DB_PATH)
            except Exception as init_error:
                print(f"⚠ Database initialization failed: {init_error}")
                # Fallback to users.db
                if DB_PATH != 'users.db':
                    DB_PATH = 'users.db'
                    try:
                        init_db()
                        conn = sqlite3.connect(DB_PATH)
                    except Exception as e:
                        return jsonify({'success': False, 'error': 'Database initialization failed. Please contact support.'}), 500
        
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_active, is_admin FROM users WHERE username = ?", (username,))
        user_data = c.fetchone()
        
        print(f"User data found: {user_data is not None}")
        
        if not user_data:
            conn.close()
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        user_id, db_username, db_password, is_active, is_admin = user_data
        
        if not is_active:
            conn.close()
            return jsonify({'success': False, 'error': 'Account disabled'}), 403
        
        if not check_password(password, db_password):
            conn.close()
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        c.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id))
        conn.commit()
        conn.close()
        
        session['user_id'] = user_id
        session['username'] = db_username
        session['is_admin'] = is_admin
        
        print(f"Login successful for user: {db_username}")
        
        return jsonify({
            'success': True, 
            'message': 'Login successful',
            'user': db_username,
            'is_admin': bool(is_admin)
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/login: {error_trace}")
        return jsonify({'success': False, 'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_active FROM users WHERE id = ?", (session['user_id'],))
        result = c.fetchone()
        conn.close()
        
        if result and result[0]:
            return jsonify({
                'authenticated': True,
                'username': session.get('username'),
                'is_admin': bool(session.get('is_admin', False))
            })
    return jsonify({'authenticated': False}), 401

@app.route('/api/admin/users', methods=['GET'])
def get_users():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, email, is_active, created_at, last_login FROM users WHERE is_admin = 0")
    users = c.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        user_list.append({
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'is_active': user[3],
            'created_at': user[4],
            'last_login': user[5]
        })
    
    return jsonify({'success': True, 'users': user_list})

@app.route('/api/admin/disable-user', methods=['POST'])
def disable_user():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    data = request.json
    user_id = data.get('user_id')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET is_active = 0 WHERE id = ? AND is_admin = 0", (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User disabled'})

@app.route('/api/admin/enable-user', methods=['POST'])
def enable_user():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    data = request.json
    user_id = data.get('user_id')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET is_active = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User enabled'})

@app.route('/api/admin/create-admin', methods=['POST'])
def create_admin():
    """Create a new admin account - requires existing admin authentication"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    hashed_password = hash_password(password)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                  (username, email, hashed_password, 1))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'Admin account "{username}" created successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username or email already exists'}), 400

@app.route('/api/admin/promote-user', methods=['POST'])
def promote_user():
    """Promote an existing user to admin - requires existing admin authentication"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'Missing user_id'}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Check if user exists
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        if not user:
            conn.close()
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Promote to admin
        c.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'User "{user[0]}" promoted to admin'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/admin/add-admin', methods=['POST'])
def add_admin():
    """Manually add an admin account - for initial setup (no auth required)"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    hashed_password = hash_password(password)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                  (username, email, hashed_password, 1))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'Admin account "{username}" added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username or email already exists'}), 400

def sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to Python native types for JSON serialization"""
    import numpy as np
    import pandas as pd
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy/pandas boolean types (check before other numpy types)
    if isinstance(obj, np.bool_) or (hasattr(np, 'bool_') and type(obj).__name__ == 'bool_'):
        return bool(obj)
    # Handle Python bool (keep as is, but ensure it's a bool)
    elif isinstance(obj, bool):
        return bool(obj)
    # Handle numpy integers
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    # Handle numpy floats
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(item) for item in obj.tolist()]
    # Handle pandas Series
    elif isinstance(obj, pd.Series):
        return [sanitize_for_json(item) for item in obj.tolist()]
    # Handle pandas DataFrame
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    # Handle dictionaries
    elif isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    # Handle pandas NaN
    elif pd.isna(obj):
        return None
    # Handle other types - try to convert if it's a numpy scalar
    elif hasattr(obj, 'item'):  # numpy scalars have .item() method
        try:
            return sanitize_for_json(obj.item())
        except:
            return str(obj)
    else:
        return obj

def require_auth():
    global DB_PATH
    if 'user_id' not in session:
        return {'error': 'Not authenticated', 'code': 401}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_active FROM users WHERE id = ?", (session['user_id'],))
        result = c.fetchone()
        conn.close()
        
        if not result or not result[0]:
            session.clear()
            return {'error': 'Account disabled', 'code': 403}
        return None
    except Exception as e:
        print(f"⚠ Database error in require_auth: {e}")
        # Try to initialize database
        try:
            init_db()
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT is_active FROM users WHERE id = ?", (session['user_id'],))
            result = c.fetchone()
            conn.close()
            
            if not result or not result[0]:
                session.clear()
                return {'error': 'Account disabled', 'code': 403}
            return None
        except Exception as e2:
            print(f"⚠ Database initialization failed in require_auth: {e2}")
            session.clear()
            return {'error': 'Database error. Please try again.', 'code': 500}

# ============================================================================
# MARKET MICROSTRUCTURE - PHASE SPACE & STATE DETECTION
# ============================================================================

def calculate_phase_space_coordinates(closes, volumes):
    """
    Calculate 3D phase space coordinates:
    - X: Price position (normalized)
    - Y: Market velocity (price momentum)
    - Z: Volume momentum
    """
    if len(closes) < 5:
        return None
    
    # Normalize price position
    price_range = np.max(closes) - np.min(closes)
    if price_range == 0:
        price_position = np.zeros_like(closes)
    else:
        price_position = (closes - np.min(closes)) / price_range * 100
    
    # Market velocity (rate of change)
    velocity = np.gradient(closes)
    
    # Volume momentum (normalized)
    if len(volumes) > 0:
        vol_ma = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
        volume_momentum = (volumes - vol_ma) / (vol_ma + 1)
    else:
        volume_momentum = np.zeros_like(closes)
    
    return {
        'price_position': price_position.tolist(),
        'velocity': velocity.tolist(),
        'volume_momentum': volume_momentum.tolist()
    }

# ============================================================================
# LIQUIDITY STRESS SCORING
# ============================================================================

def volume_zscore(volumes, lookback=50):
    """Compute volume z-score to detect abnormal volume conditions"""
    vols = np.asarray(volumes, dtype=float)
    if len(vols) < lookback + 1:
        return 0.0
    v = vols[-1]
    mu = np.mean(vols[-lookback:])
    sd = np.std(vols[-lookback:]) + 1e-9
    return float((v - mu) / sd)

def _zscore(x, eps=1e-9):
    """Compute z-score normalization"""
    x = np.asarray(x)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd + eps)

def _safe_log(x, eps=1e-12):
    """Safe logarithm with minimum value"""
    return np.log(np.maximum(x, eps))

def roll_effective_spread_proxy(closes, window=50):
    """
    Roll (1984) effective spread proxy using serial covariance of price changes.
    No bid/ask required. Can be noisy; use as optional component.
    """
    closes = np.asarray(closes, dtype=float)
    if len(closes) < window + 2:
        return np.nan

    p = closes[-(window+2):]
    dp = np.diff(p)  # price changes
    # covariance of successive price changes
    cov = np.cov(dp[1:], dp[:-1], bias=True)[0, 1]
    # Roll spread estimate: 2*sqrt(-cov), only if cov negative
    if cov < 0:
        return 2.0 * np.sqrt(-cov)
    return 0.0  # if cov not negative, proxy says "no spread signal"

def liquidity_stress_score(
    opens, highs, lows, closes, volumes,
    window=50,
    jump_sigma=3.0
):
    """
    Returns:
      lss: [0,1] (higher = more illiquid/stress)
      feats: dict of raw + normalized components for logging/RF
    Uses ONLY OHLCV.
    """
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows  = np.asarray(lows, dtype=float)
    closes= np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    n = len(closes)
    if n < window + 2:
        return 0.0, {"lss": 0.0, "note": "insufficient_history"}

    # Slice recent window
    o = opens[-window:]
    h = highs[-window:]
    l = lows[-window:]
    c = closes[-window:]
    v = volumes[-window:]

    # Returns (log)
    r = np.diff(_safe_log(c))
    r_abs = np.abs(r)
    r_std = np.nanstd(r) + 1e-12

    # Dollar volume proxy (price * volume). For futures, volume is contracts; still works as "activity".
    dv = c[1:] * v[1:]

    # 1) Amihud illiquidity: |r| / dollar_volume
    amihud = r_abs / (dv + 1e-9)
    amihud_z = _zscore(amihud)

    # 2) Volume drought: low volume increases stress
    vol_z = _zscore(v)
    vol_drought = np.clip(-vol_z, 0, None)  # only penalize low volume

    # 3) Jump intensity: fraction of returns > jump_sigma * std
    jumps = r_abs > (jump_sigma * r_std)
    jump_intensity = np.mean(jumps.astype(float))

    # 4) Wickiness / inefficiency: large range vs body (thin liquidity tends to print wicks)
    body = np.abs(c - o) + 1e-9
    hl_range = (h - l) + 1e-9
    wickiness = np.clip((hl_range / body) - 1.0, 0, None)  # 0 means body ~ range; higher = wickier
    wickiness_z = _zscore(wickiness)

    # 5) Roll spread proxy (optional)
    roll_spread = roll_effective_spread_proxy(closes, window=window)
    # Normalize roll spread by price to make scale-free
    roll_spread_pct = (roll_spread / (closes[-1] + 1e-9)) if np.isfinite(roll_spread) else 0.0

    # Normalize components into [0,1] with soft squashing
    def squash(x):
        # map z-like to 0..1
        return 1.0 / (1.0 + np.exp(-x))

    amihud_score = float(squash(np.nanmean(amihud_z)))
    vol_score    = float(squash(np.nanmean(vol_drought)))
    wick_score   = float(squash(np.nanmean(wickiness_z)))
    jump_score   = float(np.clip(jump_intensity * 5.0, 0.0, 1.0))  # 0.2 jump intensity -> 1.0

    # roll spread pct is usually tiny; scale it
    roll_score   = float(np.clip(roll_spread_pct * 200.0, 0.0, 1.0))  # tune factor as needed

    # Weighted combination (conservative)
    # Amihud + volume drought are the most reliable with OHLCV
    lss = (
        0.35 * amihud_score +
        0.25 * vol_score +
        0.20 * jump_score +
        0.15 * wick_score +
        0.05 * roll_score
    )
    lss = float(np.clip(lss, 0.0, 1.0))

    feats = {
        "lss": lss,
        "amihud_score": amihud_score,
        "vol_drought_score": vol_score,
        "jump_intensity": float(jump_intensity),
        "wickiness_score": wick_score,
        "roll_spread_pct": float(roll_spread_pct),
        "roll_score": roll_score,
    }
    return lss, feats

def apply_liquidity_stress_to_microstructure(micro, lss):
    """
    micro: dict returned by detect_market_microstructure_state
    lss: 0..1
    """
    micro = dict(micro)  # copy

    state = micro.get("state", "Unknown")
    conf  = float(micro.get("confidence", 0.0))

  
    permeability = float(micro.get("liquidity_permeability", 0.5))
    overshoot    = float(micro.get("overshoot_bias", 0.2))

    # Nudge rules
    permeability = np.clip(permeability + 0.25 * lss, 0.0, 1.0)
    overshoot    = np.clip(overshoot + 0.20 * lss, 0.0, 0.8)


    if state == "Thermal":
        conf = np.clip(conf - 0.35 * lss, 0.0, 0.95)
    elif state == "Fock":
        conf = np.clip(conf + 0.15 * lss, 0.0, 0.95)
    else:  # Coherent or Unknown
        conf = np.clip(conf - 0.10 * lss, 0.0, 0.95)

    micro["confidence"] = float(conf)
    micro["liquidity_permeability"] = float(permeability)
    micro["overshoot_bias"] = float(overshoot)

    # Add into characteristics for logging/RF
    ch = dict(micro.get("characteristics", {}))
    ch["liquidity_stress"] = float(lss)
    micro["characteristics"] = ch

    return micro



def _iso(ts):
    try:
        if isinstance(ts, pd.Timestamp):
            return ts.isoformat()
        return str(ts)
    except Exception:
        return str(ts)

def _session_date_from_timestamp(ts) -> str:
    # For intraday: session is the calendar date in exchange timezone.
    # You can refine later for futures globex.
    dt = pd.to_datetime(ts)
    return dt.date().isoformat()











def detect_market_microstructure_state(closes, volumes, returns, highs=None, lows=None):
    """
    Detect market microstructure state - FIXED
    
    Fock: Jump-dominated, fat tails
    Thermal: Diffusive, normal-ish distribution
    Coherent: Directional, structural
    """
    if len(returns) < 50:
        return {
            'state': 'Unknown',
            'confidence': 0.0,
            'characteristics': {},
            'overshoot_bias': 0.2,
            'liquidity_permeability': 0.5,
            'capture_rate': 0.5,
            'level_multipliers': {'strength': 1.0, 'breakout_prob': 1.0}
        }
    
    # ===== CORE FEATURES =====
    
    kurt = kurtosis(returns)
    skewness = skew(returns)
    vol = np.std(returns)
    
    # Jump detection: 3-sigma outliers
    abs_returns = np.abs(returns)
    jump_threshold = 3 * vol
    jumps = abs_returns > jump_threshold
    
    # Jump dominance (variance explained by jumps)
    jump_count = np.sum(jumps)
    # jump_ratio removed - redundant and misleading (frequency is poor signal compared to jump energy)
    
    if jump_count > 0:
        jump_variance = np.sum(abs_returns[jumps] ** 2)
        total_variance = np.sum(returns ** 2)
        jump_dominance = jump_variance / (total_variance + 1e-9)
        jump_score = np.mean(abs_returns[jumps]) / (vol + 1e-9)
    else:
        jump_dominance = 0
        jump_score = 0
    
    # REMOVED: price_range_pct (leaks volatility)
    # Instead: Use velocity variance for microstructure signal only
    # FIXED: Use safer normalization to prevent explosion when velocity is tiny
    velocity = np.gradient(closes)
    velocity_var_normalized = np.var(velocity) / (np.var(closes[-50:]) + 1e-9)
    
    # FIXED: Trend strength with safe denominator
    recent_displacement = abs(closes[-1] - closes[-50])
    total_path_length = np.sum(np.abs(np.diff(closes[-50:])))
    trend_strength = recent_displacement / max(total_path_length, recent_displacement * 1.1)
    # Ensures denominator >= displacement, so trend_strength <= ~0.91
    
    # ===== CORRECTED CLASSIFICATION =====
    
    # FOCK: Jump-dominated with fat tails
    # RESTORED: Original thresholds (kurt > 8, jump_dominance > 0.3)
    if (kurt > 8 and jump_dominance > 0.30) or (jump_dominance > 0.45):
        state = "Fock"
        confidence = min(0.5 + jump_dominance * 0.8 + (kurt - 8) * 0.03, 0.95)
    
    # THERMAL: Diffusive, low jump dominance, near-normal kurtosis
    # RESTORED: kurt upper bound to 10 (was lowered to 7)
    # REMOVED: price_range_pct condition (leaked volatility)
    elif jump_dominance < 0.15 and 2 < kurt < 10:
        state = "Thermal"
        normality = 1 / (1 + abs(kurt - 3))
        confidence = min(0.5 + (1 - jump_dominance) * 0.3 + normality * 0.2, 0.95)
    
    # COHERENT: Directional, structural
    elif trend_strength > 0.20 or (jump_dominance < 0.20 and 2.5 < kurt < 6):
        state = "Coherent"
        confidence = min(0.5 + trend_strength * 1.5, 0.95)
    
    # DEFAULT: Tiebreaker using jump dominance
    else:
        if jump_dominance > 0.20:
            state = "Fock"
            confidence = 0.4 + jump_dominance * 0.6
        elif velocity_var_normalized < 0.5:
            # Smooth velocity = likely Thermal
            state = "Thermal"
            confidence = 0.45
        else:
            state = "Coherent"
            confidence = 0.5
    
    # ===== CHARACTERISTICS =====
    
    characteristics = {
        'kurtosis': float(kurt),
        'skewness': float(skewness),
        'volatility': float(vol),
        'jump_score': float(jump_score),
        'jump_dominance': float(jump_dominance),
        'velocity_variance': float(velocity_var_normalized),
        'trend_strength': float(trend_strength)
    }
    
    # NEW: Add liquidity stress if OHLC data available
    lss = 0.0
    lss_features = {}
    if highs is not None and lows is not None and len(closes) > 50:
        try:
            # Use closes as opens approximation
            # Pass full arrays - function will slice internally based on window parameter
            opens_approx = closes
            lss, lss_features = liquidity_stress_score(
                opens_approx, highs, lows, closes, volumes,
                window=50, jump_sigma=3.0
            )
        except Exception as e:
            print(f"⚠ Liquidity stress calculation failed: {e}")
            lss = 0.0
            lss_features = {}
    
    # ===== STATE-SPECIFIC PARAMETERS =====
    
    if state == 'Fock':
        overshoot_bias = min(0.3 + jump_dominance * 0.5, 0.6)
        liquidity_permeability = 0.60 + jump_dominance * 0.3
        capture_rate = 0.45
        level_multipliers = {
            'strength': 0.85,
            'breakout_prob': 1.3
        }
        
    elif state == 'Thermal':
        overshoot_bias = 0.1
        liquidity_permeability = 0.35
        capture_rate = 0.60
        level_multipliers = {
            'strength': 1.0,
            'breakout_prob': 1.0
        }
        
    else:  # Coherent
        overshoot_bias = 0.25
        liquidity_permeability = 0.50
        capture_rate = 0.8711
        level_multipliers = {
            'strength': 1.15,
            'breakout_prob': 0.7
        }
    
    result = {
        'state': state,
        'confidence': float(confidence),
        'characteristics': characteristics,
        'overshoot_bias': float(overshoot_bias),
        'liquidity_permeability': float(liquidity_permeability),
        'capture_rate': float(capture_rate),
        'level_multipliers': level_multipliers
    }
    
    # NOTE: LSS adjustments are applied via microstructure state, not here
    if lss_features:
        result['liquidity_stress_features'] = lss_features
    result['lss'] = float(lss)  # Store LSS value for later use
    
    return result

# ============================================================================
# GARCH VOLATILITY MODELING - ENHANCED
# ============================================================================

def fit_garch_model(returns, p=1, q=1):
    """
    Fit GARCH(p,q) model to return series
    
    Parameters:
    -----------
    returns : array-like
        Log returns (should be in percentage form)
    p : int
        GARCH lag order (default: 1)
    q : int
        ARCH lag order (default: 1)
    
    Returns:
    --------
    dict : Contains GARCH parameters, conditional volatility, and forecasts
    """
    try:
        if len(returns) < 50:
            return None
        
        # Fit GARCH model
        model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        params = result.params
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        
        # Calculate persistence
        persistence = alpha + beta
        
        # Get conditional volatility
        cond_vol = result.conditional_volatility
        # Handle both pandas Series and numpy array
        if hasattr(cond_vol, 'iloc'):
            current_vol = float(cond_vol.iloc[-1])
        else:
            # If it's a numpy array, use indexing
            current_vol = float(cond_vol[-1])
        
        # Forecast volatility (10 days ahead)
        forecasts = result.forecast(horizon=10)
        forecast_variance = forecasts.variance.values[-1, :]
        forecast_vol = np.sqrt(forecast_variance)
        
        # Calculate long-run volatility
        if persistence < 1:
            long_run_vol = np.sqrt(omega / (1 - persistence))
            half_life = np.log(0.5) / np.log(persistence) if persistence > 0 else 999
        else:
            long_run_vol = current_vol
            half_life = 999
        
        return {
            'omega': float(omega),
            'alpha': float(alpha),
            'beta': float(beta),
            'persistence': float(persistence),
            'current_vol': float(current_vol),
            'long_run_vol': float(long_run_vol),
            'conditional_volatility': cond_vol.tolist(),
            'forecast_vol': forecast_vol.tolist(),
            'is_stationary': bool(persistence < 1),
            'half_life': float(half_life)
        }
        
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        return None


def calculate_garch_volatility_regime(closes):
    """
    Enhanced volatility regime detection using GARCH
    
    Parameters:
    -----------
    closes : array-like
        Price series
    
    Returns:
    --------
    dict : Enhanced volatility regime information
    """
    # Calculate returns (in percentage)
    returns = np.log(closes[1:] / closes[:-1]) * 100
    
    # Fit GARCH model
    garch_results = fit_garch_model(returns)
    
    if garch_results is None:
        # Fallback to simple calculation if GARCH fails
        vol = np.std(returns) * np.sqrt(252)
        return {
            'regime': 'Normal Vol',
            'regime_factor': 1.0,
            'current_vol': float(vol),
            'long_run_vol': float(vol),
            'vol_ratio': 1.0,
            'vol_trend': 'Stable',
            'forecast_vol_5d': float(vol),
            'garch_params': None,
            'is_stationary': bool(True)
        }
    
    current_vol = garch_results['current_vol']
    long_run_vol = garch_results['long_run_vol']
    forecast_vol = garch_results['forecast_vol']
    
    # Calculate vol ratio (current vs long-run)
    vol_ratio = current_vol / long_run_vol if long_run_vol > 0 else 1.0
    
    # Determine regime based on GARCH parameters and current vol
    if vol_ratio > 1.5:
        regime = "Extreme Vol Spike"
        regime_factor = 1.8
    elif vol_ratio > 1.3:
        regime = "High Vol Spike"
        regime_factor = 1.5
    elif vol_ratio > 1.1:
        regime = "Elevated Vol"
        regime_factor = 1.2
    elif vol_ratio < 0.7:
        regime = "Extreme Vol Compression"
        regime_factor = 0.6
    elif vol_ratio < 0.85:
        regime = "Low Vol Compression"
        regime_factor = 0.75
    else:
        regime = "Normal Vol"
        regime_factor = 1.0
    
    # Calculate expected vol change (forward-looking)
    avg_forecast_vol = np.mean(forecast_vol[:5])  # Next 5 days
    vol_trend = "Increasing" if avg_forecast_vol > current_vol * 1.05 else \
                "Decreasing" if avg_forecast_vol < current_vol * 0.95 else \
                "Stable"
    
    return {
        'regime': regime,
        'regime_factor': regime_factor,
        'current_vol': float(current_vol),
        'long_run_vol': float(long_run_vol),
        'vol_ratio': float(vol_ratio),
        'vol_trend': vol_trend,
        'forecast_vol_5d': float(avg_forecast_vol),
        'forecast_vol_array': [float(v) for v in forecast_vol],
        'garch_params': {
            'omega': garch_results['omega'],
            'alpha': garch_results['alpha'],
            'beta': garch_results['beta'],
            'persistence': garch_results['persistence'],
            'half_life': garch_results['half_life']
        },
        'is_stationary': bool(garch_results['is_stationary'])
    }


def enhance_levels_with_microstructure(levels, closes, volumes, current_price, garch_vol_regime, microstructure_state, sigma_price=None):
    """
    ENHANCED: Uses GARCH + Market Microstructure State for superior level predictions
    
    FIXED: Uses sigma-normalized distance instead of price percentage for scale invariance
    FIXED: GARCH only affects confidence/breakout probability, NOT level strength
    """
    # Add safety check at the start
    if not levels or len(levels) == 0:
        return [], detect_market_regime_hmm(closes), calculate_hurst_exponent(closes), garch_vol_regime, microstructure_state
    
    # Get existing regime data
    hmm_regime = detect_market_regime_hmm(closes)
    hurst_data = calculate_hurst_exponent(closes)

    # Extract GARCH factors (for confidence/breakout only, NOT strength)
    # FIXED: Use .get() with safe defaults to prevent KeyError
    vol_ratio = garch_vol_regime.get('vol_ratio', 1.0)
    regime_factor = garch_vol_regime.get('regime_factor', 1.0)
    vol_trend = garch_vol_regime.get('vol_trend', 'Stable')
    
    garch_params = garch_vol_regime.get('garch_params')
    if garch_params is not None:
        persistence = garch_params.get('persistence', 0.85)
    else:
        persistence = 0.85
    
    # Extract microstructure factors
    market_state = microstructure_state['state']
    overshoot_bias = microstructure_state['overshoot_bias']
    liquidity_permeability = microstructure_state['liquidity_permeability']
    capture_rate = microstructure_state['capture_rate']
    state_multipliers = microstructure_state['level_multipliers']
    
    # Calculate sigma_price if not provided (fallback)
    if sigma_price is None or sigma_price <= 0:
        # Fallback: use recent volatility
        returns = np.log(closes[1:] / closes[:-1]) if len(closes) > 1 else np.array([0.01])
        sigma_price = float(np.std(returns) * current_price) if current_price > 0 else current_price * 0.02
    
    # MEANSHIFT VALIDATION: Run once for all HDBSCAN levels (validator, not producer)
    meanshift_validator_levels = []
    hdbscan_levels = [l for l in levels if l.get('category') == 'Density (HDBSCAN)' or l.get('category') == 'HDBSCAN']
    if len(hdbscan_levels) > 0 and len(closes) > 50:
        try:
            meanshift_validator_levels = calculate_meanshift_levels(highs, lows, closes)
        except Exception:
            pass  # If MeanShift fails, skip validation
    
    for level in levels:
        original_strength = level.get('strength', 0.5)
        
        # FIXED: Use sigma-normalized distance instead of price percentage
        # This ensures scale invariance, session alignment, and regime stability
        distance_sigma = abs(level['price'] - current_price) / sigma_price if sigma_price > 0 else float('inf')
        
        # Keep distance_pct for metadata only (not used in calculations)
        distance_pct = abs(level['price'] - current_price) / current_price if current_price > 0 else 0
        
        # ===== MICROSTRUCTURE-ENHANCED ADJUSTMENTS =====
        
        # 1. Market State Adjustment (NEW!)
        if market_state == 'Fock':
            # Levels are more permeable, prices overshoot
            state_adjustment = 0.85
            if level['price'] > current_price:  # Resistance
                level['overshoot_probability'] = overshoot_bias * 1.5
            else:  # Support
                level['overshoot_probability'] = overshoot_bias
        elif market_state == 'Thermal':
            # Levels are stronger, precision events, extreme kurtosis
            state_adjustment = 1.10
            level['precision_event_probability'] = 0.35
            # Extreme kurtosis indicates non-random market memory
            if microstructure_state['characteristics'].get('kurtosis', 0) > 30:
                level['extreme_kurtosis'] = float(microstructure_state['characteristics']['kurtosis'])
        else:  # Coherent
            # Levels highly reliable, structural manifolds, 87.11% capture rate
            state_adjustment = 1.15
            level['manifold_capture_rate'] = capture_rate  # 87.11% for Coherent state
        
        # 2. Liquidity Permeability (NEW!)
        # How easily price passes through level
        level['liquidity_permeability'] = liquidity_permeability
        permeability_adjustment = 1.0 - (liquidity_permeability * 0.3)
        
        # 3. FIXED: Distance-based adjustment using sigma-normalized distance
        # Bounded logic: levels further away get slightly weaker, but capped
        # This replaces the old vol_ratio-based adjustment that leaked volatility
        vol_adjustment = np.clip(1.0 - 0.15 * distance_sigma, 0.8, 1.1)
        
        # NOTE: GARCH vol_ratio is NO LONGER used for strength adjustment
        # GARCH only affects confidence/breakout probability (see below)
        
        # 4. Persistence adjustment
        persistence_multiplier = 0.9 + (persistence * 0.2)
        
        # 5. Volatility trend adjustment
        if vol_trend == "Increasing":
            trend_adjustment = 0.95
        elif vol_trend == "Decreasing":
            trend_adjustment = 1.05
        else:
            trend_adjustment = 1.0
        
        # 6. HMM regime
        if hmm_regime['state'] == 0:  # Bearish
            if level['price'] > current_price:
                hmm_adjustment = 1.25
            else:
                hmm_adjustment = 1.15
        elif hmm_regime['state'] == 2:  # Bullish
            if level['price'] < current_price:
                hmm_adjustment = 1.25
            else:
                hmm_adjustment = 1.15
        else:
            hmm_adjustment = 1.0
        
        # 7. Hurst
        hurst_multiplier = hurst_data['level_multiplier']
        
        # ===== COMBINE ALL ADJUSTMENTS FOR STRUCTURAL VALIDITY =====
        # This answers: "Does this level matter?" (Level Quality)
        # Should often be 70-90%
        adjusted_strength = (original_strength * 
                           state_adjustment *
                           permeability_adjustment *
                           vol_adjustment * 
                           persistence_multiplier * 
                           trend_adjustment * 
                           hmm_adjustment * 
                           hurst_multiplier)
        
        # Cap structural validity at 0.95 (very strong levels)
        level_strength = min(adjusted_strength, 0.95)
        
        # ===== IMMEDIATE REVERSAL PROBABILITY (Event Probability) =====
        # This answers: "Will price reverse RIGHT NOW on first touch?"
        # Should rarely exceed 60%, especially intraday
        # This is a HIGH BAR - most good levels get tagged, stall, wick, rotate, then resolve later
        
        # Base from structural validity, but penalized for immediate event
        # Immediate reversal is harder than "level matters"
        base_immediate_reversion = level_strength * 0.75  # Penalty for immediate event
        
        # FIXED: GARCH only affects confidence/breakout probability, NOT structural strength
        # GARCH is slow-moving, multi-day, belief-level
        # Levels are session-level, execution-level
        current_vol = garch_vol_regime.get('current_vol', 20.0)
        forecast_vol = garch_vol_regime.get('forecast_vol_5d', current_vol)
        vol_change_factor = forecast_vol / current_vol if current_vol > 0 else 1.0
        
        # Check if volatility is expanding (reduces immediate reversal prob)
        expanding_vol = vol_change_factor > 1.1
        
        # Apply state-specific multipliers to immediate reversal
        if vol_change_factor > 1.1:  # Vol rising
            breakout_boost = 0.1 * (vol_change_factor - 1) * state_multipliers['breakout_prob']
            immediate_breakout_prob = min(base_immediate_reversion * state_multipliers['breakout_prob'] + breakout_boost, 0.95)
            immediate_reversion_prob = 1 - immediate_breakout_prob
        else:
            immediate_breakout_prob = float(base_immediate_reversion * state_multipliers['breakout_prob'])
            immediate_reversion_prob = float(1 - immediate_breakout_prob)
        
        # Get confluence factors for conditional boost
        micro_conf = microstructure_state.get('confidence', 0.5)
        lss = microstructure_state.get('lss', 0.5)
        hmm_conf = hmm_regime.get('confidence', 0.5)
        vol_regime = garch_vol_regime.get('regime', 'Unknown')
        
        # Check if all factors agree (high confluence) - for levels only
        all_agree = (
            level_strength > 0.75 and  # Strong structural validity
            micro_conf > 0.65 and
            lss > 0.5 and
            hmm_conf > 0.6 and
            not expanding_vol  # Not in expanding volatility
        )
        
        # Only boost immediate reversal prob when ALL align (small, honest boost)
        if all_agree and immediate_reversion_prob > 0.55:
            immediate_reversion_prob = min(immediate_reversion_prob * 1.07, 0.95)
            immediate_breakout_prob = 1 - immediate_reversion_prob
        
        # Confluence score: how many factors are aligned (0-1) - separate metric
        confluence_factors = [
            micro_conf > 0.6,
            lss > 0.5,
            hmm_conf > 0.6,
            vol_regime in ["Low Vol Compression", "Normal Vol", "High Vol Expansion"],
            distance_sigma < 1.0,  # Level is within 1σ
            level_strength > 0.6  # Level has good structural strength
        ]
        confluence_score = sum(confluence_factors) / len(confluence_factors)
        
        # ===== MEANSHIFT VALIDATION (validator, not producer) =====
        # MeanShift validates local modal stability - if it agrees with HDBSCAN, boost confidence
        # This is a validator, not a level producer
        meanshift_validation_boost = 0.0
        if (level.get('category') == 'Density (HDBSCAN)' or level.get('category') == 'HDBSCAN') and len(meanshift_validator_levels) > 0:
            level_price = level.get('price', current_price)
            
            # Check if MeanShift finds a level near this HDBSCAN level (within 0.5σ)
            for ms_level in meanshift_validator_levels:
                ms_price = ms_level.get('price', 0)
                ms_distance_sigma = abs(ms_price - level_price) / sigma_price if sigma_price > 0 else float('inf')
                
                if ms_distance_sigma < 0.5:  # MeanShift agrees within 0.5σ
                    # Boost confidence slightly (MeanShift validates local modal stability)
                    meanshift_validation_boost = 0.03  # Small boost for validation agreement
                    level['meanshift_validated'] = True
                    level['meanshift_validation_distance_sigma'] = float(ms_distance_sigma)
                    break
        
        # Apply MeanShift validation boost to level strength (if validated)
        if meanshift_validation_boost > 0:
            level_strength = min(level_strength + meanshift_validation_boost, 0.95)
        
        # ===== ASSIGN TO LEVEL OBJECT =====
        # Split into two separate metrics for clarity
        level['level_strength'] = float(level_strength)  # Structural validity (70-90% typical) - HEADLINE METRIC
        level['first_touch_reversal_prob'] = float(immediate_reversion_prob)  # Event probability (rarely >60%) - honest name
        level['immediate_breakout_prob'] = float(immediate_breakout_prob)
        
        # Backward compatibility: keep old fields but use new names
        level['immediate_reversion_prob'] = float(immediate_reversion_prob)  # Keep for backward compat
        level['reversionProb'] = float(immediate_reversion_prob)
        level['breakoutProb'] = float(immediate_breakout_prob)
        level['strength'] = float(level_strength)  # Structural validity, not event prob - PRIMARY METRIC
        
        # Confluence score (separate metric)
        level['confluence_score'] = float(confluence_score)
        
        # GARCH confidence boost (metadata only, for commentary)
        if vol_ratio > 1.3:
            level['garch_confidence_boost'] = 0.05  # High vol = slightly higher breakout confidence
        elif vol_ratio < 0.85:
            level['garch_confidence_boost'] = -0.05  # Low vol = slightly higher reversion confidence
        else:
            level['garch_confidence_boost'] = 0.0
        
        # Add comprehensive metadata
        level['market_state'] = market_state
        level['state_confidence'] = microstructure_state.get('confidence', 0.5)
        level['garch_vol_regime'] = garch_vol_regime.get('regime', 'Unknown')
        level['garch_current_vol'] = float(current_vol)
        level['garch_forecast_vol'] = float(forecast_vol)
        level['garch_vol_trend'] = vol_trend
        level['garch_persistence'] = float(persistence)
        level['hmm_regime'] = hmm_regime.get('regime', 'Unknown')
        level['hmm_confidence'] = hmm_regime.get('confidence', 0.5)
        level['hurst_exponent'] = hurst_data.get('hurst', 0.5)
        level['hurst_regime'] = hurst_data.get('regime', 'Random')
        
        # Distance calculations (for metadata)
        distance_dollars = abs(level['price'] - current_price)
        level['distance_dollars'] = float(distance_dollars)
        level['distance_pct'] = float(distance_pct * 100)
        level['distance_sigma'] = float(distance_sigma)  # NEW: sigma-normalized distance
    
    return levels, hmm_regime, hurst_data, garch_vol_regime, microstructure_state


def calculate_garch_confidence_bands(forecasts, garch_vol_regime):
    """
    Enhanced confidence bands using GARCH volatility forecast
    """
    if 'ensemble' not in forecasts:
        return forecasts
    
    ensemble = forecasts['ensemble']
    # Be defensive: garch_vol_regime may be missing current_vol for some timeframes (e.g. intraday/4h)
    current_vol = float(garch_vol_regime.get('current_vol', 20.0))
    forecast_vols = garch_vol_regime.get('forecast_vol_array', [current_vol] * 10)
    
    upper_band = []
    lower_band = []
    
    for i, price in enumerate(ensemble):
        if i < len(forecast_vols):
            horizon_vol = forecast_vols[i]
        else:
            horizon_vol = current_vol * (1 + 0.05 * i)
        
        upper_band.append(float(price + horizon_vol * 1.5))
        lower_band.append(float(price - horizon_vol * 1.5))
    
    forecasts['upper_confidence'] = upper_band
    forecasts['lower_confidence'] = lower_band
    forecasts['garch_enhanced'] = True
    
    return forecasts


def calculate_most_probable_price_path(closes, volumes, levels, garch_vol_regime, phase_space, microstructure_state, forecast_periods=30, iv_surface_data=None, timeframe='1d', sigma_price=None):
    """
    Calculate most probable price path using:
    - Phase space velocity/momentum for DIRECTION
    - GARCH/IV for expected RANGE (multi-day) OR session vol (intraday)
    - Levels for TARGETS
    - High probability confluence for most probable move
    
    FIXED: Uses session volatility for intraday timeframes instead of annualized
    """
    if len(closes) < 50:
        return None
    
    current_price = closes[-1]
    returns = np.log(closes[1:] / closes[:-1]) * 100
    
    # 1. DETERMINE DIRECTION from Phase Space
    if phase_space and len(phase_space.get('velocity', [])) > 0:
        recent_velocity = phase_space['velocity'][-1] if phase_space['velocity'] else 0
        # Average velocity over last 5 periods for more stable direction
        if len(phase_space['velocity']) >= 5:
            avg_velocity = np.mean(phase_space['velocity'][-5:])
        else:
            avg_velocity = recent_velocity
    else:
        # Fallback: calculate from price gradient
        recent_velocity = np.gradient(closes)[-1] if len(closes) > 1 else 0
        if len(closes) >= 5:
            avg_velocity = np.mean(np.gradient(closes)[-5:])
        else:
            avg_velocity = recent_velocity
    
    # Determine direction: positive = up, negative = down
    direction = 1 if avg_velocity > 0 else -1
    velocity_strength = abs(avg_velocity) / current_price if current_price > 0 else 0
    
    # 2. GET EXPECTED RANGE - FIXED: Use session vol for intraday, GARCH for multi-day
    # FIXED: Initialize garch_forecast_vols unconditionally to prevent UnboundLocalError
    garch_forecast_vols = garch_vol_regime.get('forecast_vol_array', []) if garch_vol_regime else []
    
    is_intraday = timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']
    
    if is_intraday and sigma_price is not None:
        # FIXED: For intraday, use session volatility (sigma_price) instead of annualized
        # sigma_price is already in price units (e.g., $10 for SPY)
        session_vol = sigma_price / current_price if current_price > 0 else 0.02
        expected_vol = session_vol  # Already session-level, no conversion needed
        print(f"✓ Using session volatility for intraday path: {expected_vol:.2%}")
    else:
        # Multi-day: use GARCH (annualized)
        current_vol = garch_vol_regime.get('current_vol', np.std(returns) * np.sqrt(252)) if garch_vol_regime else np.std(returns) * np.sqrt(252)
    
    if garch_forecast_vols:
        # Use average of next 10 days for expected range
        expected_vol = np.mean(garch_forecast_vols[:min(10, len(garch_forecast_vols))]) / 100
    else:
        expected_vol = current_vol / 100
    
    # Convert annualized to daily for multi-day paths
    daily_vol = expected_vol * np.sqrt(1/252)
    expected_vol = daily_vol
    print(f"✓ Using GARCH volatility for multi-day path: {expected_vol:.2%} (daily)")
    
    # FIXED: Calculate sigma_price in price units for consistent σ-normalized distance
    # sigma_price is always in price units (e.g., $10 for SPY), regardless of timeframe
    if is_intraday and sigma_price is not None:
        # Already have sigma_price in price units for intraday
        sigma_price_path = sigma_price
    else:
        # For multi-day, calculate sigma_price from expected_vol
        sigma_price_path = expected_vol * current_price
    
    # Expected range: 1-2 standard deviations (in price units)
    expected_range_1sd = sigma_price_path
    expected_range_2sd = sigma_price_path * 2
    
    # 3. GET ALL LEVELS and filter by direction and range
    all_levels = []
    for level_type, level_list in levels.items():
        if isinstance(level_list, list):
            all_levels.extend(level_list)
    
    # Filter levels: must be within expected range AND in direction of momentum (or very close)
    # FIXED: Use σ-normalized distance instead of % distance for consistency
    candidate_levels = []
    for level in all_levels:
        level_price = level.get('price', current_price)
        distance = level_price - current_price
        
        # FIXED: Use sigma-normalized distance (regime-invariant, execution-aligned)
        # distance is in price units, sigma_price_path is in price units → distance_sigma is unitless
        distance_sigma = abs(distance) / sigma_price_path if sigma_price_path > 0 else float('inf')
        
        # Must be within 2 standard deviations (in σ-space)
        if distance_sigma < 2.0:
            # Check if in direction of momentum (or very close for mean reversion)
            # Use small sigma threshold for "very close" (0.1σ = very close)
            in_direction = (distance * direction > 0) or (distance_sigma < 0.1)
            
            if in_direction or distance_sigma < 0.2:  # Always consider very close levels (< 0.2σ)
                candidate_levels.append(level)
    
    # 4. FIND HIGHEST PROBABILITY CONFLUENCE in the right direction
    # Prioritize confluence levels
    confluence_levels = [l for l in candidate_levels if l.get('category') == 'ML-Confluence']
    
    target_level = None
    highest_probability = 0
    
    # Score each candidate level
    for level in candidate_levels:
        level_price = level.get('price', current_price)
        distance = level_price - current_price
        
        # FIXED: Use sigma-normalized distance for scoring (consistent with filtering)
        # distance is in price units, sigma_price_path is in price units → distance_sigma is unitless
        distance_sigma = abs(distance) / sigma_price_path if sigma_price_path > 0 else float('inf')
        
        reversion_prob = level.get('reversionProb', 0)
        strength = level.get('strength', 0)
        confluence_count = level.get('confluence_count', 1)
        
        # Base probability from level strength
        base_prob = (reversion_prob * 0.6 + strength * 0.4)
        
        # Boost for confluence
        if level.get('category') == 'ML-Confluence':
            base_prob *= (1 + confluence_count * 0.15)
        
        # Direction bonus: higher score if level is in direction of momentum
        direction_bonus = 1.0
        if velocity_strength > 0.001:  # If there's meaningful momentum
            if (distance * direction > 0):  # Level is in direction of momentum
                direction_bonus = 1.3  # 30% bonus for following momentum
            elif (distance * direction < 0):  # Level is against momentum
                direction_bonus = 0.7  # Penalty for going against momentum
        
        # FIXED: Distance factor in σ-space (closer is better, but not too close)
        if distance_sigma < 0.1:
            distance_factor = 0.8  # Already very close (< 0.1σ), less interesting
        elif distance_sigma < 0.5:
            distance_factor = 1.2  # Sweet spot (0.1-0.5σ)
        else:
            distance_factor = 1.0 - (distance_sigma - 0.5) * 0.3  # Farther = less interesting (capped)
            distance_factor = max(0.5, distance_factor)  # Don't penalize too much
        
        # Market state adjustment
        market_state = microstructure_state.get('state', 'Unknown')
        state_factor = 1.0
        if market_state == 'Coherent':
            state_factor = 1.2  # Stronger in coherent
        elif market_state == 'Thermal':
            state_factor = 1.1
        
        # Combined probability
        combined_prob = base_prob * direction_bonus * distance_factor * state_factor
        
        if combined_prob > highest_probability:
            highest_probability = combined_prob
            target_level = level
    
    # 5. If no good target found, use momentum-based extension
    if not target_level or highest_probability < 0.4:
        # Extend in direction of momentum with GARCH volatility
        path = []
        for step in range(forecast_periods):
            # Momentum component (decays)
            momentum_component = avg_velocity * (0.95 ** step) * 0.5
            
            # Volatility component (from GARCH)
            if step < len(garch_forecast_vols):
                step_vol = garch_forecast_vols[step] / 100
            else:
                step_vol = expected_vol
            
            daily_vol = step_vol * np.sqrt(1/252)
            vol_component = np.random.normal(0, daily_vol) * current_price * 0.2
            
            next_price = current_price + momentum_component + vol_component
            path.append(float(next_price))
            current_price = next_price
        
        return {
            'path': path,
            'current_price': float(closes[-1]),
            'forecast_periods': forecast_periods,
            'method': 'Momentum + GARCH',
            'target_level': None,
            'probability': 0.4,
            'direction': 'up' if direction > 0 else 'down'
        }
    
    # 6. GENERATE PATH to target level using GARCH volatility
    target_price = target_level.get('price', current_price)
    distance = target_price - current_price
    # FIXED: Use sigma-normalized distance for consistency
    # distance is in price units, sigma_price_path is in price units → distance_sigma is unitless
    distance_sigma = abs(distance) / sigma_price_path if sigma_price_path > 0 else float('inf')
    
    # Calculate steps needed based on volatility
    # Use GARCH forecast to determine how fast we can move
    if garch_forecast_vols:
        avg_forecast_vol = np.mean(garch_forecast_vols[:min(forecast_periods, len(garch_forecast_vols))]) / 100
        # Convert to daily for multi-day paths
        if not is_intraday:
            avg_forecast_vol = avg_forecast_vol * np.sqrt(1/252)
    else:
        avg_forecast_vol = expected_vol
    
    # Steps to target: based on volatility and distance (in σ-space)
    # Use sigma_price_path (in price units) for move capacity
    daily_move_capacity = sigma_price_path  # Already in price units
    steps_to_target = max(3, min(forecast_periods, int(distance_sigma * 2)))  # 2 steps per σ
    
    # Generate path
    path = []
    current_pos = closes[-1]
    
    for step in range(forecast_periods):
        if step < steps_to_target:
            # Move toward target
            progress = (step + 1) / steps_to_target
            # Smooth easing function
            eased_progress = progress * progress * (3 - 2 * progress)
            
            base_move = distance * eased_progress
            
            # Add volatility from GARCH (realistic movement)
            if step < len(garch_forecast_vols):
                step_vol = garch_forecast_vols[step] / 100
            else:
                step_vol = avg_forecast_vol
            
            daily_vol = step_vol * np.sqrt(1/252)
            # Add small random component for realism (20% of full vol)
            volatility_component = np.random.normal(0, daily_vol) * current_pos * 0.2
            
            next_price = closes[-1] + base_move + volatility_component
        else:
            # Reached target, stay near it with small oscillations
            oscillation = np.sin(step * 0.2) * target_price * avg_forecast_vol * 0.01
            next_price = target_price + oscillation
        
        path.append(float(next_price))
    
    return {
        'path': path,
        'current_price': float(closes[-1]),
        'forecast_periods': forecast_periods,
        'method': 'Phase Space Direction + GARCH Range + Confluence',
        'target_level': {
            'price': float(target_price),
            'strength': float(target_level.get('strength', 0)),
            'reversionProb': float(target_level.get('reversionProb', 0)),
            'confluence_count': target_level.get('confluence_count', 1)
        },
        'probability': float(highest_probability),
        'direction': 'up' if direction > 0 else 'down',
        'velocity_strength': float(velocity_strength)
    }

# ============================================================================
# VOLATILITY SURFACE CALCULATION
# ============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option pricing"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def vega(S, K, T, r, sigma):
    """Calculate vega for Black-Scholes"""
    if T <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def get_atm_iv_yf(ticker: str, max_expiries: int = 3):
    """
    Returns a single implied volatility estimate (decimal, e.g. 0.22 for 22%)
    using near-dated ATM call/put mid IV from yfinance.

    If IV is unavailable, returns None.
    """
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return None

        # Use earliest expiries (near-dated)
        expiries = expiries[:max_expiries]

        spot = float(t.history(period="1d")["Close"].iloc[-1])

        iv_samples = []

        for exp in expiries:
            chain = t.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # yfinance usually has 'impliedVolatility'
            if "impliedVolatility" not in calls.columns or "impliedVolatility" not in puts.columns:
                continue

            # Find closest strike to spot
            calls["dist"] = (calls["strike"] - spot).abs()
            puts["dist"]  = (puts["strike"]  - spot).abs()

            c = calls.sort_values("dist").iloc[0]
            p = puts.sort_values("dist").iloc[0]

            civ = float(c.get("impliedVolatility", np.nan))
            piv = float(p.get("impliedVolatility", np.nan))

            # Use average of call/put if both exist
            vals = [v for v in [civ, piv] if np.isfinite(v) and v > 0]
            if vals:
                iv_samples.append(float(np.mean(vals)))

        if not iv_samples:
            return None

        # Median is more robust than mean
        return float(np.median(iv_samples))

    except Exception:
        return None

def implied_volatility(market_price, S, K, T, r):
    """Calculate implied volatility using Newton-Raphson"""
    sigma = 0.3
    max_iterations = 100
    tolerance = 1e-6
    
    for i in range(max_iterations):
        price = black_scholes_call(S, K, T, r, sigma)
        diff = price - market_price
        
        if abs(diff) < tolerance:
            return sigma
        
        vega_val = vega(S, K, T, r, sigma)
        if vega_val < 1e-10:
            break
            
        sigma = sigma - diff / vega_val
        sigma = max(0.01, min(sigma, 5.0))
    
    return sigma

def generate_volatility_surface(current_price, garch_vol_regime):
    """
    Generate EXPECTED VOLATILITY SURFACE (synthetic, GARCH-anchored).
    
    IMPORTANT: This is NOT an implied volatility surface from option prices.
    It is a synthetic expected volatility surface for range projection and
    distribution geometry, NOT dealer positioning or option pricing.
    
    Returns actionable scalars:
    - tail_risk_score: wing steepness → tail multiplier hint
    - compression_score: surface flatness → compression vs expansion bias
    - atm_variance_by_horizon: expected variance by maturity
    """
    
    # --- HARD GUARD: ensure garch_vol_regime is always a dict ---
    if not garch_vol_regime or not isinstance(garch_vol_regime, dict):
        garch_vol_regime = {
            'garch_params': None,
            'forecast_vol_array': [],
            'current_vol': 20.0,
            'regime_factor': 1.0
        }
    
    # FIXED: Explicit decimal/percentage naming for unit safety
    if garch_vol_regime.get('garch_params') is not None:
        # Be defensive: fall back to 20% if current_vol is missing
        atm_vol_pct = float(garch_vol_regime.get('current_vol', 20.0))  # Percentage (e.g., 20.0 = 20%)
        atm_vol_dec = atm_vol_pct / 100.0  # Decimal (e.g., 0.20 = 20%)
    else:
        atm_vol_pct = 20.0
        atm_vol_dec = 0.20
    
    # Get regime factor for regime-aware skew/smile
    regime_factor = garch_vol_regime.get('regime_factor', 1.0)
    
    moneyness_range = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
    strikes = [m * current_price for m in moneyness_range]
    maturities_days = [7, 14, 30, 60, 90, 180, 365]
    maturities = [d / 365.0 for d in maturities_days]
    
    surface_data = []
    all_ivs = []  # For extracting actionable scalars
    
    for T_days, T in zip(maturities_days, maturities):
        for moneyness, K in zip(moneyness_range, strikes):
            # FIXED: Regime-aware skew/smile (steepen in high vol, flatten in compression)
            skew = -0.15 * regime_factor * (moneyness - 1)
            smile = 0.08 * regime_factor * (moneyness - 1)**2
            
            # FIXED: Term structure using sqrt(T) for proper variance scaling
            # This ensures longer maturities widen appropriately
            term_structure = 0.06 * np.sqrt(T)
            
            # FIXED: GARCH adjustment - map maturity to forecast horizon smoothly
            # Instead of clamping to index 29, interpolate across forecast horizon
            garch_adjustment_dec = 0.0
            if garch_vol_regime.get('garch_params'):
                forecast_vols_pct = garch_vol_regime.get('forecast_vol_array', [])
                if forecast_vols_pct and len(forecast_vols_pct) > 0:
                    # Map maturity to forecast horizon (smooth interpolation)
                    horizon = min(len(forecast_vols_pct), 30)
                    t_frac = min(T_days / 365.0, 1.0)  # Fraction of year
                    idx = int(t_frac * (horizon - 1))
                    idx = min(idx, len(forecast_vols_pct) - 1)
                    
                    # Convert forecast vol to decimal for calculation
                    forecast_vol_dec = forecast_vols_pct[idx] / 100.0
                    garch_adjustment_dec = (forecast_vol_dec - atm_vol_dec) * 0.5

            # Calculate expected vol in decimal
            expected_vol_dec = max(
                0.05,
                atm_vol_dec + skew + smile + term_structure + garch_adjustment_dec
            )
            
            # Convert to percentage for output
            expected_vol_pct = expected_vol_dec * 100.0
            all_ivs.append(expected_vol_dec)

            surface_data.append({
                'strike': float(K),
                'maturity_days': int(T_days),
                'maturity_years': float(T),
                'moneyness': float(moneyness),
                'implied_vol': float(expected_vol_pct),  # FIXED: Use expected_vol_pct
                'atm_vol': float(atm_vol_pct)  # FIXED: Use atm_vol_pct
            })
    
    # Extract actionable scalars from surface for trading logic integration
    all_ivs_array = np.array(all_ivs)
    
    # Tail risk score: wing steepness (OTM vol vs ATM vol)
    # Higher = steeper wings = more tail risk = higher tail multiplier hint
    otm_vols = [iv for i, iv in enumerate(all_ivs_array) 
                if surface_data[i]['moneyness'] < 0.9 or surface_data[i]['moneyness'] > 1.1]
    atm_vols = [iv for i, iv in enumerate(all_ivs_array) 
                if 0.95 <= surface_data[i]['moneyness'] <= 1.05]
    
    if len(otm_vols) > 0 and len(atm_vols) > 0:
        avg_otm_vol = np.mean(otm_vols)
        avg_atm_vol = np.mean(atm_vols)
        tail_risk_score = float((avg_otm_vol / avg_atm_vol - 1.0) if avg_atm_vol > 0 else 0.0)
    else:
        tail_risk_score = 0.0
    
    # Compression score: surface flatness (vol range across strikes)
    # Lower = flatter = more compressed = compression bias
    # Higher = steeper = more expansion = expansion bias
    if len(all_ivs_array) > 0:
        vol_range = float(np.max(all_ivs_array) - np.min(all_ivs_array))
        compression_score = float(1.0 - min(vol_range / 0.20, 1.0))  # Normalize to 0-1
    else:
        compression_score = 0.5
    
    # ATM variance by horizon (for range projection)
    atm_variance_by_horizon = {}
    for T_days in maturities_days:
        horizon_ivs = [iv for i, iv in enumerate(all_ivs_array) 
                      if surface_data[i]['maturity_days'] == T_days 
                      and 0.95 <= surface_data[i]['moneyness'] <= 1.05]
        if len(horizon_ivs) > 0:
            # Variance = vol^2 * T (annualized)
            avg_vol_dec = np.mean(horizon_ivs)
            T_years = T_days / 365.0
            variance = float(avg_vol_dec ** 2 * T_years)
            atm_variance_by_horizon[T_days] = variance
    
    return {
        'surface': surface_data,
        'current_price': float(current_price),
        'atm_vol': float(atm_vol_pct),  # FIXED: Use atm_vol_pct
        'garch_calibrated': bool(garch_vol_regime.get('garch_params')),
        # NEW: Actionable scalars for trading logic
        'tail_risk_score': tail_risk_score,  # Wing steepness → tail multiplier hint
        'compression_score': compression_score,  # Surface flatness → compression vs expansion bias
        'atm_variance_by_horizon': atm_variance_by_horizon,  # Expected variance by maturity
        'regime_factor': float(regime_factor),  # Regime multiplier used
        'surface_type': 'expected_volatility'  # Clarify this is NOT implied volatility
    }


# ============================================================================
# FRED API & OTHER EXISTING FUNCTIONS
# ============================================================================

def get_fred_data(series_id, start_date=None):
    if FRED_API_KEY == 'YOUR_FRED_API_KEY_HERE':
        return None
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        url = f'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna()
            return df
        return None
    except:
        return None

def get_macro_indicators():
    indicators = {}
    fred_series = {
        'vix': 'VIXCLS',
        'dxy': 'DTWEXBGS',
        'rates_10y': 'DGS10',
        'fed_funds': 'DFF',
        'cpi': 'CPIAUCSL'
    }
    for name, series_id in fred_series.items():
        data = get_fred_data(series_id)
        if data is not None and len(data) > 0:
            indicators[name] = {
                'current': float(data.iloc[-1]['value']),
                'change_1m': float(data.iloc[-1]['value'] - data.iloc[-20]['value']) if len(data) > 20 else 0
            }
    return indicators

def nbeats_forecast(prices, forecast_periods=10, num_scenarios=3):
    if len(prices) < 50:
        return None
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    window = min(20, len(prices) // 3)
    if window < 5:
        window = 5
    
    # FIX: Replace deprecated fillna(method='bfill') and fillna(method='ffill')
    trend = pd.Series(prices_scaled).rolling(window=window, center=True).mean()
    trend = trend.bfill().ffill().values  # NEW WAY
    
    residual = prices_scaled - trend
    recent_trend = trend[-10:]
    trend_slope = (recent_trend[-1] - recent_trend[0]) / len(recent_trend)
    scenarios = []
    for scenario_idx in range(num_scenarios):
        forecast = []
        last_price = prices_scaled[-1]
        vol = np.std(residual[-50:])
        if scenario_idx == 0:
            trend_multiplier = 1.3
            vol_multiplier = 0.8
        elif scenario_idx == 1:
            trend_multiplier = 1.0
            vol_multiplier = 1.0
        else:
            trend_multiplier = 0.7
            vol_multiplier = 1.2
        for step in range(forecast_periods):
            trend_component = trend_slope * trend_multiplier
            mean_reversion = -0.1 * (last_price - np.mean(prices_scaled[-20:]))
            dampening = 0.95 ** step
            noise = np.random.normal(0, vol * vol_multiplier * dampening)
            next_val = last_price + trend_component + mean_reversion + noise
            forecast.append(next_val)
            last_price = next_val
        forecast_original = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        scenarios.append(forecast_original.tolist())
    return {'bullish': scenarios[0], 'base': scenarios[1], 'bearish': scenarios[2]}

def tcn_style_forecast(prices, volumes, forecast_periods=10):
    if len(prices) < 50:
        return None
    scales = [5, 10, 20, 40]
    weighted_predictions = []
    for scale in scales:
        if len(prices) < scale:
            continue
        weights = np.exp(np.linspace(-2, 0, scale))
        weights /= weights.sum()
        recent = prices[-scale:]
        weighted_avg = np.sum(recent * weights)
        trend = (prices[-1] - prices[-scale]) / scale
        weighted_predictions.append(weighted_avg + trend * forecast_periods / 2)
    if len(weighted_predictions) == 0:
        return None
    base_prediction = np.mean(weighted_predictions)
    forecast = []
    vol = np.std(np.diff(prices[-50:]))
    for step in range(forecast_periods):
        noise = np.random.normal(0, vol * (0.95 ** step))
        next_price = prices[-1] + (base_prediction - prices[-1]) * (step + 1) / forecast_periods + noise
        forecast.append(float(next_price))
    return forecast

def generate_price_forecast(closes, highs, lows, volumes, forecast_periods=20):
    forecasts = {}
    nbeats = nbeats_forecast(closes, forecast_periods=forecast_periods, num_scenarios=3)
    if nbeats:
        forecasts['scenarios'] = nbeats
    tcn = tcn_style_forecast(closes, volumes, forecast_periods=forecast_periods)
    if tcn:
        forecasts['tcn'] = tcn
    if nbeats and tcn:
        ensemble = []
        for i in range(forecast_periods):
            avg = (nbeats['base'][i] + tcn[i]) / 2
            ensemble.append(float(avg))
        forecasts['ensemble'] = ensemble
    if 'ensemble' in forecasts:
        vol = np.std(closes[-50:])
        upper_band = [p + vol * (1 + 0.1 * i) for i, p in enumerate(forecasts['ensemble'])]
        lower_band = [p - vol * (1 + 0.1 * i) for i, p in enumerate(forecasts['ensemble'])]
        forecasts['upper_confidence'] = upper_band
        forecasts['lower_confidence'] = lower_band
    return forecasts

def detect_market_regime_hmm(closes, n_states=3):
    if not HMMLEARN_AVAILABLE or GaussianHMM is None:
        return {'state': 1, 'regime': 'Neutral', 'confidence': 0.5}
    try:
        returns = np.diff(np.log(closes)).reshape(-1, 1)
        if len(returns) < 50:
            return {'state': 1, 'regime': 'Neutral', 'confidence': 0.5}
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        model.fit(returns)
        states = model.predict(returns)
        current_state = int(states[-1])
        state_probs = model.predict_proba(returns)[-1]
        confidence = float(state_probs[current_state])
        regime_names = ['Bearish', 'Neutral', 'Bullish']
        return {'state': current_state, 'regime': regime_names[current_state], 'confidence': confidence}
    except:
        return {'state': 1, 'regime': 'Neutral', 'confidence': 0.5}

def calculate_hurst_exponent(closes, max_lag=20):
    if len(closes) < max_lag * 2:
        max_lag = len(closes) // 2
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(closes[lag:], closes[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    H = poly[0]
    if H < 0.4:
        regime = 'Mean-Reverting'
        level_multiplier = 1.4
    elif H > 0.6:
        regime = 'Trending'
        level_multiplier = 0.7
    else:
        regime = 'Random Walk'
        level_multiplier = 1.0
    return {'hurst': float(H), 'regime': regime, 'level_multiplier': level_multiplier}





# [ALL THE LEVEL DETECTION FUNCTIONS - KEEPING THEM EXACTLY AS BEFORE]

def find_peaks_valleys_scipy(highs, lows, closes, prominence=0.02):
    price_range = highs.max() - lows.min()
    min_prominence = price_range * prominence
    if len(closes) > 11:
        smoothed = savgol_filter(closes, window_length=11, polyorder=3)
    else:
        smoothed = closes
    peaks, peak_props = find_peaks(smoothed, prominence=min_prominence, distance=5)
    valleys, valley_props = find_peaks(-smoothed, prominence=min_prominence, distance=5)
    levels = []
    for i, peak_idx in enumerate(peaks):
        if peak_idx >= len(highs):
            continue
        level_price = highs[peak_idx]
        touches = np.sum(np.abs(highs - level_price) < level_price * 0.005)
        bars_ago = len(closes) - peak_idx
        recency = 1.0 / (1 + bars_ago / 50)
        prom_strength = min(peak_props['prominences'][i] / min_prominence / 3, 0.9)
        strength = (prom_strength * 0.6 + recency * 0.4) * min(touches / 3, 1.0)
        levels.append({'price': float(level_price), 'type': 'Peak Resistance', 'touches': int(touches), 
                      'strength': float(strength), 'breakoutProb': float(1 - strength), 
                      'reversionProb': float(strength), 'category': 'Peak-Valley'})
    for i, valley_idx in enumerate(valleys):
        if valley_idx >= len(lows):
            continue
        level_price = lows[valley_idx]
        touches = np.sum(np.abs(lows - level_price) < level_price * 0.005)
        bars_ago = len(closes) - valley_idx
        recency = 1.0 / (1 + bars_ago / 50)
        prom_strength = min(valley_props['prominences'][i] / min_prominence / 3, 0.9)
        strength = (prom_strength * 0.6 + recency * 0.4) * min(touches / 3, 1.0)
        levels.append({'price': float(level_price), 'type': 'Valley Support', 'touches': int(touches),
                      'strength': float(strength), 'breakoutProb': float(1 - strength),
                      'reversionProb': float(strength), 'category': 'Peak-Valley'})
    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:10]

def calculate_meanshift_levels(highs, lows, closes):
    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)
    bandwidth = estimate_bandwidth(all_prices, quantile=0.15, n_samples=min(len(all_prices), 1000))
    if bandwidth == 0:
        bandwidth = (all_prices.max() - all_prices.min()) / 20
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(all_prices)
    cluster_centers = ms.cluster_centers_.flatten()
    labels = ms.labels_
    levels = []
    for i, center in enumerate(cluster_centers):
        touches = np.sum(labels == i)
        strength = min(touches / len(all_prices) * 10, 0.90)
        levels.append({'price': float(center), 'type': 'MeanShift', 'touches': int(touches), 
                      'strength': strength, 'breakoutProb': float(1 - strength), 
                      'reversionProb': float(strength), 'category': 'MeanShift'})
    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:6]


def calculate_pivot_points(hist_data, timeframe):
    if len(hist_data) < 2:
        return []
    prev = hist_data.iloc[-2]
    high, low, close = prev['High'], prev['Low'], prev['Close']
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    period_name = "Day" if timeframe == "1d" else "Period"
    return [
        {'price': float(pivot), 'type': f'{period_name} Pivot', 'strength': 0.85, 'breakoutProb': 0.15, 'reversionProb': 0.85, 'category': 'Pivot'},
        {'price': float(r1), 'type': f'{period_name} R1', 'strength': 0.75, 'breakoutProb': 0.25, 'reversionProb': 0.75, 'category': 'Pivot'},
        {'price': float(s1), 'type': f'{period_name} S1', 'strength': 0.75, 'breakoutProb': 0.25, 'reversionProb': 0.75, 'category': 'Pivot'},
        {'price': float(r2), 'type': f'{period_name} R2', 'strength': 0.65, 'breakoutProb': 0.35, 'reversionProb': 0.65, 'category': 'Pivot'},
        {'price': float(s2), 'type': f'{period_name} S2', 'strength': 0.65, 'breakoutProb': 0.35, 'reversionProb': 0.65, 'category': 'Pivot'},
        {'price': float(r3), 'type': f'{period_name} R3', 'strength': 0.55, 'breakoutProb': 0.45, 'reversionProb': 0.55, 'category': 'Pivot'},
        {'price': float(s3), 'type': f'{period_name} S3', 'strength': 0.55, 'breakoutProb': 0.45, 'reversionProb': 0.55, 'category': 'Pivot'},
    ]

def calculate_fibonacci_levels(highs, lows):
    """
    Calculate Fibonacci retracement levels.
    NOTE: These are NOT primary levels - they are psychological references.
    Use as metadata/confluence only, not as level generators.
    """
    if len(highs) < 20:
        return []
    recent_high = np.max(highs[-50:])
    recent_low = np.min(lows[-50:])
    range_val = recent_high - recent_low
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    levels = []
    for ratio in fib_ratios:
        level_from_high = recent_high - (range_val * ratio)
        levels.append({
            'price': float(level_from_high), 
            'type': f'Fib {ratio:.3f}',
            'ratio': float(ratio),
            'strength': 0.7, 
            'breakoutProb': 0.3, 
            'reversionProb': 0.7, 
            'category': 'Fibonacci',
            'is_metadata_only': True  # Flag: not a primary level
        })
    return levels

def add_fibonacci_metadata_to_levels(all_levels, fib_levels, sigma_price, threshold_sigma=1.0):
    """
    Add Fibonacci as metadata/confluence to nearby levels, not as primary levels.
    This treats Fib as a psychological reference, not discovered structure.
    """
    if not fib_levels or not all_levels or sigma_price <= 0:
        return all_levels
    
    for level in all_levels:
        level_price = level.get('price', 0)
        nearby_fibs = []
        
        for fib in fib_levels:
            fib_price = fib.get('price', 0)
            distance_sigma = abs(fib_price - level_price) / sigma_price if sigma_price > 0 else float('inf')
            
            if distance_sigma < threshold_sigma:
                nearby_fibs.append({
                    'price': float(fib_price),
                    'ratio': fib.get('ratio', 0),
                    'distance_sigma': float(distance_sigma)
                })
        
        if nearby_fibs:
            level['fibonacci_confluence'] = nearby_fibs
            level['has_fib_confluence'] = True
    
    return all_levels

def find_gap_levels(hist_data):
    gap_levels = []
    for i in range(1, len(hist_data)):
        curr = hist_data.iloc[i]
        prev = hist_data.iloc[i-1]
        if curr['Low'] > prev['High']:
            gap_mid = (curr['Low'] + prev['High']) / 2
            filled = False
            for j in range(i+1, len(hist_data)):
                if hist_data.iloc[j]['Low'] <= gap_mid:
                    filled = True
                    break
            if not filled and len(hist_data) - i < 100:
                gap_levels.append({'price': float(gap_mid), 'type': 'Gap Up', 'strength': 0.85,
                                  'breakoutProb': 0.15, 'reversionProb': 0.85, 'category': 'Gap'})
        elif curr['High'] < prev['Low']:
            gap_mid = (prev['Low'] + curr['High']) / 2
            filled = False
            for j in range(i+1, len(hist_data)):
                if hist_data.iloc[j]['High'] >= gap_mid:
                    filled = True
                    break
            if not filled and len(hist_data) - i < 100:
                gap_levels.append({'price': float(gap_mid), 'type': 'Gap Down', 'strength': 0.85,
                                  'breakoutProb': 0.15, 'reversionProb': 0.85, 'category': 'Gap'})
    return gap_levels

def find_pivot_anomalies(highs, lows, closes):
    """
    Pivots are structural anomalies in price flow.
    Uses IsolationForest to detect unusual price movements that indicate pivot points.
    """
    if len(closes) < 10:
        return []
    
    features = []
    for i in range(2, len(closes) - 2):
        # Local structure features
        features.append([
            closes[i] - closes[i-1],  # momentum
            highs[i] - lows[i],        # range
            closes[i] - closes[i-2],  # 2-bar momentum
            np.std(closes[max(0, i-2):min(len(closes), i+3)]),  # local volatility
        ])
    
    if len(features) < 5:
        return []
    
    X = np.array(features)
    
    iso = IsolationForest(
        contamination=0.05,  # 5% are pivots
        random_state=42
    )
    
    preds = iso.fit_predict(X)
    anomaly_scores = iso.score_samples(X)
    
    # Negative scores = anomalies = potential pivots
    pivot_indices = [i+2 for i, score in enumerate(anomaly_scores) 
                     if preds[i] == -1 and score < -0.5]
    
    levels = []
    for idx in pivot_indices:
        if idx >= len(closes) or idx < 2 or idx >= len(closes) - 2:
            continue
            
        # High or low pivot?
        is_high = highs[idx] > max(highs[idx-1], highs[idx+1]) if idx > 0 and idx < len(highs) - 1 else False
        is_low = lows[idx] < min(lows[idx-1], lows[idx+1]) if idx > 0 and idx < len(lows) - 1 else False
        
        score_idx = idx - 2  # Adjust for feature array indexing
        if score_idx < 0 or score_idx >= len(anomaly_scores):
            continue
        
        if is_high:
            levels.append({
                'price': float(highs[idx]),
                'type': 'Anomaly Pivot High',
                'strength': float(min(abs(anomaly_scores[score_idx]) / 2, 0.9)),
                'breakoutProb': float(1 - min(abs(anomaly_scores[score_idx]) / 2, 0.9)),
                'reversionProb': float(min(abs(anomaly_scores[score_idx]) / 2, 0.9)),
                'category': 'Isolation-Forest',
                'anomaly_score': float(anomaly_scores[score_idx])
            })
        
        if is_low:
            levels.append({
                'price': float(lows[idx]),
                'type': 'Anomaly Pivot Low',
                'strength': float(min(abs(anomaly_scores[score_idx]) / 2, 0.9)),
                'breakoutProb': float(1 - min(abs(anomaly_scores[score_idx]) / 2, 0.9)),
                'reversionProb': float(min(abs(anomaly_scores[score_idx]) / 2, 0.9)),
                'category': 'Isolation-Forest',
                'anomaly_score': float(anomaly_scores[score_idx])
            })
    
    return levels

def optics_multi_density_levels(highs, lows, closes):
    """
    OPTICS finds levels at multiple density scales.
    Catches both major S/R and minor pivots in one pass.
    """
    if len(closes) < 20:
        return []
    
    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)
    
    optics = OPTICS(
        min_samples=5,
        xi=0.05,  # Sensitivity to density changes
        min_cluster_size=10
    )
    
    labels = optics.fit_predict(all_prices)
    reachability = optics.reachability_[optics.ordering_]
    
    levels = []
    for label in set(labels):
        if label == -1:  # Noise
            continue
        
        cluster_prices = all_prices[labels == label].flatten()
        center = np.median(cluster_prices)
        
        # Density = inverse of reachability
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) == 0:
            continue
        
        # Map cluster indices to ordering indices for reachability
        ordering_map = {optics.ordering_[i]: i for i in range(len(optics.ordering_))}
        cluster_reachability = [reachability[ordering_map.get(idx, 0)] for idx in cluster_indices if idx in ordering_map]
        
        if len(cluster_reachability) == 0:
            continue
        
        density = 1.0 / (np.mean(cluster_reachability) + 1e-9)
        
        strength = min(density / 100, 0.9)  # Normalize
        
        levels.append({
            'price': float(center),
            'type': 'OPTICS Multi-Density',
            'touches': len(cluster_prices),
            'strength': float(strength),
            'breakoutProb': float(1 - strength),
            'reversionProb': float(strength),
            'category': 'OPTICS',
            'density_score': float(density)
        })
    
    return sorted(levels, key=lambda x: x.get('density_score', 0), reverse=True)[:8]

def calculate_hdbscan_levels(highs, lows, closes, timeframe='1d'):
    """
    HDBSCAN: State-of-the-art density clustering
    Automatically finds optimal structure without parameters
    
    CRITICAL: Clusters on RAW PRICES to ensure output is in price space.
    HDBSCAN handles scale differences internally via its distance metric.
    """
    if len(closes) < 20:
        print("HDBSCAN: Insufficient data (< 20 points)")
        return []
    
    # CRITICAL FIX: Cluster on RAW PRICES directly
    # This guarantees output is in price space (no coordinate transform bugs)
    all_prices = np.concatenate([highs, lows, closes])
    prices_array = all_prices.reshape(-1, 1)  # HDBSCAN expects 2D array
    
    # Adaptive parameters based on data size and timeframe
    n_samples = len(prices_array)
    if 'm' in timeframe.lower() or 'min' in timeframe.lower():
        # Intraday: Lower thresholds
        min_cluster_size = max(3, min(8, n_samples // 20))
        min_samples = max(2, min_cluster_size // 2)
    elif 'h' in timeframe.lower() or 'hour' in timeframe.lower():
        # Hourly: Medium thresholds
        min_cluster_size = max(5, min(10, n_samples // 15))
        min_samples = max(3, min_cluster_size // 2)
    else:
        # Daily+: Standard thresholds
        min_cluster_size = max(8, min(15, n_samples // 10))
        min_samples = max(5, min_cluster_size // 2)
    
    print(f"HDBSCAN: Clustering {n_samples} raw price points with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    print(f"HDBSCAN: Price range: ${all_prices.min():.2f} - ${all_prices.max():.2f}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass - better for density
    )
    
    clusterer.fit(prices_array)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"HDBSCAN: Found {n_clusters} clusters, {n_noise} noise points")
    
    levels = []
    for label in unique_labels:
        if label == -1:  # Noise
            continue
        
        cluster_mask = labels == label
        
        if np.sum(cluster_mask) == 0:
            continue
        
        # Get cluster prices (RAW PRICES - already in correct space)
        cluster_prices = all_prices[cluster_mask]
        cluster_probs = probabilities[cluster_mask]
        
        # Weighted average of RAW PRICES (guaranteed to be in price space)
        center = np.average(cluster_prices, weights=cluster_probs)
        
        # Diagnostic log
        print(f"HDBSCAN Cluster {label}: center=${center:.2f}, min=${cluster_prices.min():.2f}, max=${cluster_prices.max():.2f}, size={len(cluster_prices)}")
        
        # Strength from average membership probability
        strength = np.mean(cluster_probs) if len(cluster_probs) > 0 else 0.5
        
        # Touches - count how many actual prices are near this level
        price_range = all_prices.max() - all_prices.min()
        price_tolerance = price_range * 0.01  # 1% of price range
        touches = np.sum(np.abs(all_prices - center) < price_tolerance)
        
        # Ensure we have valid price
        if not isinstance(center, (int, float)) or np.isnan(center) or np.isinf(center):
            print(f"HDBSCAN: Skipping invalid price: {center}")
            continue
        
        # Final validation: center must be in reasonable price range
        if center < all_prices.min() * 0.5 or center > all_prices.max() * 1.5:
            print(f"HDBSCAN: Skipping out-of-range price: ${center:.2f} (range: ${all_prices.min():.2f}-${all_prices.max():.2f})")
            continue
        
        levels.append({
            'price': float(center),  # RAW PRICE - guaranteed correct
            'type': 'HDBSCAN Cluster',
            'touches': int(touches),
            'strength': float(min(max(strength, 0.1), 0.93)),  # Clamp between 0.1 and 0.93
            'breakoutProb': float(1 - min(max(strength, 0.1), 0.93)),
            'reversionProb': float(min(max(strength, 0.1), 0.93)),
            'category': 'Density (HDBSCAN)',  # Explicit structural level category
            'source': 'HDBSCAN',  # Track original source
            'avg_membership': float(strength),
            'cluster_size': int(np.sum(cluster_mask))
        })
    
    result = sorted(levels, key=lambda x: x.get('avg_membership', 0), reverse=True)[:8]
    price_list = [f"${l['price']:.2f}" for l in result]
    print(f"HDBSCAN: Returning {len(result)} levels with prices: {price_list}")
    return result

def enhanced_optics_levels(highs, lows, closes, timeframe='1d'):
    """
    OPTICS with reachability-based strength scoring
    Reachability distance = "how dense is this cluster?"
    Better than HDBSCAN for some patterns
    """
    if len(closes) < 20:
        return []
    
    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)
    
    optics = OPTICS(
        min_samples=5,
        xi=0.05,
        min_cluster_size=10,
        metric='euclidean'
    )
    
    labels = optics.fit_predict(all_prices)
    reachability = optics.reachability_[optics.ordering_]
    
    levels = []
    for label in set(labels):
        if label == -1:
            continue
        
        cluster_mask = labels == label
        cluster_prices = all_prices[cluster_mask].flatten()
        center = np.median(cluster_prices)
        
        # NEW: Use reachability distance for strength
        # Lower reachability = denser cluster = stronger level
        cluster_indices = np.where(cluster_mask)[0]
        ordering_map = {optics.ordering_[i]: i for i in range(len(optics.ordering_))}
        cluster_reachability = [reachability[ordering_map.get(idx, 0)]
                               for idx in cluster_indices if idx in ordering_map]
        
        if len(cluster_reachability) == 0:
            continue
        
        avg_reachability = np.mean(cluster_reachability)
        
        # Inverse relationship: lower reach = higher strength
        # Normalize by price scale
        price_scale = np.ptp(all_prices)
        normalized_reach = avg_reachability / (price_scale + 1e-9)
        
        # Convert to strength [0, 1]
        strength = 1.0 / (1.0 + normalized_reach * 10)  # Sigmoid-like
        
        # NEW: Valley depth metric
        # How "deep" is the valley in reachability plot?
        ordering_positions = [ordering_map.get(idx, 0) for idx in cluster_indices if idx in ordering_map]
        if len(ordering_positions) > 0:
            cluster_reach_vals = reachability[ordering_positions]
            local_min_reach = np.min(cluster_reach_vals)
            start_idx = max(0, min(ordering_positions) - 5)
            end_idx = min(len(reachability), max(ordering_positions) + 5)
            surrounding_reach = np.mean(reachability[start_idx:end_idx])
            valley_depth = (surrounding_reach - local_min_reach) / (surrounding_reach + 1e-9)
            
            # Boost strength for deep valleys
            strength *= (1.0 + 0.5 * valley_depth)
            strength = min(strength, 0.95)
        else:
            valley_depth = 0.0
        
        levels.append({
            'price': float(center),
            'type': 'OPTICS Density Valley',
            'strength': float(strength),
            'touches': len(cluster_prices),
            'avg_reachability': float(avg_reachability),
            'valley_depth': float(valley_depth),
            'category': 'OPTICS',
            'breakoutProb': float(1 - strength),
            'reversionProb': float(strength)
        })
    
    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:8]

def kde_based_levels(highs, lows, closes, n_levels=10):
    """
    Find levels using kernel density estimation
    Peaks in density = strong levels
    """
    all_prices = np.concatenate([highs, lows, closes])
    
    # Adaptive bandwidth (Scott's rule)
    kde = gaussian_kde(all_prices, bw_method='scott')
    
    # Evaluate KDE on fine grid
    price_range = np.ptp(all_prices)
    grid = np.linspace(all_prices.min(), all_prices.max(), 1000)
    density = kde(grid)
    
    # Find local maxima (peaks in density)
    peak_indices = argrelextrema(density, np.greater, order=5)[0]
    
    levels = []
    for idx in peak_indices:
        price = grid[idx]
        density_value = density[idx]
        
        # Strength from relative density
        strength = density_value / np.max(density)
        
        # Count touches (prices within ±0.5% of this level)
        touches = np.sum(np.abs(all_prices - price) < price * 0.005)
        
        # Prominence: how much does density drop around this peak?
        left_valley = np.min(density[max(0, idx-20):idx]) if idx > 20 else 0
        right_valley = np.min(density[idx:min(len(density), idx+20)]) if idx < len(density)-20 else 0
        avg_valley = (left_valley + right_valley) / 2
        prominence = (density_value - avg_valley) / (density_value + 1e-9)
        
        # Boost strength by prominence
        strength *= (1.0 + prominence)
        strength = min(strength, 0.95)
        
        levels.append({
            'price': float(price),
            'type': 'KDE Peak',
            'strength': float(strength),
            'touches': int(touches),
            'density': float(density_value),
            'prominence': float(prominence),
            'category': 'KDE',
            'breakoutProb': float(1 - strength),
            'reversionProb': float(strength)
        })
    
    # Sort by strength and return top N
    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:n_levels]

def multiscale_hdbscan_levels(highs, lows, closes, timeframe='1d'):
    """
    Run HDBSCAN at multiple scales to catch both major and minor levels
    """
    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)
    
    # Different scales
    scales = [
        {'min_cluster_size': 5, 'min_samples': 3, 'name': 'micro'},
        {'min_cluster_size': 10, 'min_samples': 5, 'name': 'meso'},
        {'min_cluster_size': 20, 'min_samples': 10, 'name': 'macro'}
    ]
    
    all_levels = []
    
    for scale in scales:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=scale['min_cluster_size'],
            min_samples=scale['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(all_prices)
        probabilities = clusterer.probabilities_
        
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_mask = labels == label
            cluster_prices = all_prices[cluster_mask].flatten()
            cluster_probs = probabilities[cluster_mask]
            
            center = np.average(cluster_prices, weights=cluster_probs)
            strength = np.mean(cluster_probs)
            
            # Boost strength for larger scales
            scale_factor = {'micro': 0.8, 'meso': 1.0, 'macro': 1.2}[scale['name']]
            strength *= scale_factor
            
            all_levels.append({
                'price': float(center),
                'type': f'HDBSCAN-{scale["name"]}',
                'strength': float(min(strength, 0.95)),
                'scale': scale['name'],
                'cluster_size': int(np.sum(cluster_mask)),
                'category': 'HDBSCAN-MultiScale',
                'breakoutProb': float(1 - min(strength, 0.95)),
                'reversionProb': float(min(strength, 0.95))
            })
    
    # Hierarchical merge across scales
    if len(all_levels) > 1:
        prices = np.array([l['price'] for l in all_levels]).reshape(-1, 1)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=np.std(prices) * 0.05,
            linkage='ward'
        )
        labels = clustering.fit_predict(prices)
        
        merged = []
        for label in set(labels):
            cluster = [l for i, l in enumerate(all_levels) if labels[i] == label]
            
            # Weighted average by strength
            strengths = np.array([l['strength'] for l in cluster])
            prices_cluster = np.array([l['price'] for l in cluster])
            
            avg_price = np.average(prices_cluster, weights=strengths)
            avg_strength = np.mean(strengths)
            
            # Boost if multiple scales agree
            scale_agreement = len(set(l['scale'] for l in cluster))
            if scale_agreement >= 2:
                avg_strength *= 1.15
            
            merged.append({
                'price': float(avg_price),
                'type': 'HDBSCAN-MultiScale',
                'strength': float(min(avg_strength, 0.95)),
                'scales_detected': [l['scale'] for l in cluster],
                'scale_agreement': scale_agreement,
                'category': 'HDBSCAN-MultiScale',
                'breakoutProb': float(1 - min(avg_strength, 0.95)),
                'reversionProb': float(min(avg_strength, 0.95))
            })
        
        return sorted(merged, key=lambda x: x['strength'], reverse=True)[:8]
    
    return sorted(all_levels, key=lambda x: x['strength'], reverse=True)[:8]

def time_weighted_hdbscan(highs, lows, closes, timestamps, half_life_days=30):
    """
    Weight recent price action more heavily
    Levels from 6 months ago are less relevant than last week's levels
    """
    all_prices = np.concatenate([highs, lows, closes])
    all_times = np.concatenate([timestamps, timestamps, timestamps])
    
    # Convert timestamps to datetime if needed
    if isinstance(all_times[0], (int, float)):
        all_times = pd.to_datetime(all_times, unit='s')
    
    # Calculate time weights (exponential decay)
    current_time = pd.to_datetime(timestamps[-1]) if isinstance(timestamps[-1], (int, float)) else timestamps[-1]
    time_diffs = np.array([(current_time - pd.to_datetime(t) if isinstance(t, (int, float)) else current_time - t).days for t in all_times])
    weights = np.exp(-time_diffs / half_life_days)
    
    # Weighted sampling (sample recent prices more)
    n_samples = len(all_prices)
    if n_samples > 0 and weights.sum() > 0:
        sample_indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True,
            p=weights / weights.sum()
        )
        
        sampled_prices = all_prices[sample_indices].reshape(-1, 1)
        
        # Run HDBSCAN on weighted sample
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(sampled_prices)
        probabilities = clusterer.probabilities_
        
        levels = []
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_mask = labels == label
            cluster_prices = sampled_prices[cluster_mask].flatten()
            cluster_probs = probabilities[cluster_mask]
            
            center = np.average(cluster_prices, weights=cluster_probs)
            strength = np.mean(cluster_probs)
            
            levels.append({
                'price': float(center),
                'type': 'Time-Weighted HDBSCAN',
                'strength': float(min(strength, 0.95)),
                'touches': len(cluster_prices),
                'category': 'HDBSCAN-TimeWeighted',
                'breakoutProb': float(1 - min(strength, 0.95)),
                'reversionProb': float(min(strength, 0.95))
            })
        
        return sorted(levels, key=lambda x: x['strength'], reverse=True)[:8]
    
    return []

def detect_wyckoff_zones(hist, lookback=50):
    """
    Detect Wyckoff accumulation (support) and distribution (resistance) zones

    Accumulation signs:
    - Price consolidates after downtrend
    - Volume increases on up-bars
    - Springs/shakeouts below support

    Distribution signs:
    - Price consolidates after uptrend
    - Volume increases on down-bars
    - Upthrusts above resistance
    """
    if len(hist) < lookback:
        return []
    
    recent = hist.tail(lookback) if hasattr(hist, 'tail') else hist[-lookback:]
    
    if isinstance(recent, pd.DataFrame):
        closes = recent['Close'].values if 'Close' in recent.columns else recent.iloc[:, -1].values
        highs = recent['High'].values if 'High' in recent.columns else recent.iloc[:, 1].values
        lows = recent['Low'].values if 'Low' in recent.columns else recent.iloc[:, 2].values
        volumes = recent['Volume'].values if 'Volume' in recent.columns else np.ones(len(recent))
    else:
        closes = np.array([c['Close'] if isinstance(c, dict) else c[-1] for c in recent])
        highs = np.array([h['High'] if isinstance(h, dict) else h[1] for h in recent])
        lows = np.array([l['Low'] if isinstance(l, dict) else l[2] for l in recent])
        volumes = np.array([v.get('Volume', 1.0) if isinstance(v, dict) else 1.0 for v in recent])
    
    levels = []
    
    # Detect consolidation zones (low volatility)
    rolling_std = pd.Series(closes).rolling(10).std()
    low_vol_periods = rolling_std < rolling_std.quantile(0.3)
    
    # Find contiguous low-vol zones
    zones = []
    in_zone = False
    zone_start = 0
    
    for i, is_low_vol in enumerate(low_vol_periods):
        if is_low_vol and not in_zone:
            zone_start = i
            in_zone = True
        elif not is_low_vol and in_zone:
            if i - zone_start >= 5:  # Minimum 5 bars
                zones.append((zone_start, i))
            in_zone = False
    
    for start, end in zones:
        zone_closes = closes[start:end]
        zone_highs = highs[start:end]
        zone_lows = lows[start:end]
        zone_volumes = volumes[start:end]
        
        # Zone characteristics
        zone_mid = (np.max(zone_highs) + np.min(zone_lows)) / 2
        zone_width = np.max(zone_highs) - np.min(zone_lows)
        
        # Check for accumulation/distribution
        # Accumulation: up-bars have higher volume
        up_bars = zone_closes[1:] > zone_closes[:-1]
        down_bars = ~up_bars
        
        up_vol = np.mean(zone_volumes[1:][up_bars]) if np.any(up_bars) else 0
        down_vol = np.mean(zone_volumes[1:][down_bars]) if np.any(down_bars) else 0
        
        if up_vol > down_vol * 1.2:
            zone_type = 'Wyckoff Accumulation'
            strength = 0.80
        elif down_vol > up_vol * 1.2:
            zone_type = 'Wyckoff Distribution'
            strength = 0.80
        else:
            zone_type = 'Wyckoff Consolidation'
            strength = 0.65
        
        levels.append({
            'price': float(zone_mid),
            'type': zone_type,
            'strength': float(strength),
            'touches': len(zone_closes),
            'zone_width': float(zone_width),
            'category': 'Wyckoff',
            'breakoutProb': float(1 - strength),
            'reversionProb': float(strength)
        })
    
    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:8]

def persistent_homology_levels(highs, lows, closes, max_levels=8):
    """
    Use persistent homology to find levels that exist across multiple scales
    
    Intuition: 
    - Build filtration at different price thresholds
    - Levels that persist = structurally significant
    """
    if not RIPSER_AVAILABLE:
        return []
    
    if len(closes) < 20:
        return []
    
    all_prices = np.concatenate([highs, lows, closes])
    
    # Build point cloud (1D)
    points = all_prices.reshape(-1, 1)
    
    try:
        # Compute persistent homology
        if not RIPSER_AVAILABLE or ripser is None:
            return None  # Skip topological analysis if ripser not available
        result = ripser(points, maxdim=0)
        diagrams = result['dgms']
        
        # 0-dimensional persistence (connected components)
        # Each bar in diagram = (birth, death) of a component
        h0_diagram = diagrams[0]
        
        levels = []
        for birth, death in h0_diagram:
            if np.isinf(death):
                continue
            
            # Persistence = how long this level "lived"
            persistence = death - birth
            
            # Level price = midpoint of birth/death
            level_price = (birth + death) / 2
            
            # Strength from persistence
            finite_deaths = h0_diagram[~np.isinf(h0_diagram[:, 1])]
            if len(finite_deaths) > 0:
                max_persistence = np.max(finite_deaths[:, 1] - finite_deaths[:, 0])
                strength = persistence / (max_persistence + 1e-9)
            else:
                strength = 0.5
            
            # Filter weak levels
            if strength < 0.3:
                continue
            
            # Count touches
            touches = np.sum(np.abs(all_prices - level_price) < level_price * 0.005)
            
            levels.append({
                'price': float(level_price),
                'type': 'Persistent Homology',
                'strength': float(strength),
                'persistence': float(persistence),
                'birth': float(birth),
                'death': float(death),
                'touches': int(touches),
                'category': 'TDA',
                'breakoutProb': float(1 - strength),
                'reversionProb': float(strength)
            })
        
        return sorted(levels, key=lambda x: x['persistence'], reverse=True)[:max_levels]
    except Exception as e:
        print(f"Persistent Homology failed: {e}")
        return []

if TORCH_AVAILABLE and nn is not None:
    class LevelDetectionNet(nn.Module):
        """
        CNN + BiLSTM + Attention level detector
        - Input 1: OHLC sequence [batch, seq_len, 4]
        - Input 2: Volume-profile features [batch, seq_len, 5] (optional)
        - Output: logits [batch, seq_len] (per-bar: level vs no-level)
        """
        def __init__(self, lookback=100, use_volume_profile=True,
                     cnn_channels=(64, 128), lstm_hidden=64, lstm_layers=1,
                     attn_heads=4, dropout=0.2):
            super().__init__()
            
            self.lookback = lookback
            self.use_volume_profile = use_volume_profile
            
            ohlc_dim = 4
            vol_dim = 5 if use_volume_profile else 0
            input_dim = ohlc_dim + vol_dim
            
            # 1) Project inputs
            self.input_proj = nn.Linear(input_dim, cnn_channels[0])
            
            # 2) CNN stack over time dimension (Conv1d expects [B, C, T])
            self.conv1 = nn.Conv1d(cnn_channels[0], cnn_channels[0], kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2)
            
            # 3) BiLSTM
            self.bilstm = nn.LSTM(
                input_size=cnn_channels[1],
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True
            )
            lstm_out_dim = lstm_hidden * 2  # bi-directional
            
            # 4) Multi-head self-attention on top of LSTM output
            self.attn = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # 5) Final classifier per time-step
            self.fc = nn.Linear(lstm_out_dim, 1)
            
            self.dropout = nn.Dropout(dropout)

        def forward(self, ohlc_features, volume_profile_features=None):
            """
            ohlc_features: [batch, seq_len, 4]
            volume_profile_features: [batch, seq_len, 5] or None
            returns: logits [batch, seq_len]
            """
            if self.use_volume_profile:
                if volume_profile_features is None:
                    raise ValueError("Model created with use_volume_profile=True but no volume_profile_features passed")
                x = torch.cat([ohlc_features, volume_profile_features], dim=-1)
            else:
                x = ohlc_features  # [B, T, 4]
            
            # Input projection
            x = self.input_proj(x)  # [B, T, C]
            x = torch.relu(x)
            x = self.dropout(x)
            
            # CNN over time
            x = x.transpose(1, 2)  # [B, C, T]
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.dropout(x)
            x = x.transpose(1, 2)  # [B, T, C_cnn]
            
            # BiLSTM
            x, _ = self.bilstm(x)  # [B, T, 2*hidden]
            x = self.dropout(x)
            
            # Self-attention
            attn_out, attn_weights = self.attn(x, x, x)  # [B, T, D], [B, T, T]
            x = x + attn_out  # residual
            x = self.dropout(x)
            
            # Per-time-step logits
            logits = self.fc(x).squeeze(-1)  # [B, T]
            return logits, attn_weights
else:
    # Dummy class when torch is not available
    class LevelDetectionNet:
        def __init__(self, *args, **kwargs):
            pass

def train_level_detection_network(ticker='SPY', timeframe='1d',
                                  lookback=100, epochs=50, batch_size=32):
    """
    Train the neural network level detector using:
    - HDBSCAN + OPTICS levels as pseudo-ground-truth
    - OHLC + volume-profile features
    """
    if not TORCH_AVAILABLE:
        return {'success': False, 'error': 'PyTorch not available'}
    
    try:
        print(f"Training level detection network for {ticker} at {timeframe}...")
        
        # 1) Fetch data
        stock = yf.Ticker(ticker)
        interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
        interval = interval_map.get(timeframe, '1d')
        period_map = {'1m': '1mo', '5m': '3mo', '15m': '6mo', '1h': '1y', '4h': '1y', '1d': '2y'}
        period = period_map.get(timeframe, '1y')
        
        hist = stock.history(period=period, interval=interval)
        if timeframe == '4h':
            # resample 1h to 4h
            if not isinstance(hist.index, pd.DatetimeIndex):
                hist.index = pd.to_datetime(hist.index)
            hist = hist.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        if len(hist) < lookback * 2:
            return {'success': False, 'error': f'Insufficient data: need at least {lookback * 2} bars, got {len(hist)}'}
        
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        volumes = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(closes))
        
        # 2) Generate windows
        print("Generating training samples...")
        X_train = []
        X_volume = []
        y_train = []
        
        for i in range(lookback, len(hist) - 10):
            window_hist = hist.iloc[i-lookback:i]
            window_highs = highs[i-lookback:i]
            window_lows = lows[i-lookback:i]
            window_closes = closes[i-lookback:i]
            window_volumes = volumes[i-lookback:i] if len(volumes) >= i else np.ones(lookback)
            
            # --- Volume profile for this window ---
            volume_profile = calculate_volume_profile(
                window_highs, window_lows, window_closes, window_volumes, bins=30
            )
            
            # --- Ground truth levels (HDBSCAN + OPTICS) ---
            hdbscan_levels = calculate_hdbscan_levels(
                window_highs, window_lows, window_closes, timeframe=timeframe
            )
            optics_levels = enhanced_optics_levels(
                window_highs, window_lows, window_closes, timeframe=timeframe
            )
            all_truth_levels = hdbscan_levels + optics_levels
            truth_prices = [l.get('price', 0) for l in all_truth_levels if 'price' in l]
            
            # --- OHLC normalization ---
            ohlc_data = window_hist[['Open', 'High', 'Low', 'Close']].values
            ohlc_mean = ohlc_data.mean(axis=0)
            ohlc_std = ohlc_data.std(axis=0) + 1e-9
            ohlc_normalized = (ohlc_data - ohlc_mean) / ohlc_std  # [lookback, 4]
            
            # --- Volume-profile per-bar features ---
            volume_features = []
            if volume_profile:
                poc = volume_profile.get('poc', np.mean(window_closes))
                va_high = volume_profile.get('value_area_high', np.max(window_closes))
                va_low = volume_profile.get('value_area_low', np.min(window_closes))
                volume_dist = volume_profile.get('volume_distribution', {})
                std_close = np.std(window_closes) + 1e-9
                max_vol = max(volume_dist.values()) if volume_dist else 1.0
                
                for close_price in window_closes:
                    dist_to_poc = abs(close_price - poc) / std_close
                    dist_to_va_high = abs(close_price - va_high) / std_close
                    dist_to_va_low = abs(close_price - va_low) / std_close
                    in_value_area = 1.0 if va_low <= close_price <= va_high else 0.0
                    
                    if volume_dist:
                        closest_price = min(volume_dist.keys(), key=lambda p: abs(p - close_price))
                        if abs(closest_price - close_price) / close_price < 0.01:
                            volume_at_price = volume_dist.get(closest_price, 0.0)
                            volume_at_price_norm = volume_at_price / (max_vol + 1e-9)
                        else:
                            volume_at_price_norm = 0.0
                    else:
                        volume_at_price_norm = 0.0
                    
                    volume_features.append([
                        float(dist_to_poc),
                        float(dist_to_va_high),
                        float(dist_to_va_low),
                        float(in_value_area),
                        float(volume_at_price_norm)
                    ])
            else:
                volume_features = [[0.0, 0.0, 0.0, 0.0, 0.0]] * lookback
            
            volume_features = np.array(volume_features, dtype=np.float32)  # [lookback, 5]
            
            # --- Labels (bars near any level) ---
            labels = np.zeros(lookback, dtype=np.float32)
            price_tolerance = np.std(window_closes) * 0.01  # 1% of std-dev
            
            for j, close_price in enumerate(window_closes):
                for truth_price in truth_prices:
                    if abs(close_price - truth_price) < price_tolerance:
                        labels[j] = 1.0
                        break
            
            # keep only windows with at least 1 positive
            if labels.sum() > 0:
                X_train.append(ohlc_normalized)
                X_volume.append(volume_features)
                y_train.append(labels)
        
        if len(X_train) == 0:
            return {'success': False, 'error': 'No training samples generated'}
        
        X_train = np.array(X_train, dtype=np.float32)      # [N, T, 4]
        X_volume = np.array(X_volume, dtype=np.float32)    # [N, T, 5]
        y_train = np.array(y_train, dtype=np.float32)      # [N, T]
        
        print(f"Generated {len(X_train)} training samples")
        print(f"Positive label rate: {np.mean(y_train):.2%}")
        
        # 3) Train/val split
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train[:split_idx]
        X_volume_train = X_volume[:split_idx]
        y_train_split = y_train[:split_idx]
        
        X_val = X_train[split_idx:]
        X_volume_val = X_volume[split_idx:]
        y_val = y_train[split_idx:]
        
        # 4) Tensors
        X_train_tensor = torch.FloatTensor(X_train_split)
        X_volume_train_tensor = torch.FloatTensor(X_volume_train)
        y_train_tensor = torch.FloatTensor(y_train_split)
        
        X_val_tensor = torch.FloatTensor(X_val)
        X_volume_val_tensor = torch.FloatTensor(X_volume_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 5) Model, optimizer, loss
        model = LevelDetectionNet(
            lookback=lookback,
            use_volume_profile=True,
            cnn_channels=(64, 128),
            lstm_hidden=64,
            lstm_layers=1,
            attn_heads=4,
            dropout=0.2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            # mini-batch training
            for batch_start in range(0, len(X_train_tensor), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train_tensor))
                
                X_batch = X_train_tensor[batch_start:batch_end]           # [B, T, 4]
                X_vol_batch = X_volume_train_tensor[batch_start:batch_end]# [B, T, 5]
                y_batch = y_train_tensor[batch_start:batch_end]           # [B, T]
                
                optimizer.zero_grad()
                logits, _ = model(X_batch, volume_profile_features=X_vol_batch)  # [B, T]
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / max(1, (len(X_train_tensor) / batch_size))
            train_losses.append(avg_train_loss)
            
            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val_tensor, volume_profile_features=X_volume_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_losses.append(val_loss)
                
                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'level_detector.pth')
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        print(f"[OK] Training complete. Best val loss: {best_val_loss:.4f}")
        print("[OK] Model saved to level_detector.pth")
        
        # Final metrics on val
        model.eval()
        with torch.no_grad():
            final_logits, _ = model(X_val_tensor, volume_profile_features=X_volume_val_tensor)
            final_probs = torch.sigmoid(final_logits)
            final_preds = (final_probs > 0.5).float()
            
            final_acc = (final_preds == y_val_tensor).float().mean().item()
            
            tp = ((final_preds == 1) & (y_val_tensor == 1)).float().sum().item()
            pp = (final_preds == 1).float().sum().item()
            ap = (y_val_tensor == 1).float().sum().item()
            
            precision = tp / (pp + 1e-9)
            recall = tp / (ap + 1e-9)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        return {
            'success': True,
            'model_path': 'level_detector.pth',
            'metrics': {
                'final_accuracy': float(final_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'best_val_loss': float(best_val_loss),
                'train_samples': int(len(X_train_split)),
                'val_samples': int(len(X_val))
            },
            'training_history': {
                'train_losses': [float(x) for x in train_losses],
                'val_losses': [float(x) for x in val_losses]
            }
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Training failed:\n{error_trace}")
        return {'success': False, 'error': str(e)}

def detect_levels_with_neural_network(hist, lookback=100, threshold=0.7):
    """
    Use trained neural network to detect levels
    Falls back to simple pattern detection if model not available
    """
    if not TORCH_AVAILABLE or len(hist) < lookback:
        return []
    
    try:
        # Prepare OHLC data
        ohlc_data = hist[['Open', 'High', 'Low', 'Close']].values[-lookback:]
        closes = hist['Close'].values[-lookback:]
        volumes = hist['Volume'].values[-lookback:] if 'Volume' in hist.columns else np.ones(lookback)
        highs = hist['High'].values[-lookback:]
        lows = hist['Low'].values[-lookback:]
        
        # Calculate volume profile
        volume_profile = calculate_volume_profile(highs, lows, closes, volumes, bins=30)
        
        # Prepare volume profile features
        volume_features = []
        if volume_profile:
            poc = volume_profile.get('poc', np.mean(closes))
            va_high = volume_profile.get('value_area_high', np.max(closes))
            va_low = volume_profile.get('value_area_low', np.min(closes))
            volume_dist = volume_profile.get('volume_distribution', {})
            
            for j, close_price in enumerate(closes):
                dist_to_poc = abs(close_price - poc) / (np.std(closes) + 1e-9)
                dist_to_va_high = abs(close_price - va_high) / (np.std(closes) + 1e-9)
                dist_to_va_low = abs(close_price - va_low) / (np.std(closes) + 1e-9)
                in_value_area = 1.0 if va_low <= close_price <= va_high else 0.0
                
                closest_price = min(volume_dist.keys(), key=lambda p: abs(p - close_price)) if volume_dist else poc
                if abs(closest_price - close_price) / close_price < 0.01:
                    volume_at_price = volume_dist.get(closest_price, 0.0)
                    max_vol = max(volume_dist.values()) if volume_dist else 1.0
                    volume_at_price_norm = volume_at_price / (max_vol + 1e-9)
                else:
                    volume_at_price_norm = 0.0
                
                volume_features.append([
                    float(dist_to_poc),
                    float(dist_to_va_high),
                    float(dist_to_va_low),
                    float(in_value_area),
                    float(volume_at_price_norm)
                ])
        else:
            volume_features = [[0.0, 0.0, 0.0, 0.0, 0.0]] * lookback
        
        volume_features = np.array(volume_features, dtype=np.float32)
        
        # Normalize OHLC
        ohlc_mean = ohlc_data.mean(axis=0)
        ohlc_std = ohlc_data.std(axis=0) + 1e-9
        ohlc_normalized = (ohlc_data - ohlc_mean) / ohlc_std
        
        # Convert to tensor
        ohlc_tensor = torch.FloatTensor(ohlc_normalized).unsqueeze(0)  # [1, lookback, 4]
        volume_tensor = torch.FloatTensor(volume_features).unsqueeze(0)  # [1, lookback, 5]
        
        # Try to load pre-trained model, otherwise use pattern-based fallback
        model_path = 'level_detector.pth'
        try:
            model = LevelDetectionNet(lookback=lookback, use_volume_profile=True)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                with torch.no_grad():
                    level_logits, _ = model(ohlc_tensor, volume_profile_features=volume_tensor)
                    # Apply sigmoid to get probabilities
                    level_probs = torch.sigmoid(level_logits)
                    # Extract levels where probability > threshold
                    level_indices = (level_probs[0] > threshold).nonzero(as_tuple=False).flatten()
                    levels = []
                    for idx in level_indices:
                        idx_val = idx.item()
                        if idx_val < len(closes):
                            price = float(closes[idx_val])
                            prob = float(level_probs[0][idx_val])
                            levels.append({
                                'price': price,
                                'type': 'Neural Network',
                                'strength': prob,
                                'category': 'Neural-Network',
                                'breakoutProb': float(1 - prob),
                                'reversionProb': prob,
                                'touches': 1
                            })
                    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:10]
            else:
                print(f"⚠ Neural network model not found at {model_path}. Use /api/train-level-detector to train it.")
        except Exception as model_error:
            print(f"Could not load neural network model: {model_error}, using fallback")
        
        # Fallback: detect local extrema as potential levels
        from scipy.signal import argrelextrema
        high_indices = argrelextrema(hist['High'].values[-lookback:], np.greater, order=5)[0]
        low_indices = argrelextrema(hist['Low'].values[-lookback:], np.less, order=5)[0]
        
        levels = []
        for idx in high_indices:
            price = float(hist['High'].iloc[-lookback + idx])
            levels.append({
                'price': price,
                'type': 'Neural Network (Local High)',
                'strength': 0.65,
                'category': 'Neural-Network',
                'breakoutProb': 0.35,
                'reversionProb': 0.65,
                'touches': 1
            })
        
        for idx in low_indices:
            price = float(hist['Low'].iloc[-lookback + idx])
            levels.append({
                'price': price,
                'type': 'Neural Network (Local Low)',
                'strength': 0.65,
                'category': 'Neural-Network',
                'breakoutProb': 0.35,
                'reversionProb': 0.65,
                'touches': 1
            })
        
        # Remove duplicates and sort
        seen_prices = set()
        unique_levels = []
        for l in levels:
            price_key = round(l['price'], 2)
            if price_key not in seen_prices:
                seen_prices.add(price_key)
                unique_levels.append(l)
        
        return sorted(unique_levels, key=lambda x: x['strength'], reverse=True)[:10]
        
    except Exception as e:
        print(f"Neural Network level detection failed: {e}")
        return []

def detect_levels_with_deepsupp(hist, model_path='deepsupp_v4.pt', device='cpu'):
    """
    DeepSupp v4 level detector (corr-series transformer autoencoder).

    Requires a pre-trained model file saved via deepsupp_levels.save_deepsupp_model().
    If the model file is missing or DeepSupp deps fail to load, returns [].
    """
    if not TORCH_AVAILABLE or hist is None or len(hist) < 60:
        return []

    try:
        if not os.path.exists(model_path):
            print(f"⚠ DeepSupp model file '{model_path}' not found. Attempting auto-train...")
            if hist is not None and len(hist) >= 400:
                try:
                    train_result = train_deepsupp_level_model(
                        ticker="SPY", timeframe="1d",
                        epochs=15, batch_size=32, model_path=model_path
                    )
                    if not train_result.get('success'):
                        print(f"⚠ DeepSupp auto-train failed: {train_result.get('error', 'unknown')}")
                        return []
                    print(f"✓ DeepSupp auto-trained and saved to {model_path}")
                except Exception as train_err:
                    print(f"⚠ DeepSupp auto-train exception: {train_err}")
                    return []
            else:
                print(f"⚠ Not enough data for auto-train ({len(hist) if hist is not None else 0} bars, need 400). Train manually via POST /api/train-deepsupp-levels")
                return []

        from deepsupp_levels import load_deepsupp_model, compute_deepsupp_levels

        # yfinance style columns → deepsupp expects lowercase ohlcv
        df = hist.copy()
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(set(map(str.lower, df.columns))):
            # If rename failed due to unexpected schema
            if not required.issubset(set(df.columns)):
                return []

        model, meta = load_deepsupp_model(model_path, device=device)

        records = compute_deepsupp_levels(
            df[['open', 'high', 'low', 'close', 'volume']].dropna(),
            model,
            vol_lookback=int(meta.vol_lookback),
            corr_window=int(meta.corr_window),
            seq_len=int(meta.seq_len),
            device=device,
            verbose=False
        )

        levels = []
        for r in records:
            strength = float(getattr(r, 'strength', 0.0))
            levels.append({
                'price': float(getattr(r, 'price', np.nan)),
                'type': 'DeepSupp',
                'strength': strength,
                'category': 'DeepSupp',
                'source': 'DeepSupp',
                'kind': str(getattr(r, 'kind', 'level')),
                'touches': int(getattr(r, 'n_members', 1)),
                'coverage': float(getattr(r, 'coverage', 0.0)),
                'quality': float(getattr(r, 'quality', 0.0)),
                'tightness': float(getattr(r, 'tightness', 0.0)),
                'score_mean': float(getattr(r, 'score_mean', 0.0)),
                'score_max': float(getattr(r, 'score_max', 0.0)),
                'price_std': float(getattr(r, 'price_std', 0.0)),
                'displacement': float(getattr(r, 'displacement', 0.0)),
                'cluster_id': int(getattr(r, 'cluster_id', -1)),
                'breakoutProb': float(1 - strength),
                'reversionProb': float(strength),
            })

        levels = [l for l in levels if isinstance(l.get('price'), (int, float)) and not (np.isnan(l.get('price')) or np.isinf(l.get('price')))]
        return sorted(levels, key=lambda x: x.get('strength', 0.0), reverse=True)[:12]

    except Exception as e:
        print(f"DeepSupp level detection failed: {e}")
        return []

def train_deepsupp_level_model(
    ticker: str = "SPY",
    timeframe: str = "1d",
    vol_lookback: int = 20,
    corr_window: int = 20,
    seq_len: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    latent_dim: int = 16,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    model_path: str = "deepsupp_v4.pt",
) -> dict:
    """
    Train a DeepSupp v4 model on real OHLCV data and save to model_path.
    """
    if not TORCH_AVAILABLE:
        return {"success": False, "error": "PyTorch not available"}

    try:
        import yfinance as yf
        import pandas as pd
        from deepsupp_levels import build_and_train, save_deepsupp_model

        stock = yf.Ticker(ticker)

        interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "1h", "1d": "1d"}
        interval = interval_map.get(timeframe, "1d")
        period_map = {"1m": "5d", "5m": "1mo", "15m": "3mo", "30m": "3mo", "1h": "6mo", "4h": "1y", "1d": "2y"}
        period = period_map.get(timeframe, "1y")

        hist = stock.history(period=period, interval=interval)
        if hist is None or len(hist) == 0:
            return {"success": False, "error": f"No historical data for {ticker} at {timeframe}"}

        if timeframe == "4h":
            # resample 1h to 4h, same style as level detector
            if not isinstance(hist.index, pd.DatetimeIndex):
                hist.index = pd.to_datetime(hist.index)
            hist = hist.resample("4H").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

        if len(hist) < 400:
            return {"success": False, "error": f"Insufficient data to train DeepSupp (need >=400 bars, have {len(hist)})"}

        df = hist[["Open", "High", "Low", "Close", "Volume"]].rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }).dropna()

        print(f"Training DeepSupp v4 for {ticker} at {timeframe} on {len(df)} bars...")

        model = build_and_train(
            df,
            vol_lookback=vol_lookback,
            corr_window=corr_window,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            latent_dim=latent_dim,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            verbose=True,
        )

        save_deepsupp_model(model, model_path)

        meta = model.metadata
        meta_dict = meta.to_dict() if meta is not None else {}

        print(f"[OK] DeepSupp training complete. Saved to {model_path}")

        return {
            "success": True,
            "model_path": model_path,
            "meta": meta_dict,
            "training": {
                "ticker": ticker,
                "timeframe": timeframe,
                "epochs": epochs,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "corr_window": corr_window,
                "vol_lookback": vol_lookback,
            },
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"DeepSupp training failed:\n{error_trace}")
        return {"success": False, "error": str(e)}

if TORCH_AVAILABLE and nn is not None:
    class LevelValidator(nn.Module):
        """
        RL agent that learns: "Is this a real level or noise?"
        """
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # [reject, accept]
            )
        
        def forward(self, features):
            return self.net(features)
else:
    # Dummy class when torch is not available
    class LevelValidator:
        def __init__(self, *args, **kwargs):
            pass

def validate_levels_with_rl(levels, current_price, sigma_price):
    """
    Use RL validator to filter out weak/noise levels
    Falls back to simple filtering if model not available
    """
    if not TORCH_AVAILABLE or not levels:
        return levels
    
    try:
        validated_levels = []
        
        # Try to load pre-trained validator model
        try:
            model = LevelValidator(n_features=10)
            model_path = 'level_validator.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
        except Exception as model_error:
            model = None
            print(f"Could not load RL validator model: {model_error}, using rule-based validation")
        
        for level in levels:
            # Extract features for validation
            strength = level.get('strength', 0.5)
            distance_sigma = abs(level.get('price', current_price) - current_price) / (sigma_price + 1e-9)
            touches = level.get('touches', 1)
            category = level.get('category', 'Unknown')
            
            # Features: strength, distance, touches, category
            features = np.array([
                strength,
                min(distance_sigma / 3.0, 1.0),  # Normalized distance
                min(touches / 20.0, 1.0),  # Normalized touches
                1.0 if category in ['Density (HDBSCAN)', 'HDBSCAN', 'OPTICS', 'KDE'] else 0.5,
                1.0 if level.get('reversionProb', 0.5) > 0.6 else 0.5,
                1.0 if level.get('breakoutProb', 0.5) < 0.4 else 0.5,
                1.0 if 'confluence' in str(category).lower() else 0.5,
                1.0 if distance_sigma < 2.0 else 0.5,
                1.0 if touches >= 3 else 0.5,
                1.0 if strength > 0.6 else 0.5
            ])
            
            # Use model if available, otherwise use rule-based scoring
            if model is not None:
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    output = model(features_tensor)
                    action_probs = torch.softmax(output, dim=1)
                    accept_prob = action_probs[0][1].item()
                accept_score = accept_prob
            else:
                # Simple rule-based acceptance
                accept_score = (
                    strength * 0.3 +
                    (1.0 - min(distance_sigma / 3.0, 1.0)) * 0.2 +
                    min(touches / 20.0, 1.0) * 0.2 +
                    (1.0 if category in ['Density (HDBSCAN)', 'HDBSCAN', 'OPTICS', 'KDE'] else 0.5) * 0.15 +
                    level.get('reversionProb', 0.5) * 0.15
                )
            
            # Accept if score > 0.5
            if accept_score > 0.5:
                level['rl_validation_score'] = float(accept_score)
                validated_levels.append(level)
        
        return validated_levels
        
    except Exception as e:
        print(f"RL validation failed: {e}")
        return levels  # Return original levels if validation fails

def calculate_contextual_success_probability(
    level_price,
    level_strength,
    current_price,
    expected_range,
    distance_sigma,
    model_accuracy,
    range_mid=None
):
    """
    Contextual Success Probability: Honest probability composition without boosting.
    
    Answers: "Given where we are in the distribution, how likely is price to respect 
    this level before invalidation?"
    
    This is NOT first-touch probability. This is contextual expectancy based on:
    1. Model accuracy (historical skill)
    2. Position in expected range (geometry)
    3. Distance from current price (sigma proximity)
    
    Rules:
    - Never boost globally
    - Only re-weight based on position in distribution
    - Cap at 0.75 (guardrail)
    - Floor at 0.35 (guardrail)
    
    Returns contextual_success probability (0.35-0.75)
    """
    # Step 1: Model skill prior
    P_model = float(model_accuracy)  # e.g. 0.55-0.65
    
    # Step 2: Distance / range geometry factor
    # Calculate where level sits relative to expected range
    if range_mid is None:
        range_mid = current_price  # Fallback if not provided
    
    range_half = expected_range / 2.0 if expected_range > 0 else abs(level_price - current_price)
    range_pos = abs(level_price - range_mid) / range_half if range_half > 0 else 0.5
    
    # Interpretation:
    # range_pos ≈ 0 → mid-range (bad for reversals)
    # range_pos ≈ 1 → range extreme (good)
    # range_pos > 1 → outside expected range (very good)
    
    # Convert to multiplier (this is the justified "pump")
    range_multiplier = np.clip(0.7 + 0.6 * range_pos, 0.7, 1.35)
    
    # Step 3: Sigma proximity (soft, not binary)
    # Closer = better, but not everything
    sigma_factor = np.clip(1.0 - 0.15 * distance_sigma, 0.6, 1.0)
    
    # Step 4: Contextual probability
    P_context = P_model * range_multiplier * sigma_factor
    
    # Hard guardrails: never exceed reasonable bounds
    P_context = np.clip(P_context, 0.35, 0.75)
    
    return float(P_context)

def get_model_accuracy_by_category(category, source=None):
    """
    Model accuracy (historical skill) by level category.
    
    These are model-conditional accuracies based on historical performance.
    In production, these should be back-filled from realized outcomes.
    
    Returns accuracy between 0.45-0.65 (conservative, honest)
    """
    # Map category/source to historical accuracy
    if category == 'Density (HDBSCAN)' or source == 'HDBSCAN' or category == 'HDBSCAN':
        return 0.62  # Structural levels: highest accuracy
    elif category == 'Interaction' or source == 'Local Density':
        return 0.55  # Local interaction: moderate accuracy
    elif category == 'Isolation-Forest' or source == 'Isolation Forest':
        return 0.48  # Event pivots: lower accuracy (fast decay)
    elif category == 'Peak-Valley':
        return 0.50  # Fallback: neutral
    elif category == 'ML-Confluence':
        return 0.60  # Confluence: slightly higher (multiple algorithms agree)
    elif category == 'Neural-Network' or source == 'Neural Network':
        return 0.52  # Neural Network: pattern-based, moderate accuracy (depends on training)
    else:
        return 0.50  # Default: neutral

def enhance_levels_with_contextual_probability(
    levels,
    current_price,
    expected_range,
    sigma_price,
    range_mid=None
):
    """
    Enhance all levels with contextual success probability.
    
    This adds a third probability dimension:
    1. level_strength (structural - does this level matter?)
    2. reversionProb (immediate - first touch rejection)
    3. contextualSuccess (contextual - given position, how favorable?)
    
    Does NOT replace existing probabilities, only adds contextualSuccess.
    """
    if not levels:
        return levels
    
    enhanced = []
    for level in levels:
        level_price = level.get('price', 0)
        level_strength = level.get('strength', level.get('reversionProb', 0.5))
        category = level.get('category', 'Unknown')
        source = level.get('source', 'Unknown')
        
        # Calculate distance in sigma
        distance_sigma = abs(level_price - current_price) / sigma_price if sigma_price > 0 else 2.0
        
        # Get model accuracy for this level type
        model_accuracy = get_model_accuracy_by_category(category, source)
        
        # Calculate contextual success probability
        contextual_success = calculate_contextual_success_probability(
            level_price=level_price,
            level_strength=level_strength,
            current_price=current_price,
            expected_range=expected_range,
            distance_sigma=distance_sigma,
            model_accuracy=model_accuracy,
            range_mid=range_mid
        )
        
        # Add contextual probability (does NOT replace existing)
        enhanced_level = level.copy()
        enhanced_level['contextualSuccess'] = contextual_success
        enhanced_level['firstTouchReversion'] = level.get('reversionProb', level_strength)  # Keep original
        enhanced_level['levelStrength'] = level_strength  # Explicit structural strength
        
        enhanced.append(enhanced_level)
    
    return enhanced

def calculate_local_interaction_levels(closes, current_price, sigma_price, lookback=200, bins=30, max_levels=5):
    """
    Local Interaction Levels: Short-memory, near current price, explicitly non-structural.
    
    This replaces MeanShift with a cleaner approach:
    - Finds local density peaks in recent price histogram
    - Only near current price (within ~2 sigma)
    - Fast decay (not structural memory)
    - Answers: "Where is price likely to react today?"
    
    This is NOT TA, NOT structural memory, NOT global clustering.
    It's simply: where does price repeatedly visit in the recent window?
    """
    if len(closes) < 50:
        return []
    
    closes = np.array(closes[-lookback:])
    
    # Build price histogram
    hist, bin_edges = np.histogram(closes, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find local density peaks (modes)
    # prominence threshold avoids tiny noise peaks
    prominence_threshold = np.max(hist) * 0.15
    peaks, properties = find_peaks(hist, prominence=prominence_threshold)
    
    if len(peaks) == 0:
        return []
    
    levels = []
    for p in peaks:
        price = float(bin_centers[p])
        density_strength = hist[p] / np.max(hist)
        distance_sigma = abs(price - current_price) / sigma_price if sigma_price > 0 else float('inf')
        
        # Soft gating: interaction should be reasonably close to current price
        if distance_sigma > 2.0:
            continue
        
        # Strength reflects local density, not structural memory
        # Cap at 0.75 to keep it below structural levels
        strength = min(0.75, 0.45 + density_strength * 0.4)
        
        levels.append({
            'price': price,
            'type': 'Local Interaction',
            'category': 'Interaction',  # Explicit category
            'source': 'Local Density',
            'strength': float(strength),
            'distance_sigma': float(distance_sigma),
            'reversionProb': float(strength),
            'breakoutProb': float(1 - strength),
            'touches': int(hist[p]),  # Density count
            'decay': 'fast',  # Explicitly short half-life
            'density_prominence': float(density_strength)
        })
    
    # Prioritize closest + strongest
    levels = sorted(levels, key=lambda x: (x['distance_sigma'], -x['strength']))
    
    result = levels[:max_levels]
    print(f"Local Interaction: Found {len(result)} levels near price (within 2 sigma)")
    return result

def merge_threshold_by_timeframe(tf):
    """
    Timeframe-aware merge thresholds for Agglomerative clustering.
    Tighter thresholds for shorter timeframes, looser for longer.
    """
    return {
        "1m": 0.0010,   # 0.10%
        "5m": 0.0015,   # 0.15%
        "15m": 0.0020,  # 0.20%
        "30m": 0.0025,  # 0.25%
        "1h": 0.0030,   # 0.30%
        "4h": 0.0040,   # 0.40%
        "1d": 0.0060    # 0.60%
    }.get(tf, 0.0025)  # Default 0.25%

def agglomerative_merge_levels(
    levels,
    distance_threshold_pct=0.0025,
    price_key="price",
    timeframe="1d"
):
    """
    Merge nearby price levels using Agglomerative Hierarchical Clustering.
    
    This is a cleaner, production-ready version that merges levels AFTER discovery
    but BEFORE scoring to prevent probability fragmentation.
    
    Parameters
    ----------
    levels : list[dict]
        Each dict must contain at least {'price': float}
    distance_threshold_pct : float, optional
        Merge distance as % of price (scale-aware). If None, uses timeframe-aware default.
    price_key : str
        Key name for level price
    timeframe : str
        Timeframe for adaptive threshold selection

    Returns
    -------
    merged_levels : list[dict]
    """
    if not levels or len(levels) <= 1:
        return levels

    # Use timeframe-aware threshold if not explicitly provided
    if distance_threshold_pct is None or distance_threshold_pct == 0.0025:
        distance_threshold_pct = merge_threshold_by_timeframe(timeframe)

    prices = np.array([lvl.get(price_key, lvl.get('price', 0)) for lvl in levels], dtype=float)
    
    # Filter out invalid prices
    valid_mask = ~(np.isnan(prices) | np.isinf(prices) | (prices <= 0))
    if not np.any(valid_mask):
        return levels
    
    prices = prices[valid_mask]
    valid_levels = [levels[i] for i in range(len(levels)) if valid_mask[i]]
    
    if len(valid_levels) <= 1:
        return valid_levels

    # Scale-aware absolute distance
    avg_price = np.mean(prices)
    distance_threshold = avg_price * distance_threshold_pct

    # Agglomerative clustering
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward"
    )

    labels = model.fit_predict(prices.reshape(-1, 1))

    merged = {}
    for label, level in zip(labels, valid_levels):
        merged.setdefault(label, []).append(level)

    merged_levels = []
    for cluster_levels in merged.values():
        cluster_prices = np.array([l.get(price_key, l.get('price', 0)) for l in cluster_levels])
        
        # Strength-weighted merge (strongest level dominates)
        weights = np.array([l.get("strength", l.get("reversionProb", 0.5)) for l in cluster_levels])
        weighted_price = np.average(cluster_prices, weights=weights)
        
        merged_level = {
            "price": float(weighted_price),
            "strength": float(np.mean([l.get("strength", l.get("reversionProb", 0.5)) for l in cluster_levels])),
            "touches": int(sum(l.get("touches", 1) for l in cluster_levels)),
            "merged_count": len(cluster_levels),
            "sources": list(set(l.get("category", "unknown") for l in cluster_levels)),
            "category": "Agglomerative-Merged",
            "breakoutProb": float(np.mean([l.get("breakoutProb", 0.5) for l in cluster_levels])),
            "reversionProb": float(np.mean([l.get("reversionProb", 0.5) for l in cluster_levels]))
        }
        
        merged_levels.append(merged_level)

    return merged_levels

def adaptive_merge_threshold(micro_state, sigma_price, spot):
    """
    Regime-aware merge threshold for hierarchical clustering.
    Tighter merge in Thermal (precision events), looser in Fock (overshoot).
    """
    base = 0.005  # 0.5% base threshold
    
    if micro_state == 'Thermal':
        return base * 0.6  # Tighter (0.3%) - precision events need exact levels
    elif micro_state == 'Fock':
        return base * 1.5  # Looser (0.75%) - overshoot means levels are less precise
    else:  # Coherent
        return base  # Standard (0.5%)


def hierarchical_merge_nearby_levels(levels, distance_thresh_pct=0.005, current_price=None):
    """
    Merge structurally similar levels BEFORE scoring using Agglomerative Clustering.
    
    Purpose:
    - Prevents probability fragmentation from competing similar levels
    - Uses Ward linkage to preserve hierarchical structure
    - Weighted by strength for better accuracy
    
    This is POST-PROCESSING, not discovery. Use BEFORE get_ml_confluence_levels.
    """
    from sklearn.cluster import AgglomerativeClustering
    
    if len(levels) < 2:
        return levels
    
    # Extract prices
    prices = np.array([l['price'] for l in levels]).reshape(-1, 1)
    
    # Use percentage-based threshold (regime-aware)
    if current_price is None:
        median_price = np.median(prices)
    else:
        median_price = current_price
    
    distance_thresh = median_price * distance_thresh_pct
    
    # Ward linkage preserves hierarchy and works well for 1D price data
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_thresh,
        linkage='ward'
    )
    
    labels = model.fit_predict(prices)
    
    # Merge levels with same label
    merged = {}
    for level, label in zip(levels, labels):
        if label not in merged:
            merged[label] = []
        merged[label].append(level)
    
    # Create merged levels (weighted by strength)
    merged_levels = []
    for group in merged.values():
        if len(group) == 1:
            # Single level, keep as-is but mark as hierarchical
            level = group[0].copy()
            level['category'] = level.get('category', 'Unknown')
            merged_levels.append(level)
            continue
        
        # Weight by strength for better accuracy
        strengths = [l.get('strength', l.get('reversionProb', 0.5)) for l in group]
        prices_list = [l['price'] for l in group]
        
        # Weighted average price
        merged_price = np.average(prices_list, weights=strengths)
        
        # Boost strength for confluence (multiple algorithms agree)
        base_strength = np.mean(strengths)
        confluence_boost = min(len(group) / 3, 1.2)  # Up to 20% boost for 3+ levels
        merged_strength = min(base_strength * confluence_boost, 0.95)
        
        # Collect source algorithms
        source_algorithms = list(set(l.get('category', 'Unknown') for l in group))
        
        merged_levels.append({
            'price': float(merged_price),
            'type': 'Hierarchical Merge',
            'touches': sum(l.get('touches', 1) for l in group),
            'strength': float(merged_strength),
            'breakoutProb': float(1 - merged_strength),
            'reversionProb': float(merged_strength),
            'level_strength': float(merged_strength),  # Structural validity
            'immediate_reversion_prob': float(merged_strength * 0.75),  # Event probability
            'category': 'Hierarchical',
            'merged_count': len(group),
            'source_algorithms': source_algorithms,
            'sources': source_algorithms,  # Also set 'sources' for compatibility with extraction code
            'original_levels': [{'price': l['price'], 'category': l.get('category', 'Unknown')} for l in group]
        })
    
    return merged_levels


def get_ml_confluence_levels(all_algorithm_levels):
    """
    ML Confluence: Meta-wrapper for levels where multiple algorithms agree.
    This is POST-PROCESSING, not discovery. The actual structural levels
    (HDBSCAN, etc.) should be shown explicitly, not hidden behind this.
    """
    final_levels = []
    used = set()
    for level in sorted(all_algorithm_levels, key=lambda x: x['price']):
        if level['price'] in used:
            continue
        similar = [l for l in all_algorithm_levels 
                  if abs(l['price'] - level['price']) / level['price'] < 0.01 
                  and l['price'] not in used]
        if len(similar) >= 2:
            avg_price = np.mean([l['price'] for l in similar])
            avg_strength = np.mean([l.get('strength', 0.5) for l in similar])
            confluence_strength = min(avg_strength * len(similar) / 2, 0.95)
            
            # Preserve source information - if HDBSCAN is in confluence, mark it
            sources = [l.get('source', l.get('category', 'Unknown')) for l in similar]
            primary_source = 'HDBSCAN' if 'HDBSCAN' in sources or 'Density (HDBSCAN)' in sources else sources[0] if sources else 'Unknown'
            
            final_levels.append({
                'price': float(avg_price), 
                'type': 'ML Confluence',
                'strength': confluence_strength, 
                'algorithms': [l.get('category', 'Unknown') for l in similar],
                'source': primary_source,  # Track primary structural source
                'confluence_count': len(similar), 
                'breakoutProb': float(1 - confluence_strength),
                'reversionProb': float(confluence_strength), 
                'category': 'ML-Confluence'  # Meta-wrapper, not primary level
            })
            for l in similar:
                used.add(l['price'])
    return final_levels  

# ============================================================================
# ENHANCED API ENDPOINT WITH MICROSTRUCTURE
# ============================================================================

@app.route('/api/data', methods=['GET'])
def get_data():
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    timeframe = request.args.get('timeframe', '1d').strip().lower().replace('240m','4h').replace('4hour','4h').replace('4hours','4h').replace('60m','1h')
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    historical_mode = request.args.get('historical_mode', 'false').lower() == 'true'
    
    try:
        print(f"\n{'='*60}")
        print(f"Analysis: {ticker} - User: {session.get('username')}")
        print(f"{'='*60}")
        
        stock = yf.Ticker(ticker)
        
        # For futures, use alternative interval formats that yfinance accepts better
        is_futures = '=' in ticker
        
        # Special handling for 1h timeframe - yfinance has issues with it for futures
        if is_futures and timeframe == '1h':
            # For futures 1h, use 60m from the start
            interval = '60m'
            print(f"⚠ Futures 1h timeframe detected for {ticker}, using 60m interval")
        elif is_futures:
            # Use minute-based intervals for futures (yfinance prefers these)
            # Note: 4h is not supported by yfinance - will use resampling from 60m
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '60m', '4h': '60m', '1d': '1d'}
            interval = interval_map.get(timeframe, '1d')
        else:
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
            interval = interval_map.get(timeframe, '1d')
        
        if start_date and end_date:
            # Date range path - handle futures intraday here too
            if is_futures and timeframe == '4h':
                # For 4h futures with date range, use resampling
                print(f"Fetching 4h data for {ticker} with date range (will resample from 1h/60m)...")
                try:
                    hist = fetch_historical_data_with_resampling(
                        ticker=ticker,
                        timeframe='4h',
                        start_date=start_date,
                        end_date=end_date,
                        is_futures=True
                    )
                except Exception as e:
                    print(f"⚠ Resampling fetch failed: {e}")
                    hist = None
            else:
                hist = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            # No date range path - handle all futures intraday timeframes here
            # Futures handling: use shorter periods for intraday timeframes
            if is_futures and timeframe in ['1m', '5m', '15m', '1h', '4h']:
                period_map = {'1m': '5d', '5m': '5d', '15m': '7d', '1h': '7d', '4h': '10d', '1d': '2y'}
            else:
                period_map = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '3mo', '4h': '3mo', '1d': '2y'}
            
            period = period_map.get(timeframe, '1y')
            
            # Try to get data, with fallback to shorter periods for futures
            hist = None
            
            # Handle 4h futures (requires resampling)
            if is_futures and timeframe == '4h':
                print(f"Fetching 4h data for {ticker} (will resample from 1h/60m)...")
                try:
                    hist = fetch_historical_data_with_resampling(
                        ticker=ticker,
                        timeframe='4h',
                        period=period,
                        is_futures=True
                    )
                except Exception as e:
                    print(f"⚠ Resampling fetch failed: {e}")
                    hist = None
            
            # Handle 1h futures
            elif is_futures and timeframe == '1h':
                # Special handling for 1h futures - try many combinations
                attempts = [
                    ('60m', '5d'),   # Most reliable for futures
                    ('60m', '3d'),
                    ('60m', '2d'),
                    ('60m', '1d'),
                    ('1h', '5d'),    # Try standard format too
                    ('1h', '3d'),
                    ('1h', '2d'),
                    ('1h', '1d'),
                ]
                
                for attempt_interval, attempt_period in attempts:
                    try:
                        print(f"Trying {ticker} 1h: interval={attempt_interval}, period={attempt_period}")
                        hist = stock.history(period=attempt_period, interval=attempt_interval)
                        if hist is not None and len(hist) > 0:
                            print(f"✓ Successfully fetched {len(hist)} bars for {ticker} 1h with interval={attempt_interval}, period={attempt_period}")
                            break
                    except Exception as e:
                        error_msg = str(e)
                        print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:150]}")
                        continue
            
            # Handle 1m, 5m, 15m futures
            elif is_futures and timeframe in ['1m', '5m', '15m']:
                if timeframe in ['15m']:
                    attempts = [period, '5d', '3d', '2d', '1d']
                else:
                    attempts = [period, '5d', '2d', '1d']
                
                for attempt_period in attempts:
                    # Try both the mapped interval and original timeframe format
                    interval_options = [interval]
                    if timeframe == '15m':
                        interval_options = ['15m']
                    
                    for attempt_interval in interval_options:
                        try:
                            hist = stock.history(period=attempt_period, interval=attempt_interval)
                            if hist is not None and len(hist) > 0:
                                print(f"✓ Successfully fetched {len(hist)} bars for {ticker} at {timeframe} with interval={attempt_interval}, period={attempt_period}")
                                break
                        except Exception as e:
                            error_msg = str(e)
                            if "pattern" not in error_msg.lower() and "expected" not in error_msg.lower():
                                print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:100]}")
                            continue
                    
                    if hist is not None and len(hist) > 0:
                        break
            
            # Handle regular (non-futures) 4h (also needs resampling)
            elif timeframe == '4h':
                print(f"Fetching 4h data for {ticker} (will resample from 1h)...")
                try:
                    hist = fetch_historical_data_with_resampling(
                        ticker=ticker,
                        timeframe='4h',
                        period=period,
                        is_futures=False
                    )
                except Exception as e:
                    print(f"⚠ Resampling fetch failed: {e}")
                    hist = None
            
            # Fallback for all other cases
            else:
                try:
                    hist = stock.history(period=period, interval=interval)
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠ Error fetching data for {ticker} {timeframe}: {error_msg}")
                    # Try alternative interval format if pattern error
                    if "pattern" in error_msg.lower() or "expected" in error_msg.lower():
                        try:
                            if interval == '1h':
                                hist = stock.history(period=period, interval='60m')
                        except:
                            hist = None
                    else:
                        hist = None
        
        if hist is None or len(hist) == 0:
            error_msg = f'No data available for {ticker} at {timeframe}'
            if '=' in ticker and timeframe in ['1m', '5m', '15m', '1h', '4h']:
                error_msg += '. Futures have limited intraday data availability from yfinance.'
            return jsonify({'success': False, 'error': error_msg}), 400
        
        price_data = []
        for idx, row in hist.iterrows():
            price_data.append({
                'date': idx.strftime('%Y-%m-%d %H:%M'),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        volumes = hist['Volume'].values
        current_price = closes[-1]
        
        print(f"Current: ${current_price:.2f} | Bars: {len(closes)}")
        
        if historical_mode:
            lookback_idx = int(len(closes) * 0.8)
            lookback_idx = min(max(lookback_idx, 50), len(closes))
            hist_closes = closes[:lookback_idx]
            hist_highs = highs[:lookback_idx]
            hist_lows = lows[:lookback_idx]
            hist_volumes = volumes[:lookback_idx]
            hist_data_subset = hist.iloc[:lookback_idx]
        else:
            hist_closes = closes
            hist_highs = highs
            hist_lows = lows
            hist_volumes = volumes
            hist_data_subset = hist
        
        returns = np.log(hist_closes[1:] / hist_closes[:-1]) * 100
        
        print("Running enhanced analysis (GARCH + Microstructure)...")
        
        # Calculate sigma_price for use in level enhancement and path calculation
        # FIXED: Calculate sigma_price in price units for consistent usage
        is_intraday = timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']
        if is_intraday and all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close']):
            try:
                vol_result = compute_session_volatility(hist, window=60)
                sigma_price = vol_result['sigma_price']  # Already in price units
            except Exception as e:
                print(f"⚠ Session vol calculation failed: {e}, using fallback")
                sigma_session = np.std(returns) if len(returns) > 0 else 0.015
                sigma_price = sigma_session * current_price
        else:
            # Multi-day: calculate from returns
            sigma_session = np.std(returns) if len(returns) > 0 else 0.015
            sigma_price = sigma_session * current_price
        
        # GARCH VOLATILITY REGIME
        garch_vol_regime = calculate_garch_volatility_regime(closes)
        print(f"✓ GARCH Regime: {garch_vol_regime['regime']}")
        
        # MARKET MICROSTRUCTURE STATE
        microstructure_state = detect_market_microstructure_state(hist_closes, hist_volumes, returns, hist_highs, hist_lows)
        print(f"✓ Market State: {microstructure_state['state']} (confidence: {microstructure_state['confidence']:.2f})")
        
        # PHASE SPACE COORDINATES
        phase_space = calculate_phase_space_coordinates(hist_closes, hist_volumes)
        
        # FORECASTS WITH GARCH ENHANCEMENT
        forecasts = generate_price_forecast(hist_closes, hist_highs, hist_lows, hist_volumes, forecast_periods=20)
        forecasts = calculate_garch_confidence_bands(forecasts, garch_vol_regime)
        print(f"✓ Forecasts generated")
        
        # MACRO INDICATORS
        macro_indicators = get_macro_indicators()
        
       # LEVEL DETECTION - Best-in-class production stack
        print("Running level detection algorithms...")
        
        # PRIMARY: HDBSCAN (state-of-the-art density clustering)
        hdbscan_levels = calculate_hdbscan_levels(hist_highs, hist_lows, hist_closes, timeframe=timeframe)
        print(f"HDBSCAN: Generated {len(hdbscan_levels) if hdbscan_levels else 0} levels")
        
        # NEW: Enhanced OPTICS with reachability plots
        try:
            enhanced_optics_levels_result = enhanced_optics_levels(hist_highs, hist_lows, hist_closes, timeframe=timeframe)
            print(f"Enhanced OPTICS: Generated {len(enhanced_optics_levels_result) if enhanced_optics_levels_result else 0} levels")
        except Exception as e:
            print(f"Enhanced OPTICS failed: {e}")
            enhanced_optics_levels_result = []
        
        # NEW: KDE-based levels
        try:
            kde_levels_result = kde_based_levels(hist_highs, hist_lows, hist_closes, n_levels=10)
            print(f"KDE: Generated {len(kde_levels_result) if kde_levels_result else 0} levels")
        except Exception as e:
            print(f"KDE levels failed: {e}")
            kde_levels_result = []
        
        # NEW: Multi-scale HDBSCAN
        try:
            multiscale_hdbscan_levels_result = multiscale_hdbscan_levels(hist_highs, hist_lows, hist_closes, timeframe=timeframe)
            print(f"Multi-scale HDBSCAN: Generated {len(multiscale_hdbscan_levels_result) if multiscale_hdbscan_levels_result else 0} levels")
        except Exception as e:
            print(f"Multi-scale HDBSCAN failed: {e}")
            multiscale_hdbscan_levels_result = []
        
        # NEW: Time-weighted HDBSCAN (if timestamps available)
        time_weighted_levels_result = []
        try:
            if hasattr(hist.index, 'values'):
                timestamps = hist.index.values
                time_weighted_levels_result = time_weighted_hdbscan(hist_highs, hist_lows, hist_closes, timestamps, half_life_days=30)
                print(f"Time-weighted HDBSCAN: Generated {len(time_weighted_levels_result) if time_weighted_levels_result else 0} levels")
        except Exception as e:
            print(f"Time-weighted HDBSCAN failed: {e}")
            time_weighted_levels_result = []
        
        # NEW: Wyckoff zones
        try:
            wyckoff_levels_result = detect_wyckoff_zones(hist_data_subset, lookback=50)
            print(f"Wyckoff: Generated {len(wyckoff_levels_result) if wyckoff_levels_result else 0} levels")
        except Exception as e:
            print(f"Wyckoff zones failed: {e}")
            wyckoff_levels_result = []
        
        # NEW: Persistent Homology (TDA)
        persistent_homology_levels_result = []
        try:
            if RIPSER_AVAILABLE:
                persistent_homology_levels_result = persistent_homology_levels(hist_highs, hist_lows, hist_closes, max_levels=8)
                print(f"Persistent Homology: Generated {len(persistent_homology_levels_result) if persistent_homology_levels_result else 0} levels")
        except Exception as e:
            print(f"Persistent Homology failed: {e}")
            persistent_homology_levels_result = []
        
        # NEW: Neural Network level detection
        neural_network_levels_result = []
        try:
            if TORCH_AVAILABLE:
                neural_network_levels_result = detect_levels_with_neural_network(hist_data_subset, lookback=100, threshold=0.7)
                print(f"Neural Network: Generated {len(neural_network_levels_result) if neural_network_levels_result else 0} levels")
        except Exception as e:
            print(f"Neural Network level detection failed: {e}")
            neural_network_levels_result = []

        # NEW: DeepSupp v4 (corr-series attention autoencoder)
        deepsupp_levels_result = []
        try:
            if TORCH_AVAILABLE:
                deepsupp_levels_result = detect_levels_with_deepsupp(hist_data_subset, model_path='deepsupp_v4.pt', device='cpu')
                print(f"DeepSupp: Generated {len(deepsupp_levels_result) if deepsupp_levels_result else 0} levels")
        except Exception as e:
            print(f"DeepSupp level detection failed: {e}")
            deepsupp_levels_result = []
        
        # SECONDARY: IsolationForest (event pivot candidates)
        isolation_forest_levels = find_pivot_anomalies(hist_highs, hist_lows, hist_closes)
        
        # INTERACTION: Local density modes (near price, short memory, explicitly non-structural)
        local_interaction_levels = calculate_local_interaction_levels(
            hist_closes, 
            current_price, 
            sigma_price,
            lookback=200 if not is_intraday else 300,  # More bars for intraday
            bins=30,
            max_levels=5
        )
        print(f"Local Interaction: Generated {len(local_interaction_levels) if local_interaction_levels else 0} levels")
        
        # FALLBACK: Peak/Valley (last-resort when density clustering fails)
        peak_valley_levels = find_peaks_valleys_scipy(hist_highs, hist_lows, hist_closes)
        
        # MeanShift removed from level production - now used as validator only
        # (validates HDBSCAN levels and boosts confidence if agrees)
        
        # CLASSICAL STRUCTURAL (constraints/magnets, not ML discovery)
        pivot_levels = calculate_pivot_points(hist_data_subset, timeframe)
        fib_levels = calculate_fibonacci_levels(hist_highs, hist_lows)  # For metadata only, not primary levels
        gap_levels = find_gap_levels(hist_data_subset)

        # ---- HARD GUARD: ensure all level outputs are lists ----
        hdbscan_levels = hdbscan_levels or []
        enhanced_optics_levels_result = enhanced_optics_levels_result or []
        kde_levels_result = kde_levels_result or []
        multiscale_hdbscan_levels_result = multiscale_hdbscan_levels_result or []
        time_weighted_levels_result = time_weighted_levels_result or []
        wyckoff_levels_result = wyckoff_levels_result or []
        persistent_homology_levels_result = persistent_homology_levels_result or []
        neural_network_levels_result = neural_network_levels_result or []
        deepsupp_levels_result = deepsupp_levels_result or []
        isolation_forest_levels = isolation_forest_levels or []
        peak_valley_levels = peak_valley_levels or []
        pivot_levels = pivot_levels or []
        fib_levels = fib_levels or []
        gap_levels = gap_levels or []
        
        # ML LEVELS: Primary discovery algorithms only (including new methods)
        all_ml_levels = (hdbscan_levels + enhanced_optics_levels_result + kde_levels_result + 
                        multiscale_hdbscan_levels_result + time_weighted_levels_result + 
                        wyckoff_levels_result + persistent_homology_levels_result + 
                        neural_network_levels_result + deepsupp_levels_result +
                        isolation_forest_levels + peak_valley_levels) 
        
        # CRITICAL: Preserve levels BEFORE merge (they get consumed by merge)
        # We need BOTH merged levels AND original levels for structural array
        hdbscan_raw_before_merge = [l.copy() for l in hdbscan_levels] if hdbscan_levels else []
        print(f"HDBSCAN RAW (before merge): {len(hdbscan_raw_before_merge)} levels")
        
        # Preserve new level types before merge (same pattern as HDBSCAN)
        enhanced_optics_raw_before_merge = [l.copy() for l in enhanced_optics_levels_result] if enhanced_optics_levels_result else []
        kde_raw_before_merge = [l.copy() for l in kde_levels_result] if kde_levels_result else []
        multiscale_hdbscan_raw_before_merge = [l.copy() for l in multiscale_hdbscan_levels_result] if multiscale_hdbscan_levels_result else []
        time_weighted_raw_before_merge = [l.copy() for l in time_weighted_levels_result] if time_weighted_levels_result else []
        wyckoff_raw_before_merge = [l.copy() for l in wyckoff_levels_result] if wyckoff_levels_result else []
        persistent_homology_raw_before_merge = [l.copy() for l in persistent_homology_levels_result] if persistent_homology_levels_result else []
        neural_network_raw_before_merge = [l.copy() for l in neural_network_levels_result] if neural_network_levels_result else []
        deepsupp_raw_before_merge = [l.copy() for l in deepsupp_levels_result] if deepsupp_levels_result else []
        
        # NEW: Agglomerative merge BEFORE confluence (prevents probability fragmentation)
        # Use timeframe-aware threshold (cleaner than regime-aware for this step)
        all_ml_levels_merged = agglomerative_merge_levels(
            all_ml_levels,
            distance_threshold_pct=None,  # Will use timeframe-aware default
            timeframe=timeframe
        )
        
        # Extract merged levels that came from HDBSCAN (check sources field)
        # Also check if original source was HDBSCAN
        hdbscan_merged = []
        for l in all_ml_levels_merged:
            if l.get('category') == 'Agglomerative-Merged':
                sources = l.get('sources', [])
                source_str = str(sources) if sources else ''
                # Check if HDBSCAN is in sources or if source field indicates HDBSCAN
                if ('Density (HDBSCAN)' in sources or 
                    'HDBSCAN' in source_str or 
                    l.get('source') == 'HDBSCAN'):
                    # Preserve HDBSCAN identity in merged level
                    l['category'] = 'Density (HDBSCAN)'  # Restore category for structural array
                    l['source'] = 'HDBSCAN'  # Ensure source is set
                    hdbscan_merged.append(l)
            elif l.get('category') == 'Density (HDBSCAN)' or l.get('category') == 'HDBSCAN':
                # Single unmerged HDBSCAN level
                hdbscan_merged.append(l)
        print(f"HDBSCAN MERGED (after agglomerative): {len(hdbscan_merged)} levels")
        
        # Use merged levels for confluence, but preserve HDBSCAN separately
        all_ml_levels = all_ml_levels_merged
        
        confluence_levels = get_ml_confluence_levels(all_ml_levels)
        confluence_levels = confluence_levels or []

        # Combine ML levels with classical structural (as constraints)
        # NOTE: Fibonacci is NOT added here - it will be added as metadata only
        all_levels_combined = (confluence_levels + all_ml_levels + 
                              pivot_levels + gap_levels)
        
        # Add Fibonacci as metadata/confluence to nearby levels (not as primary levels)
        all_levels_combined = add_fibonacci_metadata_to_levels(
            all_levels_combined, fib_levels, sigma_price, threshold_sigma=1.0
        )
        
        # MICROSTRUCTURE-ENHANCED LEVEL ADJUSTMENT
        all_levels_combined, hmm_regime, hurst_data, garch_regime, micro_state = enhance_levels_with_microstructure(
            all_levels_combined, closes, volumes, current_price, garch_vol_regime, microstructure_state, sigma_price=sigma_price
        )
        
        print(f"✓ Analysis complete (Microstructure-enhanced)")
        
        # CONTEXTUAL PROBABILITY ENHANCEMENT
        # Calculate expected range for contextual probability
        # Use GARCH volatility regime or sigma-based estimate
        if garch_vol_regime and 'expected_range' in garch_vol_regime:
            expected_range = garch_vol_regime['expected_range']
        else:
            # Fallback: estimate from sigma (2-sigma range is ~95% of moves)
            expected_range = 4.0 * sigma_price if sigma_price > 0 else abs(hist_closes.max() - hist_closes.min()) * 0.1
        
        range_mid = current_price  # Center of expected range
        
        # Enhance all levels with contextual success probability
        # This adds contextualSuccess without replacing existing probabilities
        all_levels_combined = enhance_levels_with_contextual_probability(
            all_levels_combined,
            current_price=current_price,
            expected_range=expected_range,
            sigma_price=sigma_price,
            range_mid=range_mid
        )
        print(f"✓ Contextual probabilities added to {len(all_levels_combined)} levels")
        
        # NEW: Apply RL validation to filter weak levels (before extraction)
        try:
            if TORCH_AVAILABLE:
                all_levels_combined = validate_levels_with_rl(all_levels_combined, current_price, sigma_price)
                print(f"✓ RL validation filtered to {len(all_levels_combined)} validated levels")
        except Exception as e:
            print(f"⚠ RL validation failed: {e}, using all levels")
        
        # ORGANIZE LEVELS BY CATEGORY - Separated into ML and Classical
        ml_confluence = [l for l in all_levels_combined if l['category'] == 'ML-Confluence']
        
        # HDBSCAN levels: Use the merged HDBSCAN levels we preserved
        # These are the agglomerative-merged levels that came from HDBSCAN
        # If merge didn't happen or no merged levels, fall back to raw
        if len(hdbscan_merged) > 0:
            hdbscan_ml = hdbscan_merged
            print(f"Using {len(hdbscan_ml)} merged HDBSCAN levels for structural array")
        else:
            # Fallback: Try to extract from all_levels_combined (shouldn't happen but safety)
            hdbscan_ml = [l for l in all_levels_combined if l.get('category') == 'Density (HDBSCAN)' or l.get('category') == 'HDBSCAN']
            if len(hdbscan_ml) == 0 and len(hdbscan_raw_before_merge) > 0:
                # Last resort: Use raw HDBSCAN if merge consumed them
                hdbscan_ml = hdbscan_raw_before_merge
                print(f"Fallback: Using {len(hdbscan_ml)} raw HDBSCAN levels (merge may have consumed them)")
        
        # NEW: Extract new level detection methods from merged levels (check sources) and unmerged levels
        # Extract from merged levels by checking sources field, and from unmerged by category
        enhanced_optics_ml = []
        kde_ml = []
        multiscale_hdbscan_ml = []
        time_weighted_ml = []
        wyckoff_ml = []
        persistent_homology_ml = []
        neural_network_ml = []
        deepsupp_ml = []
        
        for l in all_levels_combined:
            category = l.get('category', '')
            sources = l.get('sources', l.get('source_algorithms', []))  # Check both field names
            
            # Normalize sources to list if it's a string or other type
            if isinstance(sources, str):
                sources = [sources]
            elif not isinstance(sources, list):
                sources = list(sources) if sources else []
            
            # Check merged levels (category='Agglomerative-Merged' or 'Hierarchical' with sources)
            if category == 'Agglomerative-Merged' or category == 'Hierarchical':
                if 'OPTICS' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'OPTICS'  # Restore category
                    enhanced_optics_ml.append(l_copy)
                if 'KDE' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'KDE'  # Restore category
                    kde_ml.append(l_copy)
                if 'HDBSCAN-MultiScale' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'HDBSCAN-MultiScale'  # Restore category
                    multiscale_hdbscan_ml.append(l_copy)
                if 'HDBSCAN-TimeWeighted' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'HDBSCAN-TimeWeighted'  # Restore category
                    time_weighted_ml.append(l_copy)
                if 'Wyckoff' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'Wyckoff'  # Restore category
                    wyckoff_ml.append(l_copy)
                if 'TDA' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'TDA'  # Restore category
                    persistent_homology_ml.append(l_copy)
                if 'Neural-Network' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'Neural-Network'  # Restore category
                    neural_network_ml.append(l_copy)
                if 'DeepSupp' in sources:
                    l_copy = l.copy()
                    l_copy['category'] = 'DeepSupp'
                    deepsupp_ml.append(l_copy)
            # Check unmerged levels (preserved original categories)
            elif category == 'OPTICS':
                enhanced_optics_ml.append(l)
            elif category == 'KDE':
                kde_ml.append(l)
            elif category == 'HDBSCAN-MultiScale':
                multiscale_hdbscan_ml.append(l)
            elif category == 'HDBSCAN-TimeWeighted':
                time_weighted_ml.append(l)
            elif category == 'Wyckoff':
                wyckoff_ml.append(l)
            elif category == 'TDA':
                persistent_homology_ml.append(l)
            elif category == 'Neural-Network':
                neural_network_ml.append(l)
            elif category == 'DeepSupp':
                deepsupp_ml.append(l)
        
        # Fallback: Use raw levels if extraction found nothing (shouldn't happen but safety)
        if len(enhanced_optics_ml) == 0 and len(enhanced_optics_raw_before_merge) > 0:
            enhanced_optics_ml = enhanced_optics_raw_before_merge
        if len(kde_ml) == 0 and len(kde_raw_before_merge) > 0:
            kde_ml = kde_raw_before_merge
        if len(multiscale_hdbscan_ml) == 0 and len(multiscale_hdbscan_raw_before_merge) > 0:
            multiscale_hdbscan_ml = multiscale_hdbscan_raw_before_merge
        if len(time_weighted_ml) == 0 and len(time_weighted_raw_before_merge) > 0:
            time_weighted_ml = time_weighted_raw_before_merge
        if len(wyckoff_ml) == 0 and len(wyckoff_raw_before_merge) > 0:
            wyckoff_ml = wyckoff_raw_before_merge
        if len(persistent_homology_ml) == 0 and len(persistent_homology_raw_before_merge) > 0:
            persistent_homology_ml = persistent_homology_raw_before_merge
        if len(neural_network_ml) == 0 and len(neural_network_raw_before_merge) > 0:
            neural_network_ml = neural_network_raw_before_merge
        if len(deepsupp_ml) == 0 and len(deepsupp_raw_before_merge) > 0:
            deepsupp_ml = deepsupp_raw_before_merge
        
        # DEBUG: Log new level counts
        print(f"🔍 NEW LEVEL DETECTION METHODS:")
        print(f"   OPTICS: {len(enhanced_optics_ml)} levels")
        print(f"   KDE: {len(kde_ml)} levels")
        print(f"   Multi-Scale HDBSCAN: {len(multiscale_hdbscan_ml)} levels")
        print(f"   Time-Weighted HDBSCAN: {len(time_weighted_ml)} levels")
        print(f"   Wyckoff: {len(wyckoff_ml)} levels")
        print(f"   Persistent Homology (TDA): {len(persistent_homology_ml)} levels")
        print(f"   Neural Network: {len(neural_network_ml)} levels")
        print(f"   DeepSupp: {len(deepsupp_ml)} levels")
        if len(neural_network_ml) > 0:
            print(f"   ✓ Neural Network levels found: {[l.get('price') for l in neural_network_ml[:3]]}")
        
        # Combine all structural density-based levels
        hdbscan_ml = hdbscan_ml + enhanced_optics_ml + kde_ml + multiscale_hdbscan_ml + time_weighted_ml + wyckoff_ml + persistent_homology_ml + neural_network_ml + deepsupp_ml
        
        isolation_forest_ml = [l for l in all_levels_combined if l['category'] == 'Isolation-Forest']
        peak_valley_ml = [l for l in all_levels_combined if l['category'] == 'Peak-Valley']
        
        pivot_classical = [l for l in all_levels_combined if l['category'] == 'Pivot']
        gap_classical = [l for l in all_levels_combined if l['category'] == 'Gap']
        
        # DEBUG: Log level counts before building response
        print(f"Level organization - HDBSCAN: {len(hdbscan_ml)}, Confluence: {len(ml_confluence)}, Event: {len(isolation_forest_ml)}, Interaction: {len(local_interaction_levels)}, Fallback: {len(peak_valley_ml)}")
        
        # VALIDATION: Ensure all structural levels have valid price field
        hdbscan_ml = [l for l in hdbscan_ml if l and isinstance(l.get('price'), (int, float)) and not (np.isnan(l.get('price')) or np.isinf(l.get('price')))]
        print(f"Structural levels after validation: {len(hdbscan_ml)} levels with valid prices")
        
        # DEBUG: Log category breakdown of structural levels
        category_counts = {}
        for l in hdbscan_ml:
            cat = l.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"📊 Structural level categories: {category_counts}")
        
        # VALIDATION: Ensure interaction levels have valid prices
        local_interaction_levels = [l for l in local_interaction_levels if l and isinstance(l.get('price'), (int, float)) and not (np.isnan(l.get('price')) or np.isinf(l.get('price')))]
        print(f"Interaction after validation: {len(local_interaction_levels)} levels with valid prices")

        levels = {
            # PRIMARY STRUCTURAL LEVELS (discovered density / memory)
            'structural': hdbscan_ml,              # HDBSCAN + Agglomerative merged levels

            # EVENT / PIVOT LEVELS (behavioral, fast-decay)
            'event': isolation_forest_ml,          # Stop-runs, gaps, impulse pivots

            # INTERACTION LEVELS (local density, near price, short memory, explicitly non-structural)
            'interaction': local_interaction_levels,  # Local density modes - play zones, not structure

            # FALLBACK STRUCTURE (only if density is sparse)
            'fallback': peak_valley_ml,            # Peak / valley geometric structure

            # CLASSICAL REFERENCES (constraints, not ML discovery)
            'classicalStructural': {
                'pivots': pivot_classical,
                'gaps': gap_classical
            },
            
            # Backward compatibility: Include old fields
            'mlConfluence': ml_confluence,  # ML confluence levels
            'peakValley': peak_valley_ml,   # Peak-Valley levels (fallback)
            'meanshift': [],  # Removed from production, now validator only
            'dbscan': [],     # Removed from production
            'gmm': [],        # Removed from production
            'kmeans': [],     # Removed from production
            'volatility': [], # Removed from production
            'pivots': pivot_classical,  # Also at top level for backward compatibility
            'fibonacci': [],  # Removed as primary level - now metadata only
            'gaps': gap_classical      # Also at top level for backward compatibility
        }
        
        # CRITICAL DEBUG: Log final counts before sending to frontend
        print(f"🔍 FINAL LEVELS STRUCTURE:")
        print(f"   structural (HDBSCAN): {len(levels['structural'])}")
        print(f"   event (Isolation Forest): {len(levels['event'])}")
        print(f"   fallback (Peak-Valley): {len(levels['fallback'])}")
        print(f"   mlConfluence: {len(levels['mlConfluence'])}")
        if len(levels['structural']) > 0:
            print(f"   Sample HDBSCAN level: price={levels['structural'][0].get('price')}, category={levels['structural'][0].get('category')}")
        
        # CALCULATE MOST PROBABLE PRICE PATH
        print("Calculating most probable price path...")
        # Get IV surface data if available
        iv_surface_data = None
        try:
            vol_surface = generate_volatility_surface(current_price, garch_vol_regime)
            iv_surface_data = {'surface': vol_surface}
        except:
            pass
        
        most_probable_path = calculate_most_probable_price_path(
            closes, volumes, levels, garch_vol_regime, phase_space, micro_state, 
            forecast_periods=30, iv_surface_data=iv_surface_data, timeframe=timeframe, sigma_price=sigma_price
        )
        
        # Build response data
        response_data = {
            'success': True,
            'priceData': price_data,
            'levels': levels,
            'currentPrice': float(current_price),
            'volRegime': garch_vol_regime,
            'microstructureState': micro_state,
            'phaseSpace': phase_space,
            'hmmRegime': hmm_regime,
            'hurstData': hurst_data,
            'forecasts': forecasts,
            'macroIndicators': macro_indicators,
            'mostProbablePath': most_probable_path
        }
        
        # Sanitize entire response for JSON serialization
        sanitized_response = sanitize_for_json(response_data)
        
        # FINAL VALIDATION: Ensure structural levels survived sanitization
        if 'levels' in sanitized_response and 'structural' in sanitized_response['levels']:
            structural_count = len(sanitized_response['levels']['structural']) if isinstance(sanitized_response['levels']['structural'], list) else 0
            print(f"✅ HDBSCAN STRUCTURAL COUNT IN RESPONSE: {structural_count}")
        
        return jsonify(sanitized_response)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/data: {error_trace}")
        error_msg = str(e) if str(e) else "Unknown error occurred"
        return jsonify({'success': False, 'error': error_msg}), 400

# NEW ENDPOINT: VOLATILITY SURFACE
@app.route('/api/volatility-surface', methods=['GET'])
def get_volatility_surface():
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    
    try:
        print(f"Generating volatility surface for {ticker}...")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y', interval='1d')
        
        if len(hist) == 0:
            return jsonify({'success': False, 'error': 'No data available'}), 400
        
        closes = hist['Close'].values
        current_price = closes[-1]
        
        # Get GARCH regime for calibration
        garch_vol_regime = calculate_garch_volatility_regime(closes)
        
        # Generate volatility surface
        vol_surface = generate_volatility_surface(current_price, garch_vol_regime)
        
        print(f"✓ Surface generated with {len(vol_surface['surface'])} points")
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'surface': vol_surface,
            'garch_regime': garch_vol_regime
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/data: {error_trace}")
        error_msg = str(e) if str(e) else "Unknown error occurred"
        return jsonify({'success': False, 'error': error_msg}), 400

# NEW ENDPOINT: 3D PHASE SPACE DATA
@app.route('/api/phase-space', methods=['GET'])
def get_phase_space():
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    
    try:
        print(f"Generating phase space data for {ticker}...")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period='6mo', interval='1d')
        
        if len(hist) == 0:
            return jsonify({'success': False, 'error': 'No data available'}), 400
        
        closes = hist['Close'].values
        volumes = hist['Volume'].values
        
        # Add check for sufficient data
        if len(closes) < 50:
            return jsonify({'success': False, 'error': 'Insufficient data for phase space analysis'}), 400
            
        returns = np.log(closes[1:] / closes[:-1]) * 100
        
        # Calculate phase space coordinates
        phase_space = calculate_phase_space_coordinates(closes, volumes)
        
        # Check if phase_space calculation succeeded
        if phase_space is None:
            return jsonify({'success': False, 'error': 'Phase space calculation failed'}), 400
        
        # Detect microstructure state
        # Note: highs/lows not available in this endpoint context, pass None
        microstructure_state = detect_market_microstructure_state(closes, volumes, returns, None, None)
        
        # GARCH regime
        garch_vol_regime = calculate_garch_volatility_regime(closes)
        
        print(f"✓ Phase space data generated")
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'phaseSpace': phase_space,
            'microstructureState': microstructure_state,
            'garchRegime': garch_vol_regime,
            'currentPrice': float(closes[-1])
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/data: {error_trace}")
        error_msg = str(e) if str(e) else "Unknown error occurred"
        return jsonify({'success': False, 'error': error_msg}), 400

# NEW ENDPOINT: DETAILED GARCH ANALYSIS
@app.route('/api/garch-analysis', methods=['GET'])
def get_garch_analysis():
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y', interval='1d')
        
        if len(hist) == 0:
            return jsonify({'success': False, 'error': 'No data available'}), 400
        
        closes = hist['Close'].values
        returns = np.log(closes[1:] / closes[:-1]) * 100
        
        # Fit GARCH
        garch_results = fit_garch_model(returns)
        
        if garch_results is None:
            return jsonify({'success': False, 'error': 'GARCH fitting failed'}), 400
        
        # Get regime
        garch_vol_regime = calculate_garch_volatility_regime(closes)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'garch_params': garch_results,
            'vol_regime': garch_vol_regime,
            'returns': returns.tolist()[-100:],
            'conditional_vol': garch_results['conditional_volatility'][-100:]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# FEATURE ORDER - defines the order of features for ML models
# MODIFY THIS ONLY if you're using intraday features
FEATURE_ORDER = [
    "sigma_realized_pct", "sigma_garch_pct", "vol_ratio", "vol_trend",
    "compression_pctile", "trend_strength", "close_location", "gap_pct",
    "level_density", "oi_asym",
    # ADD THESE ONLY FOR INTRADAY:
    # "time_normalized", "time_to_close", "range_consumption"
]

# ============================================================================
# EQUATION ARCHITECTURE: FVECM + LEVEL REFINEMENT
# ============================================================================
#
# HARD PHYSICS (never learned, always fixed):
# - GARCH volatility estimation (fit_garch_model, calculate_garch_volatility_regime)
# - State machine detection (detect_market_microstructure_state)
# - Liquidity stress scoring (liquidity_stress_score)
#
# FVECM PIPELINE (theoretical_bounds package):
# - Layer 1: VECM on (mid_log, range_log) -- cointegration-based structural envelope
# - Layer 2: LightGBM residual corrector on normalized (e_mid, e_range)
# - Layer 3: Hard constraints + confidence bands (reconstruct.py)
# - Layer 4: Corrected HOD/LOD via LivePredictor (pre-open, zero look-ahead)
#
# STRUCTURAL CONSTRAINTS (enforced, never learned):
# - HOD > LOD (always enforced)
# - Range clipped to [0.1%, 15%] of close
# - Level refinement applied after FVECM prediction
#
# ============================================================================

# State and regime mappings for ML features
STATE_MAP = {
    "Thermal": 0,
    "Coherent": 1,
    "Fock": 2,
    "Unknown": -1  # Handle edge cases
}

REGIME_MAP = {
    "compressing": -1,
    "compression": -1,  # Alias
    "stable": 0,
    "normal": 0,  # Alias
    "expanding": 1,
    "expansion": 1  # Alias
}

# ML feature list (separate from FEATURE_ORDER for the state machine)
ML_FEATURES = [
    "sigma_daily_pct", "sigma_garch_pct", "vol_ratio", "vol_trend", "vol_of_vol",
    "micro_state", "micro_confidence", "jump_dominance", "jump_score", "velocity_variance",
    "garch_regime", "hmm_regime", "hurst_state", "regime_disagreement",
    "z_open", "abs_z_open", "z_prev_close",
    "compression_score", "range_consumption",
    "level_density_1sigma", "nearest_level_distance",
    "day_of_week", "is_opex_week",
    # NEW: Liquidity stress features
    "liquidity_stress", "amihud_score", "vol_drought_score",
    "jump_intensity", "wickiness_score", "tail_usage_mult"
]









# ============================================================================
# ADVANCED ML MODELS FOR HOD/LOD PREDICTION
# ============================================================================


if TORCH_AVAILABLE and nn is not None:
    class AttentionHODLOD(nn.Module):
        """
        Neural Network with Attention Mechanism for HOD/LOD prediction
        Learns which features matter WHEN
        """
        def __init__(self, n_features, hidden_dim=64):
            super().__init__()
            
            # Feature embedding
            self.feature_embed = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Prediction heads
            self.hod_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            self.lod_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # x: [batch, n_features]
            embedded = self.feature_embed(x)  # [batch, hidden_dim]
            
            # Attention weights
            attn_weights = torch.softmax(self.attention(embedded), dim=0)  # [batch, 1]
            
            # Weighted features
            attended = embedded * attn_weights
            
            # Predictions
            hod_pred = self.hod_head(attended)
            lod_pred = self.lod_head(attended)
            
            return hod_pred, lod_pred, attn_weights
else:
    # Dummy class when torch is not available
    class AttentionHODLOD:
        def __init__(self, *args, **kwargs):
            pass

if TORCH_AVAILABLE and nn is not None:
    class TemporalConvNet(nn.Module):
        """
        Temporal Convolution Network for sequence modeling
        Captures temporal patterns in HOD/LOD
        """
        def __init__(self, n_features, n_channels=[64, 64, 32], kernel_size=3):
            super().__init__()
            
            layers = []
            num_levels = len(n_channels)
            
            for i in range(num_levels):
                dilation = 2 ** i
                in_channels = n_features if i == 0 else n_channels[i-1]
                out_channels = n_channels[i]
                
                layers.append(nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation,
                    padding=(kernel_size-1) * dilation
                ))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
            
            self.network = nn.Sequential(*layers)
            
            # Output heads
            self.hod_head = nn.Linear(n_channels[-1], 1)
            self.lod_head = nn.Linear(n_channels[-1], 1)
        
        def forward(self, x):
            # x: [batch, seq_len, n_features]
            x = x.transpose(1, 2)  # [batch, n_features, seq_len]
            out = self.network(x)  # [batch, n_channels[-1], seq_len]
            out = out[:, :, -1]  # Take last timestep
            
            hod_pred = self.hod_head(out)
            lod_pred = self.lod_head(out)
            
            return hod_pred, lod_pred
else:
    # Dummy class when torch is not available
    class TemporalConvNet:
        def __init__(self, *args, **kwargs):
            pass


def realized_kernel_volatility(highs, lows, opens, closes, H=5):
    """
    Realized Kernel estimator with Parzen kernel
    More robust to microstructure noise than Garman-Klass
    
    H: bandwidth parameter (typically 5-10)
    """
    try:
        n = len(closes)
        if n < H * 2:
            return None
        
        # Log returns
        log_prices = np.log(np.concatenate([opens[:1], closes]))
        r = np.diff(log_prices)
        
        # Parzen kernel weights
        def parzen_weight(x, H):
            x = np.abs(x)
            if x <= 0.5:
                return 1 - 6*x**2 + 6*x**3
            elif x <= 1:
                return 2*(1-x)**3
            else:
                return 0
        
        # Kernel weights
        weights = np.array([parzen_weight(h/H, H) for h in range(-H, H+1)])
        weights = weights / weights.sum()
        
        # Realized kernel
        gamma = np.zeros(2*H + 1)
        for h in range(-H, H+1):
            if h == 0:
                gamma[H] = np.sum(r**2)
            else:
                gamma[H + h] = np.sum(r[max(0, -h):min(n, n-h)] * r[max(0, h):min(n, n+h)])
        
        RK = np.sum(weights * gamma)
        
        return np.sqrt(RK * 252)  # Annualized
    except Exception as e:
        print(f"realized_kernel_volatility failed: {e}")
        return None

if TORCH_AVAILABLE and nn is not None:
    class TransformerHODLOD(nn.Module):
        """
        Transformer with Positional Encoding for HOD/LOD prediction
        State-of-the-art for sequence prediction
        """
        def __init__(self, n_features, d_model=128, nhead=8, num_layers=3):
            super().__init__()
            
            self.embedding = nn.Linear(n_features, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.hod_head = nn.Linear(d_model, 1)
            self.lod_head = nn.Linear(d_model, 1)
        
        def forward(self, x, mask=None):
            # x: [batch, seq_len, n_features]
            x = self.embedding(x)  # [batch, seq_len, d_model]
            
            # Add positional encoding
            seq_len = x.size(1)
            position = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_encoding = self.positional_encoding(position, d_model=x.size(2))
            x = x + pos_encoding
            
            # Transformer
            out = self.transformer(x, src_key_padding_mask=mask)
            
            # Use last timestep
            out = out[:, -1, :]
            
            hod_pred = self.hod_head(out)
            lod_pred = self.lod_head(out)
            
            return hod_pred, lod_pred
        
        def positional_encoding(self, position, d_model):
            """Sinusoidal positional encoding"""
            pe = torch.zeros(position.size(0), position.size(1), d_model)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=position.device, dtype=torch.float32) * -(np.log(10000.0) / d_model))
            pe[:, :, 0::2] = torch.sin(position.float() * div_term)
            pe[:, :, 1::2] = torch.cos(position.float() * div_term)
            return pe
else:
    # Dummy class when torch is not available
    class TransformerHODLOD:
        def __init__(self, *args, **kwargs):
            pass

if TORCH_AVAILABLE and nn is not None:
    class QuantileSelector(nn.Module):
        """
        RL Agent for Adaptive Quantile Selection
        Instead of fixed 80th percentile, learn WHICH quantile to use for each state
        """
        def __init__(self, n_features, n_actions=10):
            """
            n_actions: 10 quantiles [0.1, 0.2, ..., 1.0]
            """
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        
        def forward(self, state):
            return self.network(state)
else:
    # Dummy class when torch is not available
    class QuantileSelector:
        def __init__(self, *args, **kwargs):
            pass

if TORCH_AVAILABLE:
    class DQNAgent:
        """
        Deep Q-Network Agent for Adaptive Quantile Selection
        """
        def __init__(self, n_features, n_actions=10):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for DQN agent")
            
            self.n_actions = n_actions
            self.quantiles = np.linspace(0.1, 1.0, n_actions)
            
            self.policy_net = QuantileSelector(n_features, n_actions)
            self.target_net = QuantileSelector(n_features, n_actions)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
            self.memory = []
            self.max_memory = 10000
            
            self.epsilon = 1.0  # Exploration rate
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            
            self.gamma = 0.95  # Discount factor
        
        def select_quantile(self, state):
            """Select which quantile to use"""
            import random
            if random.random() < self.epsilon:
                action = random.randrange(self.n_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            return self.quantiles[action], action
        
        def train_step(self, batch_size=32):
            """Train on a batch of experiences"""
            import random
            if len(self.memory) < batch_size:
                return
            
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Target Q values
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Loss
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
else:
    # Dummy class when torch is not available
    class DQNAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for DQN agent")


# ============================================================================

def ewma_volatility(returns, lam=0.94):
    """
    EWMA volatility (RiskMetrics style)
    
    Parameters:
    -----------
    returns : array-like
        Log returns (should be in decimal, not percentage)
    lam : float
        Decay factor (0.94 for daily, 0.97 for weekly)
    
    Returns:
    --------
    float : EWMA volatility (annualized if returns are daily)
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    
    # Initialize with first squared return
    var = returns[0] ** 2
    
    # Recursive update
    for r in returns[1:]:
        var = lam * var + (1 - lam) * r**2
    
    # Return annualized volatility (assumes daily returns)
    return np.sqrt(var * 252)

def parkinson_volatility(high, low):
    """
    Parkinson volatility estimator (range-based)
    More efficient than close-to-close when you have OHLC data
    
    Parameters:
    -----------
    high, low : array-like
        High and low prices over the same period
    
    Returns:
    --------
    float : Parkinson volatility (annualized)
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    
    if len(high) == 0 or len(low) == 0:
        return 0.0
    
    # Log of high/low ratio
    log_hl = np.log(high / (low + 1e-9))  # Avoid division by zero
    
    # Parkinson formula
    # Factor is 1/(4*ln(2)) ≈ 0.361
    variance = (1.0 / (4 * np.log(2))) * np.mean(log_hl**2)
    
    # Return annualized volatility (assumes daily data)
    return np.sqrt(variance * 252)

def rogers_satchell_volatility(open_, high, low, close):
    """
    Rogers-Satchell estimator (drift-independent, uses OHLC)
    Even more efficient than Parkinson for trending markets
    
    Parameters:
    -----------
    open_, high, low, close : array-like
        OHLC prices
    
    Returns:
    --------
    float : Rogers-Satchell volatility (annualized)
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    if len(o) == 0:
        return 0.0
    
    # Rogers-Satchell formula
    log_ho = np.log(h / (o + 1e-9))
    log_lo = np.log(l / (o + 1e-9))
    log_hc = np.log(h / (c + 1e-9))
    log_lc = np.log(l / (c + 1e-9))
    
    variance = np.mean(log_ho * log_hc + log_lo * log_lc)
    
    return np.sqrt(variance * 252)

def yang_zhang_volatility(open_, high, low, close, window=20):
    """
    Yang-Zhang estimator (combines overnight + intraday volatility)
    Most efficient unbiased estimator for OHLC data
    
    This is the GOLD STANDARD for volatility estimation
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    if len(c) < 2:
        return 0.0
    
    # Overnight volatility (close to open)
    log_co = np.log(o[1:] / (c[:-1] + 1e-9))
    overnight_var = np.var(log_co)
    
    # Open to close volatility (Rogers-Satchell)
    log_ho = np.log(h / (o + 1e-9))
    log_lo = np.log(l / (o + 1e-9))
    log_hc = np.log(h / (c + 1e-9))
    log_lc = np.log(l / (c + 1e-9))
    rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)
    
    # Close to close volatility
    log_cc = np.log(c[1:] / (c[:-1] + 1e-9))
    close_var = np.var(log_cc)
    
    # Yang-Zhang combination (optimal weights)
    k = 0.34 / (1 + (window + 1) / (window - 1))
    variance = overnight_var + k * close_var + (1 - k) * rs_var
    
    return np.sqrt(variance * 252)

def garman_klass_volatility(open_, high, low, close):
    """
    Garman-Klass volatility estimator (ANNUALIZED)
    More accurate than simple standard deviation as it uses OHLC data
    Returns annualized volatility (for backward compatibility)
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    log_hl = np.log(h / (l + 1e-9))
    log_co = np.log(c / (o + 1e-9))
    
    variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    return np.sqrt(np.mean(variance) * 252)

# ============================================================================
# DAILY (1-PERIOD) VOLATILITY ESTIMATORS
# For next-period predictions (NOT annualized)
# ============================================================================

def garman_klass_daily_volatility(open_, high, low, close):
    """
    Garman-Klass for SINGLE PERIOD (daily/intraday)
    Returns volatility for the NEXT PERIOD, not annualized
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    log_hl = np.log(h / (l + 1e-9))
    log_co = np.log(c / (o + 1e-9))
    
    variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    # Return DAILY volatility (no sqrt(252) multiplier)
    return np.sqrt(np.mean(variance))

def compute_session_sigma_from_range(hist: pd.DataFrame, window: int = 60) -> float:
    """
    Robust session volatility estimator from intraday/daily ranges.

    Uses a median ensemble of:
      - Parkinson
      - Rogers–Satchell
      - Garman–Klass
    over a rolling window, then takes the latest sigma.
    """
    if len(hist) < 10:
        raise ValueError("Not enough data for range-based session sigma")

    recent = hist.tail(window).copy()

    # Per-bar variances
    var_pk = []
    var_rs = []
    var_gk = []
    for _, row in recent.iterrows():
        try:
            var_pk.append(parkinson_daily_volatility([row["High"]], [row["Low"]])**2)
        except Exception:
            var_pk.append(np.nan)
        try:
            var_rs.append(rogers_satchell_daily_volatility([row["Open"]], [row["High"]], [row["Low"]], [row["Close"]])**2)
        except Exception:
            var_rs.append(np.nan)
        try:
            var_gk.append(garman_klass_daily_volatility([row["Open"]], [row["High"]], [row["Low"]], [row["Close"]])**2)
        except Exception:
            var_gk.append(np.nan)

    var_pk = np.array(var_pk, dtype=float)
    var_rs = np.array(var_rs, dtype=float)
    var_gk = np.array(var_gk, dtype=float)

    # Rolling means and median ensemble
    roll_pk = pd.Series(var_pk).rolling(window=min(window, len(var_pk)), min_periods=5).mean().values
    roll_rs = pd.Series(var_rs).rolling(window=min(window, len(var_rs)), min_periods=5).mean().values
    roll_gk = pd.Series(var_gk).rolling(window=min(window, len(var_gk)), min_periods=5).mean().values

    var_stack = np.vstack([roll_pk, roll_rs, roll_gk])
    var_median = np.nanmedian(var_stack, axis=0)

    if not np.isfinite(var_median[-1]) or var_median[-1] <= 0:
        raise ValueError("Invalid median variance from range estimators")

    sigma_session = float(np.sqrt(var_median[-1]))  # decimal, e.g. 0.015 = 1.5%
    return sigma_session

def parkinson_daily_volatility(high, low):
    """
    Parkinson for SINGLE PERIOD
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    
    if len(high) == 0 or len(low) == 0:
        return 0.0
    
    log_hl = np.log(high / (low + 1e-9))
    variance = (1.0 / (4 * np.log(2))) * np.mean(log_hl**2)
    
    # No annualization - this is for next period
    return np.sqrt(variance)

def rogers_satchell_daily_volatility(open_, high, low, close):
    """
    Rogers-Satchell for SINGLE PERIOD
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    if len(o) == 0:
        return 0.0
    
    log_ho = np.log(h / (o + 1e-9))
    log_lo = np.log(l / (o + 1e-9))
    log_hc = np.log(h / (c + 1e-9))
    log_lc = np.log(l / (c + 1e-9))
    
    variance = np.mean(log_ho * log_hc + log_lo * log_lc)
    
    # No annualization
    return np.sqrt(variance)

def yang_zhang_daily_volatility(open_, high, low, close, window=20):
    """
    Yang-Zhang for SINGLE PERIOD (most accurate)
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    if len(c) < 2:
        return 0.0
    
    log_co = np.log(o[1:] / (c[:-1] + 1e-9))
    overnight_var = np.var(log_co)
    
    log_ho = np.log(h / (o + 1e-9))
    log_lo = np.log(l / (o + 1e-9))
    log_hc = np.log(h / (c + 1e-9))
    log_lc = np.log(l / (c + 1e-9))
    rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)
    
    log_cc = np.log(c[1:] / (c[:-1] + 1e-9))
    close_var = np.var(log_cc)
    
    k = 0.34 / (1 + (window + 1) / (window - 1))
    variance = overnight_var + k * close_var + (1 - k) * rs_var
    
    # No annualization
    return np.sqrt(variance)

def compute_optimal_daily_sigma(
    hist: pd.DataFrame,
    garch_vol: float = None,
    window: int = 20
) -> dict:
    """
    Optimal DAILY (1-period) volatility estimation
    Returns volatility for NEXT PERIOD, not annualized
    
    Parameters:
    -----------
    hist : pd.DataFrame
        Must have columns: Open, High, Low, Close
    garch_vol : float, optional
        GARCH forecast volatility (annualized %) - will be converted to daily
    window : int
        Lookback window for estimators
    
    Returns:
    --------
    dict : {
        'sigma_daily': daily volatility (decimal, e.g. 0.015 for 1.5% expected move),
        'sigma_daily_pct': daily volatility in percentage,
        'sigma_price': expected price range (sigma_daily * current_price),
        'components': breakdown of each estimator,
        'method': which estimators were used
    }
    """
    if len(hist) < max(window, 20):
        raise ValueError(f"Need at least {max(window, 20)} periods of data")
    
    recent = hist.tail(window)
    opens = recent['Open'].values
    highs = recent['High'].values
    lows = recent['Low'].values
    closes = recent['Close'].values
    current_price = closes[-1]
    
    components = {}
    
    # 1. Parkinson (fast, range-based)
    try:
        sigma_park = parkinson_daily_volatility(highs, lows)
        components['parkinson'] = sigma_park
    except:
        sigma_park = None
    
    # 2. Garman-Klass (proven)
    try:
        sigma_gk = garman_klass_daily_volatility(opens, highs, lows, closes)
        components['garman_klass'] = sigma_gk
    except:
        sigma_gk = None
    
    # 3. Rogers-Satchell (drift-independent)
    try:
        sigma_rs = rogers_satchell_daily_volatility(opens, highs, lows, closes)
        components['rogers_satchell'] = sigma_rs
    except:
        sigma_rs = None
    
    # 4. Yang-Zhang (most efficient)
    try:
        sigma_yz = yang_zhang_daily_volatility(opens, highs, lows, closes, window=window)
        components['yang_zhang'] = sigma_yz
    except:
        sigma_yz = None
    
    # 5. GARCH forecast (convert from annualized to daily)
    if garch_vol is not None:
        # GARCH gives annualized %, convert to daily decimal
        garch_daily = (garch_vol / 100.0) / np.sqrt(252)
        components['garch'] = garch_daily
    
    # Ensemble weighting (Yang-Zhang is best if available)
    weights = {}
    
    if 'yang_zhang' in components and components['yang_zhang'] > 0:
        weights = {
            'yang_zhang': 0.35,
            'garman_klass': 0.25,
            'rogers_satchell': 0.15,
            'parkinson': 0.15,
            'garch': 0.10 if garch_vol else 0
        }
    elif 'garman_klass' in components:
        weights = {
            'garman_klass': 0.40,
            'parkinson': 0.25,
            'rogers_satchell': 0.20 if sigma_rs else 0,
            'garch': 0.15 if garch_vol else 0
        }
    else:
        weights = {
            'parkinson': 0.60 if sigma_park else 0,
            'garch': 0.40 if garch_vol else 0
        }
    
    # Normalize weights
    total_weight = sum(w for est, w in weights.items() if est in components and components[est] is not None)
    
    if total_weight == 0:
        raise ValueError("No valid volatility estimators available")
    
    weights = {k: v / total_weight for k, v in weights.items() if k in components and components[k] is not None}
    
    # Compute final daily volatility
    sigma_daily = sum(components[est] * weight for est, weight in weights.items())
    
    return {
        'sigma_daily': float(sigma_daily),  # Decimal (e.g., 0.015)
        'sigma_daily_pct': float(sigma_daily * 100),  # Percentage (e.g., 1.5)
        'sigma_price': float(sigma_daily * current_price),  # Price units
        'components': {k: float(v) for k, v in components.items()},
        'weights': {k: float(v) for k, v in weights.items()},
        'method': 'daily_ensemble',
        'n_estimators': len(components)
    }

# GARMAN-KLASS VOLATILITY ESTIMATOR (backward compatibility)
def garman_klass_daily_sigma_pct(hist):
    """
    Calculate daily volatility using Garman-Klass estimator
    More accurate than simple standard deviation as it uses OHLC data
    (Backward compatibility wrapper)
    """
    # hist must have Open, High, Low, Close
    o = hist["Open"].values
    h = hist["High"].values
    l = hist["Low"].values
    c = hist["Close"].values
    
    sigma = garman_klass_volatility(o, h, l, c)
    # Convert from decimal to percentage
    return float(sigma * 100)

# ============================================================================
# SESSION VOLATILITY (for next-period prediction)
# ============================================================================

def compute_session_volatility(hist: pd.DataFrame, window: int = 60) -> dict:
    """
    Compute volatility for NEXT SESSION (intraday or daily)
    
    Returns both:
    - Annualized volatility (for GARCH comparison)
    - Session volatility (for HOD/LOD prediction)
    
    This function bridges the gap between your annualized GARCH
    and the actual expected move for the next trading session.
    """
    if len(hist) < 20:
        raise ValueError("Need at least 20 periods")
    
    recent = hist.tail(window)
    opens = recent['Open'].values
    highs = recent['High'].values
    lows = recent['Low'].values
    closes = recent['Close'].values
    current_price = closes[-1]
    
    # 1. Try robust range-based ensemble (Parkinson + RS + GK) for session sigma
    try:
        sigma_session = compute_session_sigma_from_range(recent, window=min(window, len(recent)))
        method = 'range_ensemble_session'
    except Exception as e:
        print(f"⚠ compute_session_volatility: range-based sigma failed ({e}), falling back to GK/returns")
        # 2. Fallback: DAILY (non-annualized) volatility using Garman-Klass
        opens = np.maximum(opens, 1e-9)
        highs = np.maximum(highs, opens * 0.99)
        lows = np.maximum(lows, opens * 0.99)
        closes = np.maximum(closes, lows)
        
        log_hl = np.log(highs / (lows + 1e-9))
        log_co = np.log(closes / (opens + 1e-9))
        variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        variance = np.maximum(variance, 1e-9)  # Ensure non-negative
        mean_variance = np.mean(variance)
        if mean_variance <= 0 or not np.isfinite(mean_variance):
            returns = np.diff(np.log(closes))
            mean_variance = np.var(returns)
            if mean_variance <= 0 or not np.isfinite(mean_variance):
                price_range = np.max(highs) - np.min(lows)
                mean_variance = (price_range / current_price) ** 2 / max(len(closes), 1)
        sigma_session = np.sqrt(mean_variance)
        method = 'garman_klass_session_fallback'
    
    # Ensure sigma_session is valid
    if sigma_session <= 0 or not np.isfinite(sigma_session):
        # Fallback: use 1% of current price as default volatility
        sigma_session = 0.01
        print(f"⚠ compute_session_volatility: Invalid sigma_session, using fallback: {sigma_session}")
    
    # Annualized volatility (for GARCH comparison)
    sigma_annual = sigma_session * np.sqrt(252)  # Annualize for comparison
    
    sigma_price = sigma_session * current_price
    
    # Ensure sigma_price is valid and reasonable
    if sigma_price <= 0 or not np.isfinite(sigma_price):
        sigma_price = current_price * 0.01  # 1% of price as default
        print(f"⚠ compute_session_volatility: Invalid sigma_price, using fallback: {sigma_price:.4f}")
    
    return {
        'sigma_session': float(sigma_session),          # Next period vol (decimal)
        'sigma_session_pct': float(sigma_session * 100),# Next period vol (%)
        'sigma_annual_pct': float(sigma_annual * 100),  # Annualized (%)
        'sigma_price': float(sigma_price),              # Expected $ range
        'method': method
    }

def compute_volatility_cone(hist: pd.DataFrame, window: int = 252) -> dict:
    """
    Build historical volatility distribution to calibrate expectations
    Shows if current vol is 10th percentile (quiet) or 90th percentile (explosive)
    
    Parameters:
    -----------
    hist : pd.DataFrame
        Historical price data with Open, High, Low, Close
    window : int
        Lookback window for historical distribution (default: 252 trading days)
    
    Returns:
    --------
    dict : {
        'current_vol': current session volatility (%),
        'percentile': percentile rank (0-1),
        'p10', 'p25', 'p50', 'p75', 'p90': volatility percentiles,
        'regime': 'compressed', 'normal', or 'elevated'
    }
    """
    if len(hist) < 40:
        # Not enough data for meaningful distribution
        return {
            'current_vol': None,
            'percentile': 0.5,
            'p10': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p90': None,
            'regime': 'normal'
        }
    
    if len(hist) < window:
        window = len(hist)
    
    # Rolling session volatility
    rolling_vols = []
    for i in range(20, len(hist)):
        if i < 20:
            continue
        try:
            window_data = hist.iloc[i-20:i]
            if len(window_data) >= 20:
                vol_result = compute_session_volatility(window_data, window=20)
                rolling_vols.append(vol_result['sigma_session_pct'])
        except Exception as e:
            # Skip if calculation fails for this window
            continue
    
    if len(rolling_vols) == 0:
        return {
            'current_vol': None,
            'percentile': 0.5,
            'p10': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p90': None,
            'regime': 'normal'
        }
    
    rolling_vols = np.array(rolling_vols)
    
    # Current vol
    try:
        current_vol_result = compute_session_volatility(hist.tail(20), window=20)
        current_vol = current_vol_result['sigma_session_pct']
    except Exception:
        # Fallback to simple std if compute_session_volatility fails
        returns = np.log(hist['Close'].values[1:] / hist['Close'].values[:-1])
        current_vol = float(np.std(returns) * 100)
    
    # Percentile ranking
    sorted_vols = np.sort(rolling_vols)
    percentile = float(np.searchsorted(sorted_vols, current_vol) / len(rolling_vols))
    
    return {
        'current_vol': float(current_vol),
        'percentile': float(percentile),
        'p10': float(np.percentile(rolling_vols, 10)),
        'p25': float(np.percentile(rolling_vols, 25)),
        'p50': float(np.percentile(rolling_vols, 50)),
        'p75': float(np.percentile(rolling_vols, 75)),
        'p90': float(np.percentile(rolling_vols, 90)),
        'regime': 'compressed' if percentile < 0.25 else 'elevated' if percentile > 0.75 else 'normal'
    }

def compute_range_consumption(hist: pd.DataFrame, current_price: float, sigma_price: float) -> dict:
    """
    How much of today's expected range has already been consumed?
    Critical for intraday - if you've already hit 2σ by 11am, unlikely to go further
    
    Parameters:
    -----------
    hist : pd.DataFrame
        Historical price data with datetime index
    current_price : float
        Current price
    sigma_price : float
        Expected price range (in price units, not percentage)
    
    Returns:
    --------
    dict : {
        'consumed_pct': percentage of expected range consumed (0-1),
        'remaining_up': remaining upward headroom,
        'remaining_down': remaining downward headroom,
        'session_high': session high so far,
        'session_low': session low so far,
        'bias': 'exhausted' or 'active'
    }
    """
    try:
        # Get today's data
        if len(hist) == 0:
            return {
                'consumed_pct': 0.0,
                'remaining_up': sigma_price,
                'remaining_down': sigma_price,
                'session_high': current_price,
                'session_low': current_price,
                'bias': 'active'
            }
        
        # Get the most recent date
        today = hist.index[-1]
        if hasattr(today, 'date'):
            today_date = today.date()
        else:
            # If index is already date, use it directly
            today_date = today
        
        # Filter today's data
        if hasattr(hist.index, 'date'):
            today_data = hist[hist.index.date == today_date]
        else:
            # If index doesn't have date attribute, assume all data is from today
            today_data = hist.tail(100)  # Use recent data as approximation
        
        if len(today_data) == 0:
            return {
                'consumed_pct': 0.0,
                'remaining_up': sigma_price,
                'remaining_down': sigma_price,
                'session_high': current_price,
                'session_low': current_price,
                'bias': 'active'
            }
        
        session_high = float(today_data['High'].max()) if 'High' in today_data.columns else current_price
        session_low = float(today_data['Low'].min()) if 'Low' in today_data.columns else current_price
        session_range = session_high - session_low
        
        # Expected range for full session (2σ envelope)
        expected_range = 2 * sigma_price
        
        # How much consumed
        consumed_pct = session_range / expected_range if expected_range > 0 else 0
        
        # Remaining headroom
        remaining_up = max(0, (current_price + sigma_price) - session_high)
        remaining_down = max(0, session_low - (current_price - sigma_price))
        
        return {
            'consumed_pct': float(min(consumed_pct, 1.0)),
            'remaining_up': float(remaining_up),
            'remaining_down': float(remaining_down),
            'session_high': float(session_high),
            'session_low': float(session_low),
            'bias': 'exhausted' if consumed_pct > 0.85 else 'active'
        }
    except Exception as e:
        print(f"⚠ Range consumption calculation failed: {e}")
        return {
            'consumed_pct': 0.0,
            'remaining_up': sigma_price,
            'remaining_down': sigma_price,
            'session_high': current_price,
            'session_low': current_price,
            'bias': 'active'
        }

def compute_mtf_confluence(
    ticker: str,
    spot: float,
    sigma_price: float,
    micro_state: str,
    lookback: int = 20
) -> dict:
    """
    Multi-timeframe structural confluence.
    
    Purpose:
    - Validate whether higher timeframes CARE about the same zone
    - Improve confidence, not expand range
    - Act as soft structural ceilings/floors
    
    Returns STRUCTURE, not signals.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    spot : float
        Current price
    sigma_price : float
        Expected price range (in price units)
    micro_state : str
        Market microstructure state (Fock, Thermal, Coherent)
    lookback : int
        Number of periods to look back for recent high/low
    
    Returns:
    --------
    dict : {
        'apply': whether to apply MTF confluence,
        'reason': why it applies or doesn't,
        'resistance': resistance level from MTF confluence,
        'support': support level from MTF confluence,
        'confidence_boost': confidence boost (0-0.15),
        'details': additional information
    }
    """
    timeframes = ['1h', '4h', '1d']
    mtf_levels = []

    # Fock regimes do not respect HTF structure
    if micro_state == "Fock" or sigma_price <= 0:
        return {
            "apply": False,
            "reason": "fock_or_invalid_sigma",
            "resistance": None,
            "support": None,
            "confidence_boost": 0.0,
            "details": {}
        }

    for tf in timeframes:
        try:
            hist = yf.Ticker(ticker).history(period="1mo", interval=tf)
            if len(hist) < lookback:
                continue

            recent_high = float(hist['High'].iloc[-lookback:].max())
            recent_low  = float(hist['Low'].iloc[-lookback:].min())

            dist_high_sigma = abs(recent_high - spot) / sigma_price if sigma_price > 0 else float('inf')
            dist_low_sigma  = abs(spot - recent_low) / sigma_price if sigma_price > 0 else float('inf')

            mtf_levels.append({
                "tf": tf,
                "high": recent_high,
                "low": recent_low,
                "high_dist_sigma": dist_high_sigma,
                "low_dist_sigma": dist_low_sigma
            })

        except Exception as e:
            print(f"⚠ MTF confluence failed for {tf}: {e}")
            continue

    # Identify clusters (within 0.75σ = actionable today)
    resistance_cluster = [
        l for l in mtf_levels if l["high_dist_sigma"] <= 0.75
    ]
    support_cluster = [
        l for l in mtf_levels if l["low_dist_sigma"] <= 0.75
    ]

    resistance_level = (
        float(np.mean([l["high"] for l in resistance_cluster]))
        if len(resistance_cluster) >= 2 else None
    )

    support_level = (
        float(np.mean([l["low"] for l in support_cluster]))
        if len(support_cluster) >= 2 else None
    )

    # Confidence logic (soft, capped)
    confidence_boost = 0.0
    if resistance_level or support_level:
        confidence_boost = min(0.05 * max(len(resistance_cluster), len(support_cluster)), 0.15)

    return {
        "apply": bool(resistance_level or support_level),
        "reason": "mtf_structure_confirmed" if (resistance_level or support_level) else "no_cluster",
        "resistance": resistance_level,
        "support": support_level,
        "confidence_boost": confidence_boost,
        "details": {
            "levels": mtf_levels,
            "resistance_count": len(resistance_cluster),
            "support_count": len(support_cluster)
        }
    }

# ============================================================================
# OPTIMAL VOLATILITY ENSEMBLE
# ============================================================================

def compute_optimal_sigma(
    hist: pd.DataFrame,
    garch_vol: float = None,
    iv_surface: float = None,
    use_iv: bool = True,
    window: int = 20
) -> dict:
    """
    Optimal volatility estimation using ensemble of estimators
    
    Parameters:
    -----------
    hist : pd.DataFrame
        Must have columns: Open, High, Low, Close, Volume
    garch_vol : float, optional
        GARCH forecast volatility (annualized %)
    iv_surface : float, optional
        Implied volatility from options (annualized %)
    use_iv : bool
        Whether to use IV in ensemble
    window : int
        Lookback window for estimators
    
    Returns:
    --------
    dict : {
        'sigma_final': final blended volatility,
        'components': breakdown of each estimator,
        'weights': weights used in ensemble,
        'method': which estimators were used
    }
    """
    # Ensure we have enough data
    if len(hist) < max(window, 20):
        raise ValueError(f"Need at least {max(window, 20)} periods of data")
    
    # Extract recent window
    recent = hist.tail(window)
    opens = recent['Open'].values
    highs = recent['High'].values
    lows = recent['Low'].values
    closes = recent['Close'].values
    
    # Calculate returns for EWMA
    returns = np.log(closes[1:] / closes[:-1])
    
    # ===== COMPUTE ALL ESTIMATORS =====
    
    components = {}
    
    # 1. EWMA (forward-looking, adapts to recent changes)
    try:
        sigma_ewma = ewma_volatility(returns, lam=0.94)
        components['ewma'] = sigma_ewma
    except:
        sigma_ewma = None
    
    # 2. Parkinson (efficient, range-based)
    try:
        sigma_parkinson = parkinson_volatility(highs, lows)
        components['parkinson'] = sigma_parkinson
    except:
        sigma_parkinson = None
    
    # 3. Garman-Klass (your existing implementation)
    try:
        sigma_gk = garman_klass_volatility(opens, highs, lows, closes)
        components['garman_klass'] = sigma_gk
    except:
        sigma_gk = None
    
    # 4. Rogers-Satchell (drift-independent)
    try:
        sigma_rs = rogers_satchell_volatility(opens, highs, lows, closes)
        components['rogers_satchell'] = sigma_rs
    except:
        sigma_rs = None
    
    # 5. Yang-Zhang (most efficient)
    try:
        sigma_yz = yang_zhang_volatility(opens, highs, lows, closes, window=window)
        components['yang_zhang'] = sigma_yz
    except:
        sigma_yz = None
    
    # 6. GARCH forecast (forward-looking)
    if garch_vol is not None:
        # Convert from percentage to decimal if needed
        garch_vol_decimal = garch_vol / 100.0 if garch_vol > 1.0 else garch_vol
        components['garch'] = garch_vol_decimal
    
    # 7. Implied Volatility (market's expectation)
    if use_iv and iv_surface is not None:
        # Convert from percentage to decimal if needed
        iv_decimal = iv_surface / 100.0 if iv_surface > 1.0 else iv_surface
        components['implied_vol'] = iv_decimal
    
    # ===== ENSEMBLE WEIGHTING =====
    
    # Define weights based on estimator quality
    # Yang-Zhang is theoretically optimal, so weight it highest
    weights = {}
    
    if 'yang_zhang' in components and components['yang_zhang'] > 0:
        # Yang-Zhang available (best case)
        weights = {
            'yang_zhang': 0.30,
            'ewma': 0.20,
            'garman_klass': 0.15,
            'rogers_satchell': 0.10,
            'parkinson': 0.10,
            'garch': 0.10 if garch_vol else 0,
            'implied_vol': 0.05 if (use_iv and iv_surface) else 0
        }
    elif 'garman_klass' in components:
        # Fallback to Garman-Klass + EWMA
        weights = {
            'garman_klass': 0.35,
            'ewma': 0.30,
            'parkinson': 0.15,
            'rogers_satchell': 0.10 if sigma_rs else 0,
            'garch': 0.05 if garch_vol else 0,
            'implied_vol': 0.05 if (use_iv and iv_surface) else 0
        }
    else:
        # Worst case: only EWMA and Parkinson
        weights = {
            'ewma': 0.50 if sigma_ewma else 0,
            'parkinson': 0.30 if sigma_parkinson else 0,
            'garch': 0.15 if garch_vol else 0,
            'implied_vol': 0.05 if (use_iv and iv_surface) else 0
        }
    
    # Normalize weights to sum to 1
    total_weight = sum(w for est, w in weights.items() if est in components and components[est] is not None)
    
    if total_weight == 0:
        raise ValueError("No valid volatility estimators available")
    
    weights = {k: v / total_weight for k, v in weights.items() if k in components and components[k] is not None}
    
    # ===== COMPUTE FINAL BLENDED VOLATILITY =====
    
    sigma_final = sum(components[est] * weight for est, weight in weights.items())
    
    # ===== DIAGNOSTICS =====
    
    # Check for volatility regime
    all_vols = [v for v in components.values() if v is not None and v > 0]
    vol_spread = (max(all_vols) - min(all_vols)) / np.mean(all_vols) if all_vols else 0
    
    regime = "stable" if vol_spread < 0.15 else "dispersed" if vol_spread < 0.30 else "extreme"
    
    return {
        'sigma_final': float(sigma_final),
        'components': {k: float(v) for k, v in components.items()},
        'weights': {k: float(v) for k, v in weights.items()},
        'vol_spread': float(vol_spread),
        'regime': regime,
        'method': 'ensemble',
        'n_estimators': len(components)
    }

# ============================================================================
# SIMPLIFIED API FOR YOUR EXISTING CODE
# ============================================================================

def compute_daily_sigma_pct(hist: pd.DataFrame, garch_vol=None, iv_surface=None) -> float:
    """
    Drop-in replacement for your existing garman_klass_daily_sigma_pct
    
    Returns:
    --------
    float : Daily volatility in percentage (annualized)
    """
    result = compute_optimal_sigma(hist, garch_vol, iv_surface)
    # Convert from decimal to percentage
    return float(result['sigma_final'] * 100)

# ============================================================================
# STATE MACHINE ENHANCEMENTS - FOR IMPROVED HOD/LOD PREDICTIONS
# ============================================================================







def add_intraday_features(feature_dict, timestamp, hist):
    """
    Add time-of-day context for intraday timeframes - NEW FUNCTION
    Call this AFTER compute_features_daily() to augment features
    """
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.to_datetime(timestamp)
    
    market_open = timestamp.replace(hour=9, minute=30, second=0)
    minutes_since_open = (timestamp - market_open).total_seconds() / 60
    time_normalized = min(max(minutes_since_open / 390.0, 0.0), 1.0)
    time_to_close = 1.0 - time_normalized
    
    today_mask = hist.index.date == timestamp.date()
    if today_mask.any():
        today_data = hist[today_mask]
        day_high = today_data['High'].max()
        day_low = today_data['Low'].min()
        current_price = today_data['Close'].iloc[-1]
        
        realized_range_pct = (day_high - day_low) / current_price * 100
        sigma_daily = feature_dict.get('sigma_realized_pct', 1.0)
        range_consumption = realized_range_pct / sigma_daily if sigma_daily > 0 else 0
    else:
        range_consumption = 0
    
    feature_dict.update({
        'time_normalized': float(time_normalized),
        'time_to_close': float(time_to_close),
        'range_consumption': float(min(range_consumption, 2.0))
    })
    
    return feature_dict

# Helper functions for level-constrained HOD/LOD prediction
def state_policy(state):
    """
    State-aware policy for level selection
    Returns timeframe weights, bound preference, and minimum strength threshold
    """
    name = state.get("state", "UNKNOWN").upper()
    lss = state.get("liquidity_stress", 0.5)
    
    # Map your microstructure states to policy
    if name in ("FOCK", "TRENDING", "EXPANSION"):
        # Trending/volatile: favor higher timeframes, levels near bounds
        # LOWERED: min_strength to allow more levels through (was 0.55/0.65)
        return {
            "tf_w": {"1m": 0.3, "5m": 0.6, "15m": 0.9, "1h": 1.0, "4h": 1.1, "1d": 1.2},
            "bound_power": 1.3,      # favor levels near theoretical bound
            "min_strength": 0.40 if lss < 0.6 else 0.50
        }
    elif name in ("THERMAL", "COMPRESSION", "CHOPPY"):
        # Ranging/quiet: favor lower timeframes, earlier pivot points
        # LOWERED: min_strength to allow more levels through (was 0.60/0.70)
        return {
            "tf_w": {"1m": 0.7, "5m": 1.0, "15m": 1.1, "1h": 0.9, "4h": 0.7, "1d": 0.6},
            "bound_power": 0.7,      # penalize levels too close to bound
            "min_strength": 0.45 if lss < 0.6 else 0.55
        }
    else:  # COHERENT or UNKNOWN
        return {
            "tf_w": {"1m": 0.8, "5m": 0.9, "15m": 1.0, "1h": 1.0, "4h": 0.9, "1d": 0.8},
            "bound_power": 1.0,
            "min_strength": 0.6
        }

def score_candidate(level, spot, bound, side, policy, timeframe):
    """
    Score a level candidate based on:
    - Strength from detection algorithm
    - Timeframe weight (from policy)
    - Distance to theoretical bound (via bound_power)
    """
    price = level["price"]
    strength = level.get("strength", 0.5)
    
    # Get timeframe weight (default to current timeframe if level doesn't specify)
    level_tf = level.get("timeframe", timeframe)
    tf_w = policy["tf_w"].get(level_tf, 1.0)
    
    # Calculate position within theoretical envelope (0 = at spot, 1 = at bound)
    if side == "HOD":
        denom = max(1e-9, bound - spot)
        near_bound = (price - spot) / denom  # 0..1
    else:  # LOD
        denom = max(1e-9, spot - bound)
        near_bound = (spot - price) / denom  # 0..1
    
    # Clamp to [0, 1]
    near_bound = max(0.0, min(1.0, near_bound))
    
    # Apply bound_power:
    # > 1: favor levels near bound (late pivot)
    # < 1: favor levels near spot (early pivot)
    bound_component = near_bound ** policy["bound_power"]
    
    # Confluence bonus
    confluence_count = level.get("confluence_count", 1)
    confluence_mult = 1.0 + (confluence_count - 1) * 0.15
    
    # Combined score
    return strength * tf_w * bound_component * confluence_mult

def refine_extrema_with_levels(spot, hod_th, lod_th, levels, state, timeframe="1d", lower_tf_lod=None):
    """
    Refine theoretical HOD/LOD bounds using detected levels
    
    Parameters:
    -----------
    spot : float
        Current price
    hod_th : float
        Theoretical HOD (from sigma/GARCH)
    lod_th : float
        Theoretical LOD (from sigma/GARCH)
    levels : list
        All detected levels (from your various algorithms)
    state : dict
        Microstructure state (must have 'state' key, optionally 'liquidity_stress')
    timeframe : str
        Current timeframe being analyzed
    lower_tf_lod : float, optional
        Lower timeframe theoretical LOD (used as floor to prevent unbelievable LOD)
    
    Returns:
    --------
    (refined_hod, refined_lod, debug_info)
    """
    # Get state-specific policy
    policy = state_policy(state)
    
    # Filter candidates: must be inside envelope and meet minimum strength
    hod_cands = [
        l for l in levels 
        if spot < l["price"] <= hod_th 
        and l.get("strength", 0.5) >= policy["min_strength"]
    ]
    
    lod_cands = [
        l for l in levels 
        if lod_th <= l["price"] < spot 
        and l.get("strength", 0.5) >= policy["min_strength"]
    ]
    
    # Score and select best HOD candidate
    if not hod_cands:
        refined_hod = hod_th
        best_hod = None
    else:
        # Score all candidates (filter out None/Invalid scores)
        scored = []
        for l in hod_cands:
            try:
                score = score_candidate(l, spot, hod_th, "HOD", policy, timeframe)
                if score is not None and np.isfinite(score):
                    scored.append((l, score))
            except Exception:
                continue  # Skip malformed levels
        
        # Validate scored list is not empty
        if scored:
            best_hod, best_score = max(scored, key=lambda x: x[1])
            refined_hod = best_hod["price"]
        else:
            refined_hod = hod_th
            best_hod = None
    
    # Score and select best LOD candidate
    if not lod_cands:
        refined_lod = lod_th
        best_lod = None
    else:
        # Score all candidates (filter out None/Invalid scores)
        scored = []
        for l in lod_cands:
            try:
                score = score_candidate(l, spot, lod_th, "LOD", policy, timeframe)
                if score is not None and np.isfinite(score):
                    scored.append((l, score))
            except Exception:
                continue  # Skip malformed levels
        
        # Validate scored list is not empty
        if scored:
            best_lod, best_score = max(scored, key=lambda x: x[1])
            refined_lod = best_lod["price"]
        else:
            refined_lod = lod_th
            best_lod = None
    
    # Validate LOD: Use lower timeframe theoretical LOD as floor
    # If predicted LOD is below lower TF theoretical LOD, it's "unbelievable"
    if lower_tf_lod is not None and refined_lod < lower_tf_lod:
        # LOD is too low - use lower TF theoretical LOD as minimum
        print(f"⚠ LOD at unbelievable level (${refined_lod:.2f} < ${lower_tf_lod:.2f}). Using lower TF theoretical LOD.")
        refined_lod = lower_tf_lod
        best_lod = None  # Reset since we're using theoretical
    
    # Debug info
    debug = {
        "policy": policy,
        "state": state.get("state", "UNKNOWN"),
        "n_hod_candidates": len(hod_cands),
        "n_lod_candidates": len(lod_cands),
        "best_hod": best_hod,
        "best_lod": best_lod,
        "used_theoretical_hod": best_hod is None,
        "used_theoretical_lod": best_lod is None
    }
    
    return refined_hod, refined_lod, debug


def calculate_level_confidence(predicted_price, levels, current_price, sigma_price):
    """
    Calculate confidence in the prediction based on:
    1. How many levels are nearby
    2. Strength of nearby levels
    3. Distance from current price (too far = less confident)
    """
    if not levels:
        return 0.5
    
    # Find levels near prediction (within 1% of predicted price)
    nearby = [l for l in levels if abs(l['price'] - predicted_price) < predicted_price * 0.01]
    
    if not nearby:
        # No levels near prediction, lower confidence
        return 0.4
    
    # Average strength of nearby levels
    avg_strength = np.mean([l.get('strength', 0.5) for l in nearby])
    
    # Number of nearby levels (more = higher confidence)
    count_score = min(len(nearby) / 3, 1.0)
    
    # Distance from current (farther = less confident)
    distance_pct = abs(predicted_price - current_price) / current_price
    distance_factor = 1.0 / (1.0 + distance_pct * 10)
    
    confidence = (
        avg_strength * 0.5 +
        count_score * 0.3 +
        distance_factor * 0.2
    )
    
    return float(np.clip(confidence, 0.0, 1.0))

# ============================================================================
# VOLUME PROFILE & LEVEL REACTION ANALYSIS
# ============================================================================

def calculate_volume_profile(highs, lows, closes, volumes, bins=30):
    """
    Calculate volume profile (value areas) for directional understanding
    
    Returns:
    --------
    dict: {
        'poc': float,  # Point of Control (highest volume price)
        'value_area_high': float,  # 70% value area high
        'value_area_low': float,   # 70% value area low
        'profile': list,  # [(price, volume), ...]
        'volume_distribution': dict  # {price_bin: volume}
    }
    """
    if len(closes) == 0:
        return None
    
    price_range = (np.max(highs) - np.min(lows))
    if price_range == 0:
        return None
    
    # Create price bins
    min_price = np.min(lows)
    max_price = np.max(highs)
    bin_edges = np.linspace(min_price, max_price, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Distribute volume across price bins
    volume_distribution = np.zeros(bins)
    
    for i in range(len(closes)):
        # For each bar, distribute volume across the price range it traded
        bar_low = lows[i]
        bar_high = highs[i]
        bar_volume = volumes[i]
        
        # Find which bins this bar overlaps
        low_bin = np.searchsorted(bin_edges, bar_low) - 1
        high_bin = np.searchsorted(bin_edges, bar_high)
        
        low_bin = max(0, min(low_bin, bins - 1))
        high_bin = max(0, min(high_bin, bins))
        
        # Distribute volume evenly across overlapping bins
        if high_bin > low_bin:
            volume_per_bin = bar_volume / (high_bin - low_bin)
            for b in range(low_bin, high_bin):
                if 0 <= b < bins:
                    volume_distribution[b] += volume_per_bin
    
    # Find POC (Point of Control)
    poc_idx = np.argmax(volume_distribution)
    poc = bin_centers[poc_idx]
    
    # Calculate 70% value area
    total_volume = np.sum(volume_distribution)
    target_volume = total_volume * 0.70
    
    # Find value area by expanding from POC
    sorted_indices = np.argsort(volume_distribution)[::-1]
    cumulative_volume = 0
    value_area_indices = []
    
    for idx in sorted_indices:
        cumulative_volume += volume_distribution[idx]
        value_area_indices.append(idx)
        if cumulative_volume >= target_volume:
            break
    
    value_area_prices = [bin_centers[i] for i in value_area_indices]
    value_area_high = np.max(value_area_prices)
    value_area_low = np.min(value_area_prices)
    
    # Build profile
    profile = [(float(bin_centers[i]), float(volume_distribution[i])) 
               for i in range(bins) if volume_distribution[i] > 0]
    
    return {
        'poc': float(poc),
        'value_area_high': float(value_area_high),
        'value_area_low': float(value_area_low),
        'profile': profile,
        'volume_distribution': {float(bin_centers[i]): float(volume_distribution[i]) 
                               for i in range(bins) if volume_distribution[i] > 0}
    }

def analyze_level_volume_profile_interaction(level, volume_profile, current_price):
    """
    Analyze how a level interacts with the volume profile
    
    Goes through every level on the volume profile and sees the interaction
    Uses value areas to help gauge direction bias
    
    Returns:
    --------
    dict: {
        'volume_at_level': float,  # Volume at this price level
        'distance_to_poc': float,   # Distance to POC (%)
        'in_value_area': bool,      # Is level in value area?
        'above_value_area': bool,    # Is level above value area?
        'below_value_area': bool,    # Is level below value area?
        'direction_bias': str,      # 'bullish', 'bearish', 'neutral'
        'volume_strength': float,    # 0-1, how much volume supports this level
        'interaction_score': float   # Overall interaction score
    }
    """
    if not volume_profile or not level:
        return None
    
    level_price = level.get('price', current_price)
    poc = volume_profile.get('poc', current_price)
    va_high = volume_profile.get('value_area_high', current_price)
    va_low = volume_profile.get('value_area_low', current_price)
    volume_distribution = volume_profile.get('volume_distribution', {})
    profile = volume_profile.get('profile', [])
    
    # Find volume at this specific level
    volume_at_level = 0
    if volume_distribution:
        # Find closest price bin
        closest_price = min(volume_distribution.keys(), 
                          key=lambda p: abs(p - level_price))
        if abs(closest_price - level_price) / level_price < 0.01:  # Within 1%
            volume_at_level = volume_distribution.get(closest_price, 0)
    
    # If not found in distribution, check profile
    if volume_at_level == 0 and profile:
        closest_profile = min(profile, key=lambda x: abs(x[0] - level_price))
        if abs(closest_profile[0] - level_price) / level_price < 0.01:
            volume_at_level = closest_profile[1]
    
    # Normalize volume (0-1 scale)
    if profile:
        max_volume = max([v for _, v in profile]) if profile else 1
        volume_strength = min(1.0, volume_at_level / max_volume) if max_volume > 0 else 0
    else:
        volume_strength = 0
    
    # Distance to POC
    distance_to_poc = abs(level_price - poc) / current_price if current_price > 0 else 0
    
    # Position relative to value area
    in_value_area = va_low <= level_price <= va_high
    above_value_area = level_price > va_high
    below_value_area = level_price < va_low
    
    # Direction bias based on value area position
    if in_value_area:
        # In value area - check position relative to POC
        if level_price > poc:
            direction_bias = 'bullish'  # Above POC in VA = slight bullish
        elif level_price < poc:
            direction_bias = 'bearish'  # Below POC in VA = slight bearish
        else:
            direction_bias = 'neutral'
    elif above_value_area:
        direction_bias = 'bullish'  # Above VA = bullish (price likely to return to VA)
    elif below_value_area:
        direction_bias = 'bearish'  # Below VA = bearish (price likely to return to VA)
    else:
        direction_bias = 'neutral'
    
    # Interaction score (0-1): combines volume strength and value area position
    if in_value_area:
        interaction_score = 0.7 + (volume_strength * 0.3)  # High score if in VA
    elif above_value_area or below_value_area:
        interaction_score = 0.3 + (volume_strength * 0.2)  # Lower score outside VA
    else:
        interaction_score = volume_strength * 0.5  # Very low if no clear position
    
    return {
        'volume_at_level': float(volume_at_level),
        'distance_to_poc': float(distance_to_poc * 100),  # As percentage
        'in_value_area': in_value_area,
        'above_value_area': above_value_area,
        'below_value_area': below_value_area,
        'direction_bias': direction_bias,
        'volume_strength': float(volume_strength),
        'interaction_score': float(interaction_score),
        'poc': float(poc),
        'va_high': float(va_high),
        'va_low': float(va_low)
    }

def predict_level_as_hod_lod(level, current_price, all_levels, volume_profile, 
                              microstructure_state, sigma_price, timeframe):
    """
    Predict if a level will become the actual HOD or LOD
    
    Uses:
    - Level strength and confluence
    - Volume profile (value areas)
    - Distance from current price
    - Microstructure state
    - Other competing levels
    
    Returns:
    --------
    dict: {
        'will_be_hod': bool,
        'will_be_lod': bool,
        'hod_probability': float,  # 0-1
        'lod_probability': float,  # 0-1
        'confidence': float,
        'reasoning': str
    }
    """
    if level is None or 'price' not in level:
        return None
    
    level_price = level.get('price', current_price)
    is_above = level_price > current_price
    is_below = level_price < current_price
    
    if not (is_above or is_below):
        return None  # Level is at current price
    
    # Level strength factors
    level_strength = level.get('strength', level.get('levelStrength', 0.5))
    confluence_count = level.get('confluence_count', 1)
    
    # Distance from current (closer = more likely to be HOD/LOD)
    distance_pct = abs(level_price - current_price) / current_price
    distance_factor = 1.0 / (1.0 + distance_pct * 10)  # Closer = higher
    
    # Volume profile context
    volume_weight = 0.5
    if volume_profile:
        poc = volume_profile.get('poc', current_price)
        va_high = volume_profile.get('value_area_high', current_price)
        va_low = volume_profile.get('value_area_low', current_price)
        
        # If level is near POC or in value area, more likely to be HOD/LOD
        dist_to_poc = abs(level_price - poc) / current_price
        if dist_to_poc < 0.01:
            volume_weight = 1.0
        elif va_low <= level_price <= va_high:
            volume_weight = 0.8
        else:
            volume_weight = 0.3
    
    # Check for competing levels (other levels closer to theoretical bounds)
    competing_factor = 1.0
    if is_above:
        # Check if there are stronger levels above this one
        stronger_above = [l for l in all_levels 
                         if l.get('price', 0) > level_price and 
                         (l.get('strength', 0) > level_strength or 
                          l.get('confluence_count', 0) > confluence_count)]
        if stronger_above:
            competing_factor = 0.6  # Less likely if stronger levels exist above
    else:
        # Check if there are stronger levels below this one
        stronger_below = [l for l in all_levels 
                         if l.get('price', 0) < level_price and 
                         (l.get('strength', 0) > level_strength or 
                          l.get('confluence_count', 0) > confluence_count)]
        if stronger_below:
            competing_factor = 0.6
    
    # Microstructure context
    micro_state = microstructure_state.get('state', 'Unknown') if microstructure_state else 'Unknown'
    is_trending = micro_state in ['Fock', 'Trending', 'Expansion']
    
    # Calculate probabilities
    base_prob = (level_strength * 0.4 + 
                min(confluence_count / 5, 1.0) * 0.3 + 
                distance_factor * 0.2 + 
                volume_weight * 0.1) * competing_factor
    
    if is_above:
        hod_prob = base_prob
        lod_prob = 0.1  # Very unlikely to be LOD if above current
        will_be_hod = hod_prob > 0.5
        will_be_lod = False
        reasoning = f"Resistance level at ${level_price:.2f}. Strength: {level_strength:.2f}, Confluence: {confluence_count}"
    else:
        lod_prob = base_prob
        hod_prob = 0.1  # Very unlikely to be HOD if below current
        will_be_hod = False
        will_be_lod = lod_prob > 0.5
        reasoning = f"Support level at ${level_price:.2f}. Strength: {level_strength:.2f}, Confluence: {confluence_count}"
    
    confidence = min(0.9, base_prob + (confluence_count / 10))
    
    return {
        'will_be_hod': will_be_hod,
        'will_be_lod': will_be_lod,
        'hod_probability': float(np.clip(hod_prob, 0.0, 1.0)),
        'lod_probability': float(np.clip(lod_prob, 0.0, 1.0)),
        'confidence': float(confidence),
        'reasoning': reasoning,
        'level_price': float(level_price),
        'distance_pct': float(distance_pct * 100)
    }

def predict_level_reaction(level, current_price, start_of_move_price, sigma_price, 
                          volume_profile, microstructure_state, hurst_data, garch_regime, 
                          hmm_regime, timeframe):
    """
    Predict how price will react when reaching a level
    
    Enhanced with:
    - Hurst exponent (trending vs mean-reverting)
    - GARCH volatility regime (volatility context)
    - HMM regime (market state)
    - Microstructure state (Fock/Thermal/Coherent)
    - Start of move (how far we've come)
    - Volume profile (value areas)
    - Level strength
    
    Returns:
    --------
    dict: {
        'reaction_type': str,  # 'bounce', 'break', 'pause', 'reject'
        'probability': float,  # 0-1
        'expected_move_after': float,  # % move after reaction
        'confidence': float,
        'factors': dict  # Breakdown of contributing factors
    }
    """
    if level is None or 'price' not in level:
        return None
    
    level_price = level.get('price', current_price)
    distance_to_level = abs(level_price - current_price) / current_price
    
    # Distance from start of move
    move_from_start = abs(current_price - start_of_move_price) / start_of_move_price
    move_to_level = abs(level_price - start_of_move_price) / start_of_move_price
    
    # Level strength
    level_strength = level.get('strength', level.get('levelStrength', 0.5))
    confluence_count = level.get('confluence_count', 1)
    
    # ===== HURST EXPONENT ANALYSIS =====
    hurst = hurst_data.get('hurst', 0.5) if hurst_data else 0.5
    hurst_regime = hurst_data.get('regime', 'Random Walk') if hurst_data else 'Random Walk'
    is_mean_reverting = hurst < 0.4  # Mean-reverting: levels more likely to hold
    is_trending_hurst = hurst > 0.6  # Trending: levels more likely to break
    is_random_walk = 0.4 <= hurst <= 0.6
    
    # Hurst impact on reaction
    # Mean-reverting: price tends to return to levels (bounce more likely)
    # Trending: price tends to continue through levels (break more likely)
    hurst_bounce_factor = 1.4 if is_mean_reverting else (0.7 if is_trending_hurst else 1.0)
    hurst_break_factor = 0.7 if is_mean_reverting else (1.3 if is_trending_hurst else 1.0)
    
    # ===== GARCH VOLATILITY REGIME =====
    garch_regime_name = garch_regime.get('regime', 'Normal Vol') if garch_regime else 'Normal Vol'
    vol_ratio = garch_regime.get('vol_ratio', 1.0) if garch_regime else 1.0
    is_high_vol = vol_ratio > 1.3  # Elevated volatility
    is_extreme_vol = vol_ratio > 1.5  # Extreme volatility spike
    
    # High vol = more likely to break levels, less likely to hold
    vol_break_factor = 1.2 if is_high_vol else (1.5 if is_extreme_vol else 1.0)
    vol_bounce_factor = 0.8 if is_high_vol else (0.6 if is_extreme_vol else 1.0)
    
    # ===== HMM REGIME =====
    hmm_state = hmm_regime.get('state', 'Unknown') if hmm_regime else 'Unknown'
    is_bullish_regime = hmm_state in ['Bull', 'Strong Bull']
    is_bearish_regime = hmm_state in ['Bear', 'Strong Bear']
    
    # Regime impact: bullish = more likely to break resistance, bearish = more likely to break support
    if current_price < level_price:  # Approaching resistance
        regime_break_factor = 1.2 if is_bullish_regime else (0.8 if is_bearish_regime else 1.0)
        regime_bounce_factor = 0.8 if is_bullish_regime else (1.2 if is_bearish_regime else 1.0)
    else:  # Approaching support
        regime_break_factor = 1.2 if is_bearish_regime else (0.8 if is_bullish_regime else 1.0)
        regime_bounce_factor = 0.8 if is_bearish_regime else (1.2 if is_bullish_regime else 1.0)
    
    # ===== MICROSTRUCTURE STATE =====
    micro_state = microstructure_state.get('state', 'Unknown') if microstructure_state else 'Unknown'
    is_fock = micro_state == 'Fock'  # Jump-dominated, fat tails - more likely to overshoot/break
    is_coherent = micro_state == 'Coherent'  # Directional - more likely to continue trend
    is_thermal = micro_state == 'Thermal'  # Diffusive - more likely to respect levels
    
    # Microstructure impact
    if is_fock:
        # Fock: High jump probability, levels more likely to break
        micro_break_factor = 1.3
        micro_bounce_factor = 0.7
    elif is_coherent:
        # Coherent: Directional, trend continuation
        micro_break_factor = 1.1
        micro_bounce_factor = 0.9
    elif is_thermal:
        # Thermal: Diffusive, levels more likely to hold
        micro_break_factor = 0.8
        micro_bounce_factor = 1.2
    else:
        micro_break_factor = 1.0
        micro_bounce_factor = 1.0
    
    # Volatility context (sigma)
    sigma_pct = sigma_price / current_price
    
    # ===== COMBINED REACTION PREDICTION =====
    reaction_type = 'pause'  # Default
    probability = 0.5
    
    # Calculate expected move based on volatility (sigma_price), not arbitrary percentages
    # Strong reactions: 1.5-2.5σ moves, weak reactions: 0.5-1.0σ moves
    # Convert sigma_price to percentage for expected_move_after
    base_move_pct = sigma_price / current_price if current_price > 0 else 0.01  # At least 1%
    
    # Calculate combined factors
    combined_bounce_factor = hurst_bounce_factor * vol_bounce_factor * regime_bounce_factor * micro_bounce_factor
    combined_break_factor = hurst_break_factor * vol_break_factor * regime_break_factor * micro_break_factor
    
    # Strong level = likely bounce/reject (adjusted by factors)
    if level_strength > 0.7:
        if current_price < level_price:
            # Approaching resistance from below
            bounce_prob = 0.7 + (level_strength * 0.2)
            break_prob = 0.3 - (level_strength * 0.2)
            
            # Apply factors
            bounce_prob *= combined_bounce_factor
            break_prob *= combined_break_factor
            
            if bounce_prob > break_prob:
                reaction_type = 'bounce'
                probability = min(0.95, bounce_prob)
                # Bounce: expect 1.0-1.5σ pullback
                expected_move_after = -base_move_pct * (1.0 + vol_ratio * 0.5)  # Negative = pullback
            else:
                reaction_type = 'break'
                probability = min(0.95, break_prob)
                # Break: expect 1.5-2.5σ continuation
                expected_move_after = base_move_pct * (1.5 + vol_ratio * 1.0)  # Positive = continue up
        else:
            # Approaching support from above
            bounce_prob = 0.6 + (level_strength * 0.2)
            break_prob = 0.4 - (level_strength * 0.2)
            
            bounce_prob *= combined_bounce_factor
            break_prob *= combined_break_factor
            
            if bounce_prob > break_prob:
                reaction_type = 'bounce'
                probability = min(0.95, bounce_prob)
                # Bounce: expect 1.0-1.5σ bounce up
                expected_move_after = base_move_pct * (1.0 + vol_ratio * 0.5)  # Positive = bounce up
            else:
                reaction_type = 'break'
                probability = min(0.95, break_prob)
                # Break: expect 1.5-2.5σ continuation down
                expected_move_after = -base_move_pct * (1.5 + vol_ratio * 1.0)  # Negative = continue down
    
    # Weak level = likely break (adjusted by factors)
    elif level_strength < 0.4:
        reaction_type = 'break'
        base_prob = 0.6 + ((1 - level_strength) * 0.3)
        probability = min(0.95, base_prob * combined_break_factor)
        
        if current_price < level_price:
            # Weak resistance: expect 1.0-2.0σ break up
            expected_move_after = base_move_pct * (1.0 + vol_ratio * 1.0)
        else:
            # Weak support: expect 1.0-2.0σ break down
            expected_move_after = -base_move_pct * (1.0 + vol_ratio * 1.0)
    
    # Medium strength = pause/consolidation
    else:
        reaction_type = 'pause'
        probability = 0.5
        # Pause: small move, 0.3-0.7σ
        expected_move_after = base_move_pct * (0.3 + vol_ratio * 0.4)  # Small move, scaled by vol
    
    # Ensure expected_move_after is meaningful (at least 0.5% or 0.5σ)
    min_move_pct = max(0.005, base_move_pct * 0.5)
    if abs(expected_move_after) < min_move_pct:
        expected_move_after = min_move_pct if expected_move_after > 0 else -min_move_pct
    
    # Adjust based on move distance (fatigue)
    if move_to_level > 0.03:  # Moved more than 3%
        probability *= 0.8  # Less likely to react strongly
        # Reduce expected move slightly if already moved far
        expected_move_after *= 0.85
        if reaction_type == 'break':
            probability *= 1.1  # But more likely to break if already weak
    
    # Confidence based on confluence and factor agreement
    base_confidence = 0.5 + (confluence_count / 5) * 0.3
    factor_agreement = 1.0  # How much factors agree
    
    # If factors strongly favor one direction, increase confidence
    if abs(combined_bounce_factor - combined_break_factor) > 0.3:
        factor_agreement = 1.2
    
    confidence = min(0.95, base_confidence * factor_agreement)
    
    # Build factors breakdown
    factors = {
        'hurst': {
            'value': float(hurst),
            'regime': hurst_regime,
            'bounce_factor': float(hurst_bounce_factor),
            'break_factor': float(hurst_break_factor)
        },
        'garch_regime': {
            'regime': garch_regime_name,
            'vol_ratio': float(vol_ratio),
            'break_factor': float(vol_break_factor),
            'bounce_factor': float(vol_bounce_factor)
        },
        'hmm_regime': {
            'state': hmm_state,
            'break_factor': float(regime_break_factor),
            'bounce_factor': float(regime_bounce_factor)
        },
        'microstructure': {
            'state': micro_state,
            'break_factor': float(micro_break_factor),
            'bounce_factor': float(micro_bounce_factor)
        },
        'combined': {
            'bounce_factor': float(combined_bounce_factor),
            'break_factor': float(combined_break_factor)
        }
    }
    
    return {
        'reaction_type': reaction_type,
        'probability': float(np.clip(probability, 0.0, 1.0)),
        'expected_move_after': float(expected_move_after),
        'confidence': float(confidence),
        'level_price': float(level_price),
        'distance_pct': float(distance_to_level * 100),
        'volume_context': {
            'in_value_area': is_in_value_area,
            'volume_at_level': volume_at_level
        },
        'factors': factors
    }

# ============================================================================
# MULTI-TIMEFRAME LEVEL-BASED LSTM FORECASTING
# Predicts: Which levels will be touched, in what order, and when
# ============================================================================

def get_multi_timeframe_levels(ticker: str, base_timeframe: str = '5m', hist_base=None):
    """
    Fetch and detect levels across multiple timeframes
    Creates a hierarchical level structure
    """
    # Define timeframe hierarchy (each level is ~5x the previous)
    tf_hierarchy = {
        '1m': ['1m', '5m', '15m', '1h', '4h', '1d'],
        '5m': ['5m', '15m', '1h', '4h', '1d'],
        '15m': ['15m', '1h', '4h', '1d'],
        '1h': ['1h', '4h', '1d'],
        '4h': ['4h', '1d'],
        '1d': ['1d']
    }
    
    timeframes = tf_hierarchy.get(base_timeframe, ['5m', '1h', '1d'])
    
    stock = yf.Ticker(ticker)
    all_mtf_levels = {}
    
    for tf in timeframes:
        # Fetch appropriate period for each timeframe
        period_map = {
            '1m': '5d', '5m': '5d', '15m': '1mo', 
            '1h': '3mo', '4h': '6mo', '1d': '2y'
        }
        
        try:
            # For futures, handle interval conversion
            is_futures = '=' in ticker
            if is_futures and tf == '1h':
                interval = '60m'
            elif is_futures and tf == '4h':
                interval = '240m'
            else:
                interval = tf
            
            hist = stock.history(period=period_map.get(tf, '1y'), interval=interval)
            if len(hist) < 20:
                continue
            
            # Use existing level detection functions
            highs = hist['High'].values
            lows = hist['Low'].values
            closes = hist['Close'].values
            
            # Detect levels using existing methods
            levels = []
            
            # HDBSCAN levels
            hdbscan_levels = calculate_hdbscan_levels(highs, lows, closes, timeframe=tf)
            levels.extend([{**l, 'method': 'hdbscan'} for l in hdbscan_levels])
            
            # OPTICS levels
            optics_levels = enhanced_optics_levels(highs, lows, closes, timeframe=tf)
            levels.extend([{**l, 'method': 'optics'} for l in optics_levels])
            
            # Add timeframe metadata
            for level in levels:
                level['timeframe'] = tf
                level['tf_weight'] = get_timeframe_weight(base_timeframe, tf)
            
            all_mtf_levels[tf] = levels
            
        except Exception as e:
            print(f"⚠ Failed to fetch {tf}: {e}")
            continue
    
    return all_mtf_levels

def get_timeframe_weight(base_tf: str, target_tf: str) -> float:
    """Calculate weight based on timeframe relationship"""
    tf_order = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    try:
        base_idx = tf_order.index(base_tf)
        target_idx = tf_order.index(target_tf)
        
        if target_idx == base_idx:
            return 1.0
        elif target_idx > base_idx:
            diff = target_idx - base_idx
            return 1.0 + (diff * 0.2)
        else:
            diff = base_idx - target_idx
            return max(0.3, 1.0 - (diff * 0.15))
    except:
        return 1.0

def engineer_mtf_level_features(
    current_price: float,
    current_bar_data: Dict,
    mtf_levels: Dict[str, List[Dict]],
    historical_level_touches: List[Dict],
    lookback_bars: pd.DataFrame
) -> np.ndarray:
    """
    Engineer features that capture multi-timeframe level structure
    """
    features = []
    
    # === PRICE POSITION FEATURES ===
    recent_high = lookback_bars['High'].max()
    recent_low = lookback_bars['Low'].min()
    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
    
    features.extend([
        float(price_position),
        float((current_price - lookback_bars['Close'].mean()) / (lookback_bars['Close'].std() + 1e-8)),
    ])
    
    # === MOMENTUM FEATURES ===
    returns = lookback_bars['Close'].pct_change().fillna(0)
    features.extend([
        float(returns.iloc[-1]),
        float(returns.tail(5).mean()),
        float(returns.tail(10).mean()),
        float(returns.std()),
    ])
    
    # === MULTI-TIMEFRAME LEVEL FEATURES ===
    for tf, levels in sorted(mtf_levels.items()):
        if not levels:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        
        levels_above = [l for l in levels if l.get('price', 0) > current_price]
        levels_below = [l for l in levels if l.get('price', float('inf')) < current_price]
        
        # Nearest resistance
        if levels_above:
            nearest_res = min(levels_above, key=lambda x: x.get('price', float('inf')))
            res_dist = (nearest_res.get('price', current_price) - current_price) / current_price
            res_strength = nearest_res.get('strength', nearest_res.get('levelStrength', 0.5))
            res_weight = nearest_res.get('tf_weight', 1.0)
        else:
            res_dist = 0.05
            res_strength = 0.0
            res_weight = 0.0
        
        # Nearest support
        if levels_below:
            nearest_sup = max(levels_below, key=lambda x: x.get('price', 0))
            sup_dist = (current_price - nearest_sup.get('price', current_price)) / current_price
            sup_strength = nearest_sup.get('strength', nearest_sup.get('levelStrength', 0.5))
            sup_weight = nearest_sup.get('tf_weight', 1.0)
        else:
            sup_dist = 0.05
            sup_strength = 0.0
            sup_weight = 0.0
        
        features.extend([
            float(res_dist),
            float(res_strength * res_weight),
            float(sup_dist),
            float(sup_strength * sup_weight),
            float(len(levels_above) / 10),
            float(len(levels_below) / 10)
        ])
    
    # === LEVEL CONFLUENCE FEATURES ===
    confluence_zones = find_confluence_zones(mtf_levels, current_price)
    features.extend([
        float(len(confluence_zones.get('resistance', []))),
        float(len(confluence_zones.get('support', []))),
    ])
    
    # === HISTORICAL TOUCH PATTERN FEATURES ===
    if historical_level_touches:
        recent_touches = historical_level_touches[-10:]
        if len(recent_touches) >= 2:
            touch_intervals = [recent_touches[i].get('bar', 0) - recent_touches[i-1].get('bar', 0) 
                             for i in range(1, len(recent_touches))]
            avg_interval = np.mean(touch_intervals) if touch_intervals else 10.0
        else:
            avg_interval = 10.0
        
        features.append(float(1.0 / (avg_interval + 1)))
        
        last_touch = recent_touches[-1] if recent_touches else None
        if last_touch:
            features.extend([
                float((last_touch.get('level_price', current_price) / current_price) - 1),
                float(last_touch.get('level_strength', 0.5)),
            ])
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    return np.array(features, dtype=np.float32)

def find_confluence_zones(mtf_levels: Dict, current_price: float, tolerance: float = 0.01) -> Dict:
    """Find price zones where multiple timeframes have levels"""
    all_levels = []
    for tf, levels in mtf_levels.items():
        for level in levels:
            price = level.get('price', 0)
            if price > 0:
                all_levels.append({**level, 'timeframe': tf})
    
    if not all_levels:
        return {'resistance': [], 'support': []}
    
    resistance = []
    support = []
    
    for level in all_levels:
        price = level.get('price', 0)
        if price <= 0:
            continue
            
        if abs(price - current_price) / current_price < tolerance:
            continue
        
        if price > current_price:
            found = False
            for zone in resistance:
                if abs(price - zone['price']) / price < tolerance:
                    zone['count'] += 1
                    zone['total_strength'] += level.get('strength', level.get('levelStrength', 0.5))
                    found = True
                    break
            if not found:
                resistance.append({
                    'price': price,
                    'count': 1,
                    'total_strength': level.get('strength', level.get('levelStrength', 0.5))
                })
        else:
            found = False
            for zone in support:
                if abs(price - zone['price']) / price < tolerance:
                    zone['count'] += 1
                    zone['total_strength'] += level.get('strength', level.get('levelStrength', 0.5))
                    found = True
                    break
            if not found:
                support.append({
                    'price': price,
                    'count': 1,
                    'total_strength': level.get('strength', level.get('levelStrength', 0.5))
                })
    
    resistance = [z for z in resistance if z['count'] >= 2]
    support = [z for z in support if z['count'] >= 2]
    
    return {'resistance': resistance, 'support': support}

if torch is not None:
    class LevelSequenceLSTM(nn.Module):
        """
        LSTM that predicts next N levels that will be touched, in order
        """
        def __init__(
            self,
            n_features: int,
            hidden_dim: int = 256,
            n_layers: int = 3,
            max_levels_predict: int = 5,
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.max_levels = max_levels_predict
            
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=True
            )
            
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=dropout
            )
            
            self.level_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, max_levels_predict * 3)
            )
            
            self.direction_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            
        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            lstm_out_t = lstm_out.transpose(0, 1)
            attn_out, attn_weights = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
            attn_out = attn_out.transpose(0, 1)
            context = attn_out[:, -1, :]
            
            level_raw = self.level_predictor(context)
            level_predictions = level_raw.view(-1, self.max_levels, 3)
            
            direction_logits = self.direction_head(context)
            direction_probs = torch.softmax(direction_logits, dim=1)
            
            return level_predictions, direction_probs, attn_weights
else:
    LevelSequenceLSTM = None

def find_touched_levels_in_window(
    future_data: pd.DataFrame,
    mtf_levels: Dict[str, List[Dict]],
    start_price: float,
    max_levels: int = 5,
    touch_tolerance: float = 0.002
) -> List[Dict]:
    """Find which levels were touched in the future window, in order"""
    all_levels = []
    for tf, levels in mtf_levels.items():
        for level in levels:
            price = level.get('price', 0)
            if price > 0:
                all_levels.append({**level, 'timeframe': tf})
    
    touches = []
    
    for idx, (bar_idx, bar) in enumerate(future_data.iterrows()):
        high = bar['High']
        low = bar['Low']
        
        for level in all_levels:
            price = level.get('price', 0)
            if price <= 0:
                continue
            
            if low <= price <= high:
                distance = abs(price - start_price) / start_price
                
                if distance < touch_tolerance:
                    continue
                
                already_touched = any(
                    abs(t.get('price', 0) - price) / price < touch_tolerance 
                    for t in touches
                )
                
                if not already_touched:
                    touches.append({
                        'price': price,
                        'bar_index': idx,
                        'level_strength': level.get('strength', level.get('levelStrength', 0.5)),
                        'timeframe': level.get('timeframe', 'unknown'),
                        'tf_weight': level.get('tf_weight', 1.0)
                    })
    
    touches = sorted(touches, key=lambda x: x['bar_index'])
    return touches[:max_levels]

def find_closest_mtf_level(price: float, mtf_levels: Dict[str, List[Dict]], tolerance: float = 0.01) -> Optional[Dict]:
    """Find closest actual level from multi-timeframe levels"""
    all_levels = []
    for tf, levels in mtf_levels.items():
        for level in levels:
            level_price = level.get('price', 0)
            if level_price > 0:
                all_levels.append({**level, 'timeframe': tf})
    
    if not all_levels:
        return None
    
    closest = min(all_levels, key=lambda l: abs(l.get('price', 0) - price))
    
    if abs(closest.get('price', 0) - price) / price < tolerance:
        return closest
    return None

def get_timeframe_minutes(tf: str) -> int:
    """Convert timeframe to minutes"""
    map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
    return map.get(tf, 5)

def generate_level_sequence_training_data(
    ticker: str,
    base_timeframe: str = '5m',
    lookback_window: int = 50,
    max_levels_per_sample: int = 5,
    min_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data that learns level touch sequences
    
    Returns:
    - X: [n_samples, lookback_window, n_features]
    - y_levels: [n_samples, max_levels, 3] (price_offset, time_bars, confidence)
    - y_direction: [n_samples, 2] (up_prob, down_prob)
    """
    print(f"Generating training data for {ticker} {base_timeframe}...")
    
    # Fetch MTF levels
    mtf_levels = get_multi_timeframe_levels(ticker, base_timeframe)
    
    # Fetch base timeframe data
    stock = yf.Ticker(ticker)
    
    # Handle futures interval conversion
    is_futures = '=' in ticker
    if is_futures and base_timeframe == '1h':
        interval = '60m'
    elif is_futures and base_timeframe == '4h':
        interval = '240m'
    else:
        interval = base_timeframe
    
    period_map = {'1m': '5d', '5m': '1mo', '15m': '3mo', '1h': '6mo', '4h': '1y', '1d': '2y'}
    
    try:
        hist = stock.history(period=period_map.get(base_timeframe, '1y'), interval=interval)
    except:
        # Try fallback
        try:
            if is_futures and base_timeframe == '1h':
                hist = stock.history(period='5d', interval='60m')
            else:
                hist = stock.history(period='1mo', interval=base_timeframe)
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker} {base_timeframe}: {e}")
    
    if len(hist) < lookback_window + 50:
        raise ValueError(f"Insufficient data: {len(hist)} bars (need {lookback_window + 50})")
    
    X_samples = []
    y_level_samples = []
    y_direction_samples = []
    
    # Track historical level touches (reset per sample to avoid data leakage)
    historical_touches = []
    
    # Statistics for validation
    samples_skipped_no_touches = 0
    total_touches_found = 0
    
    # Generate samples
    for i in range(lookback_window, len(hist) - 30):  # Leave 30 bars for future
        # Get lookback window
        window_data = hist.iloc[i-lookback_window:i]
        current_price = float(hist.iloc[i]['Close'])
        current_bar_idx = i
        
        # Get next 30 bars to see what levels were actually touched
        future_data = hist.iloc[i:i+30]
        
        # Find which levels were touched in the future
        touched_levels = find_touched_levels_in_window(
            future_data, mtf_levels, current_price, max_levels=max_levels_per_sample
        )
        
        if not touched_levels:
            samples_skipped_no_touches += 1
            continue  # No levels touched, skip
        
        total_touches_found += len(touched_levels)
        
        # Engineer features for this sample
        features_sequence = []
        for j in range(len(window_data)):
            bar = window_data.iloc[j]
            sub_window = window_data.iloc[:j+1]
            
            try:
                feats = engineer_mtf_level_features(
                    current_price=float(bar['Close']),
                    current_bar_data=bar.to_dict(),
                    mtf_levels=mtf_levels,
                    historical_level_touches=historical_touches,
                    lookback_bars=sub_window
                )
                features_sequence.append(feats)
            except Exception as e:
                # If feature engineering fails for a bar, use previous features or zeros
                if features_sequence:
                    features_sequence.append(features_sequence[-1])
                else:
                    # Use zeros as fallback - get feature size from a test call
                    try:
                        test_feats = engineer_mtf_level_features(
                            current_price=current_price,
                            current_bar_data=bar.to_dict(),
                            mtf_levels=mtf_levels,
                            historical_level_touches=[],
                            lookback_bars=sub_window
                        )
                        n_feat = len(test_feats)
                    except:
                        n_feat = 100  # Default fallback
                    features_sequence.append(np.zeros(n_feat, dtype=np.float32))
        
        if len(features_sequence) != lookback_window:
            continue  # Skip if sequence incomplete
        
        X_samples.append(np.array(features_sequence))
        
        # Create target: next N levels touched
        level_targets = np.zeros((max_levels_per_sample, 3))
        for idx, touch in enumerate(touched_levels[:max_levels_per_sample]):
            # Price offset from current (normalized)
            touch_price = touch.get('price', current_price)
            price_offset = (touch_price - current_price) / current_price if current_price > 0 else 0
            # Time in bars (from historical touches - bar_index is relative to future_data start)
            bar_index = touch.get('bar_index', 0)
            time_bars = max(1, bar_index)  # bar_index is already relative to start of future_data
            # Confidence (based on level strength)
            confidence = touch.get('level_strength', touch.get('strength', 0.5))
            
            level_targets[idx] = [price_offset, min(time_bars / 30.0, 1.0), confidence]  # Normalize time
        
        y_level_samples.append(level_targets)
        
        # Direction target
        final_price = float(future_data.iloc[-1]['Close'])
        if final_price > current_price:
            y_direction_samples.append([1.0, 0.0])  # Up
        else:
            y_direction_samples.append([0.0, 1.0])  # Down
        
        # Update historical touches (use only touches from current sample window to avoid data leakage)
        # Reset historical_touches to only include touches from current sample's lookback window
        historical_touches = []
        # Look back through recent bars in lookback window for touches
        for j in range(max(0, i - 20), i):  # Look at last 20 bars for context
            past_bar = hist.iloc[j]
            past_high = past_bar['High']
            past_low = past_bar['Low']
            past_close = past_bar['Close']
            
            # Check if any MTF level was touched in this past bar
            for tf, levels in mtf_levels.items():
                for level in levels:
                    level_price = level.get('price', 0)
                    if level_price > 0 and past_low <= level_price <= past_high:
                        historical_touches.append({
                            'level_price': level_price,
                            'level_strength': level.get('strength', level.get('levelStrength', 0.5)),
                            'bar': j,
                            'timeframe': tf
                        })
        
        # Limit to most recent 50 touches
        historical_touches = historical_touches[-50:]
    
    if len(X_samples) < min_samples:
        raise ValueError(f"Only generated {len(X_samples)} samples (need {min_samples})")
    
    # Calculate statistics
    avg_touches_per_sample = total_touches_found / len(X_samples) if len(X_samples) > 0 else 0
    skip_rate = samples_skipped_no_touches / (len(X_samples) + samples_skipped_no_touches) if (len(X_samples) + samples_skipped_no_touches) > 0 else 0
    
    print(f"✓ Generated {len(X_samples)} training samples")
    print(f"  Feature dimension: {X_samples[0].shape[1]}")
    print(f"  Sequence length: {lookback_window}")
    print(f"  Samples skipped (no touches): {samples_skipped_no_touches} ({skip_rate*100:.1f}%)")
    print(f"  Average touches per sample: {avg_touches_per_sample:.2f}")
    
    # Validate data quality
    X_array = np.array(X_samples, dtype=np.float32)
    y_levels_array = np.array(y_level_samples, dtype=np.float32)
    y_direction_array = np.array(y_direction_samples, dtype=np.float32)
    
    # Check for NaN/Inf
    if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
        print("⚠ WARNING: NaN or Inf values found in features!")
    if np.any(np.isnan(y_levels_array)) or np.any(np.isinf(y_levels_array)):
        print("⚠ WARNING: NaN or Inf values found in level targets!")
    if np.any(np.isnan(y_direction_array)) or np.any(np.isinf(y_direction_array)):
        print("⚠ WARNING: NaN or Inf values found in direction targets!")
    
    # Check price_offset range (should be reasonable, not >50%)
    price_offsets = y_levels_array[:, :, 0].flatten()
    price_offsets = price_offsets[price_offsets != 0]  # Remove padding zeros
    if len(price_offsets) > 0:
        max_offset_pct = np.max(np.abs(price_offsets)) * 100
        if max_offset_pct > 50:
            print(f"⚠ WARNING: Large price offsets detected (max: {max_offset_pct:.1f}%)")
    
    return (
        np.array(X_samples, dtype=np.float32),
        np.array(y_level_samples, dtype=np.float32),
        np.array(y_direction_samples, dtype=np.float32)
    )

def predict_level_sequence(
    model: 'LevelSequenceLSTM',
    recent_features: np.ndarray,
    current_price: float,
    mtf_levels: Dict[str, List[Dict]],
    base_timeframe: str = '5m'
) -> Dict:
    """
    Predict next levels and their touch sequence using LevelSequenceLSTM
    
    Returns rich prediction with:
    - level_path: ordered list of levels to be touched
    - time_estimates: when each level will be touched
    - confidence_scores: confidence in each prediction
    - direction_bias: overall directional bias
    """
    if model is None or torch is None:
        return None
    
    model.eval()
    
    with torch.no_grad():
        X = torch.FloatTensor(recent_features).unsqueeze(0)  # [1, seq_len, features]
        
        level_preds, direction_probs, attn_weights = model(X)
        
        # Extract predictions
        level_preds = level_preds.squeeze(0).cpu().numpy()  # [max_levels, 3]
        direction_probs = direction_probs.squeeze(0).cpu().numpy()  # [2]
        
        # Build level path
        level_path = []
        
        for i, (price_offset, time_norm, confidence) in enumerate(level_preds):
            if confidence < 0.3:  # Skip low confidence
                continue
            
            predicted_price = current_price * (1 + price_offset)
            predicted_time_bars = int(time_norm * 30)  # Denormalize
            
            # Find closest actual level from MTF
            closest_level = find_closest_mtf_level(predicted_price, mtf_levels)
            
            level_path.append({
                'sequence_num': i + 1,
                'predicted_price': float(predicted_price),
                'actual_level_price': closest_level.get('price') if closest_level else None,
                'actual_level_strength': closest_level.get('strength', closest_level.get('levelStrength', 0)) if closest_level else 0,
                'actual_level_timeframe': closest_level.get('timeframe', 'unknown') if closest_level else None,
                'time_bars': predicted_time_bars,
                'time_minutes': predicted_time_bars * get_timeframe_minutes(base_timeframe),
                'confidence': float(confidence),
                'price_offset_pct': float(price_offset * 100)
            })
        
        # Sort by time (order of touch)
        level_path = sorted(level_path, key=lambda x: x['time_bars'])
        
        return {
            'level_path': level_path,
            'direction_bias': {
                'up_probability': float(direction_probs[0]),
                'down_probability': float(direction_probs[1]),
                'bias': 'bullish' if direction_probs[0] > direction_probs[1] else 'bearish'
            },
            'total_levels_predicted': len(level_path),
            'average_confidence': float(np.mean([l['confidence'] for l in level_path])) if level_path else 0.0
        }

# ============================================================================
# LEVEL-BASED LSTM FORECAST: "Where is price going today?"
# ============================================================================

def engineer_level_features_for_lstm(
    current_price,
    theoretical_hod_premarket,
    theoretical_lod_premarket,
    theoretical_hod_intraday,
    theoretical_lod_intraday,
    hdbscan_levels,
    optics_levels,
    interaction_levels,
    ml_confluence_levels,
    multiscale_levels,
    neural_network_levels=None,
    volume_profile=None,
    all_levels=None
):
    """
    Convert levels into LSTM-ready features
    
    Key idea: Encode price's RELATIONSHIP to each level type
    Not the absolute prices, but relative positions
    
    Returns:
    --------
    np.array of shape (n_features,) ready for LSTM input
    """
    features = []
    
    # ===== 1. THEORETICAL BOUNDS (baseline expectation) =====
    
    # Pre-market bounds (set at open, static)
    if theoretical_hod_premarket > theoretical_lod_premarket:
        dist_to_pm_hod = (theoretical_hod_premarket - current_price) / current_price
        dist_to_pm_lod = (current_price - theoretical_lod_premarket) / current_price
        pm_range_position = (current_price - theoretical_lod_premarket) / \
                            (theoretical_hod_premarket - theoretical_lod_premarket)
    else:
        dist_to_pm_hod = 0.05
        dist_to_pm_lod = 0.05
        pm_range_position = 0.5
    
    features.extend([
        float(dist_to_pm_hod),      # How far to pre-market HOD (%)
        float(dist_to_pm_lod),      # How far to pre-market LOD (%)
        float(pm_range_position)    # Position in pre-market range (0-1)
    ])
    
    # Intraday bounds (updated as session progresses)
    if theoretical_hod_intraday > theoretical_lod_intraday:
        dist_to_id_hod = (theoretical_hod_intraday - current_price) / current_price
        dist_to_id_lod = (current_price - theoretical_lod_intraday) / current_price
        id_range_position = (current_price - theoretical_lod_intraday) / \
                            (theoretical_hod_intraday - theoretical_lod_intraday)
    else:
        dist_to_id_hod = 0.05
        dist_to_id_lod = 0.05
        id_range_position = 0.5
    
    # Bound evolution (how have bounds changed intraday vs pre-market?)
    hod_expansion = (theoretical_hod_intraday - theoretical_hod_premarket) / current_price
    lod_expansion = (theoretical_lod_premarket - theoretical_lod_intraday) / current_price
    
    features.extend([
        float(dist_to_id_hod),
        float(dist_to_id_lod),
        float(id_range_position),
        float(hod_expansion),       # Did HOD expand? (positive = yes)
        float(lod_expansion)        # Did LOD expand? (positive = yes)
    ])
    
    # ===== 2. HDBSCAN STRUCTURAL LEVELS =====
    
    # Find nearest HDBSCAN levels above/below
    hdbscan_above = [l for l in hdbscan_levels if l.get('price', 0) > current_price]
    hdbscan_below = [l for l in hdbscan_levels if l.get('price', 0) < current_price]
    
    if hdbscan_above:
        nearest_above = min(hdbscan_above, key=lambda x: x.get('price', float('inf')))
        hdbscan_resistance_dist = (nearest_above.get('price', current_price) - current_price) / current_price
        hdbscan_resistance_strength = nearest_above.get('strength', nearest_above.get('levelStrength', 0.5))
    else:
        hdbscan_resistance_dist = 0.05  # Default: 5% above
        hdbscan_resistance_strength = 0.0
    
    if hdbscan_below:
        nearest_below = max(hdbscan_below, key=lambda x: x.get('price', 0))
        hdbscan_support_dist = (current_price - nearest_below.get('price', current_price)) / current_price
        hdbscan_support_strength = nearest_below.get('strength', nearest_below.get('levelStrength', 0.5))
    else:
        hdbscan_support_dist = 0.05  # Default: 5% below
        hdbscan_support_strength = 0.0
    
    # Count levels in vicinity (within ±2%)
    hdbscan_density_above = sum(1 for l in hdbscan_above 
                                if (l.get('price', current_price) - current_price) / current_price < 0.02)
    hdbscan_density_below = sum(1 for l in hdbscan_below 
                                if (current_price - l.get('price', current_price)) / current_price < 0.02)
    
    features.extend([
        float(hdbscan_resistance_dist),
        float(hdbscan_resistance_strength),
        float(hdbscan_support_dist),
        float(hdbscan_support_strength),
        float(hdbscan_density_above / 10),   # Normalize by dividing by max expected
        float(hdbscan_density_below / 10)
    ])
    
    # ===== 3. OPTICS MULTI-DENSITY LEVELS =====
    
    # Same pattern as HDBSCAN
    optics_above = [l for l in optics_levels if l.get('price', 0) > current_price]
    optics_below = [l for l in optics_levels if l.get('price', 0) < current_price]
    
    if optics_above:
        nearest = min(optics_above, key=lambda x: x.get('price', float('inf')))
        optics_resistance_dist = (nearest.get('price', current_price) - current_price) / current_price
        optics_resistance_density = nearest.get('density_score', nearest.get('strength', 0.5))
    else:
        optics_resistance_dist = 0.05
        optics_resistance_density = 0.0
    
    if optics_below:
        nearest = max(optics_below, key=lambda x: x.get('price', 0))
        optics_support_dist = (current_price - nearest.get('price', current_price)) / current_price
        optics_support_density = nearest.get('density_score', nearest.get('strength', 0.5))
    else:
        optics_support_dist = 0.05
        optics_support_density = 0.0
    
    features.extend([
        float(optics_resistance_dist),
        float(optics_resistance_density),
        float(optics_support_dist),
        float(optics_support_density)
    ])
    
    # ===== 4. INTERACTION LEVELS (local density, short memory) =====
    
    interaction_above = [l for l in interaction_levels if l.get('price', 0) > current_price]
    interaction_below = [l for l in interaction_levels if l.get('price', 0) < current_price]
    
    # Interaction levels are short-memory, so weight by recency
    if interaction_above:
        nearest = min(interaction_above, key=lambda x: x.get('price', float('inf')))
        interaction_resistance_dist = (nearest.get('price', current_price) - current_price) / current_price
        interaction_resistance_density = nearest.get('density_prominence', nearest.get('strength', 0.5))
    else:
        interaction_resistance_dist = 0.02  # Smaller default (local)
        interaction_resistance_density = 0.0
    
    if interaction_below:
        nearest = max(interaction_below, key=lambda x: x.get('price', 0))
        interaction_support_dist = (current_price - nearest.get('price', current_price)) / current_price
        interaction_support_density = nearest.get('density_prominence', nearest.get('strength', 0.5))
    else:
        interaction_support_dist = 0.02
        interaction_support_density = 0.0
    
    features.extend([
        float(interaction_resistance_dist),
        float(interaction_resistance_density),
        float(interaction_support_dist),
        float(interaction_support_density)
    ])
    
    # ===== 5. ML-CONFLUENCE LEVELS (algorithm agreement) =====
    
    ml_above = [l for l in ml_confluence_levels if l.get('price', 0) > current_price]
    ml_below = [l for l in ml_confluence_levels if l.get('price', 0) < current_price]
    
    if ml_above:
        nearest = min(ml_above, key=lambda x: x.get('price', float('inf')))
        ml_resistance_dist = (nearest.get('price', current_price) - current_price) / current_price
        ml_resistance_confluence = min(nearest.get('confluence_count', 1) / 5, 1.0)  # Normalize
    else:
        ml_resistance_dist = 0.05
        ml_resistance_confluence = 0.0
    
    if ml_below:
        nearest = max(ml_below, key=lambda x: x.get('price', 0))
        ml_support_dist = (current_price - nearest.get('price', current_price)) / current_price
        ml_support_confluence = min(nearest.get('confluence_count', 1) / 5, 1.0)
    else:
        ml_support_dist = 0.05
        ml_support_confluence = 0.0
    
    features.extend([
        float(ml_resistance_dist),
        float(ml_resistance_confluence),
        float(ml_support_dist),
        float(ml_support_confluence)
    ])
    
    # ===== 6. MULTI-SCALE HDBSCAN LEVELS =====
    
    # Separate by scale
    micro_levels = [l for l in multiscale_levels if l.get('scale') == 'micro']
    meso_levels = [l for l in multiscale_levels if l.get('scale') == 'meso']
    macro_levels = [l for l in multiscale_levels if l.get('scale') == 'macro']
    
    def nearest_level_distance(levels, above=True):
        if above:
            filtered = [l for l in levels if l.get('price', 0) > current_price]
            if filtered:
                nearest = min(filtered, key=lambda x: x.get('price', float('inf')))
                return (nearest.get('price', current_price) - current_price) / current_price
        else:
            filtered = [l for l in levels if l.get('price', 0) < current_price]
            if filtered:
                nearest = max(filtered, key=lambda x: x.get('price', 0))
                return (current_price - nearest.get('price', current_price)) / current_price
        return 0.05  # Default
    
    features.extend([
        float(nearest_level_distance(micro_levels, above=True)),   # Micro resistance
        float(nearest_level_distance(micro_levels, above=False)),  # Micro support
        float(nearest_level_distance(meso_levels, above=True)),    # Meso resistance
        float(nearest_level_distance(meso_levels, above=False)),   # Meso support
        float(nearest_level_distance(macro_levels, above=True)),   # Macro resistance
        float(nearest_level_distance(macro_levels, above=False))   # Macro support
    ])
    
    # ===== 7. NEURAL NETWORK LEVELS (pattern + volume profile based) =====
    
    if neural_network_levels is None:
        neural_network_levels = []
    
    nn_above = [l for l in neural_network_levels if l.get('price', 0) > current_price]
    nn_below = [l for l in neural_network_levels if l.get('price', 0) < current_price]
    
    if nn_above:
        nearest = min(nn_above, key=lambda x: x.get('price', float('inf')))
        nn_resistance_dist = (nearest.get('price', current_price) - current_price) / current_price
        nn_resistance_strength = nearest.get('strength', nearest.get('levelStrength', 0.5))
    else:
        nn_resistance_dist = 0.05
        nn_resistance_strength = 0.0
    
    if nn_below:
        nearest = max(nn_below, key=lambda x: x.get('price', 0))
        nn_support_dist = (current_price - nearest.get('price', current_price)) / current_price
        nn_support_strength = nearest.get('strength', nearest.get('levelStrength', 0.5))
    else:
        nn_support_dist = 0.05
        nn_support_strength = 0.0
    
    # Count neural network levels in vicinity
    nn_density_above = sum(1 for l in nn_above 
                           if (l.get('price', current_price) - current_price) / current_price < 0.02)
    nn_density_below = sum(1 for l in nn_below 
                           if (current_price - l.get('price', current_price)) / current_price < 0.02)
    
    features.extend([
        float(nn_resistance_dist),
        float(nn_resistance_strength),
        float(nn_support_dist),
        float(nn_support_strength),
        float(nn_density_above),
        float(nn_density_below)
    ])
    
    # ===== 8. CROSS-LEVEL AGREEMENT (meta-feature) =====
    
    # Do all level types agree on nearest resistance/support?
    all_resistance_dists = [
        hdbscan_resistance_dist,
        optics_resistance_dist,
        interaction_resistance_dist,
        ml_resistance_dist,
        nn_resistance_dist
    ]
    all_support_dists = [
        hdbscan_support_dist,
        optics_support_dist,
        interaction_support_dist,
        ml_support_dist,
        nn_support_dist
    ]
    
    # Agreement = low variance in distances (all see same level)
    resistance_agreement = 1.0 / (1.0 + np.std(all_resistance_dists) if len(all_resistance_dists) > 0 else 1.0)
    support_agreement = 1.0 / (1.0 + np.std(all_support_dists) if len(all_support_dists) > 0 else 1.0)
    
    features.extend([
        float(resistance_agreement),
        float(support_agreement)
    ])
    
    # ===== 9. VOLUME PROFILE FEATURES =====
    if volume_profile:
        poc = volume_profile.get('poc', current_price)
        va_high = volume_profile.get('value_area_high', current_price)
        va_low = volume_profile.get('value_area_low', current_price)
        
        # Distance to POC and value area
        dist_to_poc = (poc - current_price) / current_price
        dist_to_va_high = (va_high - current_price) / current_price
        dist_to_va_low = (current_price - va_low) / current_price
        
        # Position in value area (0 = at VA low, 1 = at VA high, 0.5 = at POC)
        if va_high > va_low:
            va_position = (current_price - va_low) / (va_high - va_low)
        else:
            va_position = 0.5
        
        # Volume profile direction bias
        if current_price < va_low:
            va_bias = -1.0  # Below VA = bearish
        elif current_price > va_high:
            va_bias = 1.0   # Above VA = bullish
        else:
            va_bias = (current_price - poc) / (va_high - va_low) if va_high > va_low else 0.0
        
        features.extend([
            float(dist_to_poc),
            float(dist_to_va_high),
            float(dist_to_va_low),
            float(va_position),
            float(va_bias)
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.5, 0.0])  # Defaults
    
    # ===== 9. LEVEL DENSITY AND PATTERNS =====
    if all_levels:
        # Count levels in different zones
        levels_above = [l for l in all_levels if l.get('price', 0) > current_price]
        levels_below = [l for l in all_levels if l.get('price', 0) < current_price]
        
        # Density in near zones (within 1%, 2%, 5%)
        density_1pct_above = sum(1 for l in levels_above 
                                if (l.get('price', current_price) - current_price) / current_price < 0.01)
        density_2pct_above = sum(1 for l in levels_above 
                                if (l.get('price', current_price) - current_price) / current_price < 0.02)
        density_5pct_above = sum(1 for l in levels_above 
                                if (l.get('price', current_price) - current_price) / current_price < 0.05)
        
        density_1pct_below = sum(1 for l in levels_below 
                                if (current_price - l.get('price', current_price)) / current_price < 0.01)
        density_2pct_below = sum(1 for l in levels_below 
                                if (current_price - l.get('price', current_price)) / current_price < 0.02)
        density_5pct_below = sum(1 for l in levels_below 
                                if (current_price - l.get('price', current_price)) / current_price < 0.05)
        
        # Average strength of nearby levels
        nearby_above = [l for l in levels_above 
                       if (l.get('price', current_price) - current_price) / current_price < 0.02]
        nearby_below = [l for l in levels_below 
                       if (current_price - l.get('price', current_price)) / current_price < 0.02]
        
        avg_strength_above = np.mean([l.get('strength', l.get('levelStrength', 0.5)) 
                                      for l in nearby_above]) if nearby_above else 0.0
        avg_strength_below = np.mean([l.get('strength', l.get('levelStrength', 0.5)) 
                                     for l in nearby_below]) if nearby_below else 0.0
        
        features.extend([
            float(density_1pct_above / 5),   # Normalize
            float(density_2pct_above / 10),
            float(density_5pct_above / 20),
            float(density_1pct_below / 5),
            float(density_2pct_below / 10),
            float(density_5pct_below / 20),
            float(avg_strength_above),
            float(avg_strength_below)
        ])
    else:
        features.extend([0.0] * 8)  # Defaults
    
    return np.array(features, dtype=np.float32)

# LSTM Model (only if torch is available)
if TORCH_AVAILABLE:
    class LevelBasedLSTM(nn.Module):
        """
        LSTM that learns: "Given current level configuration, where does price go?"
        
        Input: Sequence of level features (timesteps × features)
        Output: Next price target (regression)
        """
        def __init__(
            self,
            n_features=43,        # From engineer_level_features_for_lstm
            hidden_dim=128,
            n_layers=2,
            dropout=0.2
        ):
            super().__init__()
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
            
            # Attention mechanism (which timestep matters most?)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Output heads
            self.price_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)  # Predict next price target
            )
            
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Confidence (0-1)
            )
            
            self.time_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus()  # Time to target (positive)
            )
            
            # HOD/LOD level prediction heads (optional - for new models)
            # These predict probability distribution over candidate levels
            self.max_levels = 50  # Max candidate levels
            self.hod_level_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.max_levels)  # Probability over candidate levels
            )
            self.lod_level_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.max_levels)  # Probability over candidate levels
            )
        
        def forward(self, x, return_attention=False):
            """
            x: [batch, seq_len, n_features]
            """
            # LSTM
            lstm_out, (h_n, c_n) = self.lstm(x)
            # lstm_out: [batch, seq_len, hidden_dim]
            
            # Attention weights
            attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_dim]
            
            # Predictions
            price_pred = self.price_head(context)       # [batch, 1]
            confidence = self.confidence_head(context)  # [batch, 1]
            time_pred = self.time_head(context)         # [batch, 1]
            
            # HOD/LOD level predictions (if model has these heads)
            if hasattr(self, 'hod_level_head'):
                hod_level_logits = self.hod_level_head(context)  # [batch, max_levels]
                lod_level_logits = self.lod_level_head(context)  # [batch, max_levels]
                hod_level_probs = torch.softmax(hod_level_logits, dim=1)  # [batch, max_levels]
                lod_level_probs = torch.softmax(lod_level_logits, dim=1)  # [batch, max_levels]
                
                if return_attention:
                    return price_pred, confidence, time_pred, attn_weights, hod_level_probs, lod_level_probs
                return price_pred, confidence, time_pred, hod_level_probs, lod_level_probs
            
            if return_attention:
                return price_pred, confidence, time_pred, attn_weights
            return price_pred, confidence, time_pred
else:
    LevelBasedLSTM = None

def generate_lstm_training_data(
    hist,
    all_levels_dict,
    theoretical_hod_lod_series,  # Pre-computed theoretical bounds for each bar
    lookback_window=20,
    forecast_horizon=30  # How many bars ahead to predict
):
    """
    Generate training samples from historical data
    
    For each bar t:
    - Input: Features from bars [t-lookback_window : t]
    - Target: Actual price at bar [t + forecast_horizon]
    
    Returns:
    --------
    X: [n_samples, lookback_window, n_features]
    y_price: [n_samples] - actual prices reached
    y_time: [n_samples] - actual bars to reach target
    """
    X_sequences = []
    y_prices = []
    y_times = []
    
    for i in range(lookback_window, len(hist) - forecast_horizon):
        # Build feature sequence
        sequence = []
        
        for t in range(i - lookback_window, i):
            current_bar = hist.iloc[t]
            current_price = current_bar['Close']
            
            # Get theoretical bounds for this bar
            theo_bounds = theoretical_hod_lod_series[t] if t < len(theoretical_hod_lod_series) else theoretical_hod_lod_series[-1]
            
            # Get levels valid at this time (levels detected up to bar t)
            # This simulates real-time: you only know levels detected so far
            levels_at_t = {
                'hdbscan': [l for l in all_levels_dict.get('hdbscan', []) if l.get('detected_at_bar', 0) <= t],
                'optics': [l for l in all_levels_dict.get('optics', []) if l.get('detected_at_bar', 0) <= t],
                'interaction': [l for l in all_levels_dict.get('interaction', []) if l.get('detected_at_bar', 0) <= t],
                'ml_confluence': [l for l in all_levels_dict.get('ml_confluence', []) if l.get('detected_at_bar', 0) <= t],
                'multiscale': [l for l in all_levels_dict.get('multiscale', []) if l.get('detected_at_bar', 0) <= t]
            }
            
            # Engineer features for this timestep
            features = engineer_level_features_for_lstm(
                current_price=current_price,
                theoretical_hod_premarket=theo_bounds.get('pm_hod', current_price * 1.02),
                theoretical_lod_premarket=theo_bounds.get('pm_lod', current_price * 0.98),
                theoretical_hod_intraday=theo_bounds.get('id_hod', current_price * 1.02),
                theoretical_lod_intraday=theo_bounds.get('id_lod', current_price * 0.98),
                hdbscan_levels=levels_at_t['hdbscan'],
                optics_levels=levels_at_t['optics'],
                interaction_levels=levels_at_t['interaction'],
                ml_confluence_levels=levels_at_t['ml_confluence'],
                multiscale_levels=levels_at_t['multiscale']
            )
            
            sequence.append(features)
        
        # Target: price at t + forecast_horizon
        target_price = hist.iloc[i + forecast_horizon]['Close']
        
        # Normalize target as % move from current
        current_price_now = hist.iloc[i]['Close']
        target_pct_move = (target_price - current_price_now) / current_price_now
        
        # Time to reach target (simulate: find first bar that crosses target)
        time_to_target = forecast_horizon  # Default
        for j in range(i, i + forecast_horizon):
            if abs(hist.iloc[j]['Close'] - target_price) < target_price * 0.005:  # Within 0.5%
                time_to_target = j - i
                break
        
        X_sequences.append(np.array(sequence))
        y_prices.append(target_pct_move)
        y_times.append(time_to_target)
    
    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_prices, dtype=np.float32),
        np.array(y_times, dtype=np.float32)
    )

def calculate_time_to_target(target_price, current_price, sigma_price, volatility_factor=1.0):
    """
    Calculate time estimate to reach target based on:
    - Distance to target (price units)
    - Volatility (sigma) - how fast price moves
    - Volatility factor (adjusts speed estimate)
    
    Returns estimated bars to reach target
    """
    if sigma_price <= 0:
        return 20  # Default if no volatility data
    
    # Distance to target in price units
    distance = abs(target_price - current_price)
    
    # Distance in terms of standard deviations
    distance_sigma = distance / sigma_price
    
    # Typical move per bar (average of last N bars)
    # Assume price moves ~0.5-1.0 sigma per bar on average (varies by regime)
    typical_move_per_bar = sigma_price * 0.75 * volatility_factor
    
    # Estimate bars = distance / typical_move_per_bar
    if typical_move_per_bar > 0:
        estimated_bars = distance / typical_move_per_bar
    else:
        estimated_bars = distance_sigma * 2  # Fallback: ~2 bars per sigma
    
    # Clamp to reasonable range (5-100 bars)
    estimated_bars = max(5, min(100, int(estimated_bars)))
    
    return estimated_bars

def predict_price_target(
    model,
    recent_features,  # [lookback_window, n_features]
    current_price,
    sigma_price=0.0,  # Optional: volatility for time calculation
    volatility_factor=1.0  # Optional: adjust time estimate speed
):
    """
    Answer: "Where is price going today?"
    
    Returns:
    --------
    dict : {
        'target_price': predicted price level,
        'target_pct_move': % move from current,
        'confidence': model confidence (0-1),
        'expected_time_bars': bars to reach target (calculated from distance/volatility, not LSTM output),
        'attention_weights': which timesteps matter most
    }
    """
    if not TORCH_AVAILABLE or model is None:
        return None
    
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        X = torch.FloatTensor(recent_features).unsqueeze(0)  # [1, seq_len, features]
        
        # Forward pass with attention
        result = model(X, return_attention=True)
        
        # Handle both old and new model formats
        if len(result) == 4:
            # Old model format
            price_pred, confidence, time_pred_raw, attn_weights = result
            hod_probs = None
            lod_probs = None
        else:
            # New model format with HOD/LOD predictions
            price_pred, confidence, time_pred_raw, attn_weights, hod_probs, lod_probs = result
        
        # Convert % move to absolute price
        target_pct_move = float(price_pred.squeeze().item())
        target_price = current_price * (1 + target_pct_move)
        
        # Combine LSTM's time prediction with distance-based estimate
        # LSTM learned temporal patterns - use them, but weight with distance-based calculation
        model_confidence = float(confidence.squeeze().item())
        
        # Extract LSTM time prediction (convert to bars, clamp to reasonable range)
        if time_pred_raw is not None:
            lstm_time_raw = float(time_pred_raw.squeeze().item())
            # LSTM outputs normalized or raw time - try to interpret
            # If it's very small (< 1), treat as normalized (multiply by 30)
            # If it's larger, treat as bars directly
            if lstm_time_raw < 1.0:
                lstm_time_bars = int(lstm_time_raw * 30)  # Denormalize
            else:
                lstm_time_bars = int(lstm_time_raw)
            # Clamp to reasonable range (1-200 bars)
            lstm_time_bars = max(1, min(200, lstm_time_bars))
        else:
            lstm_time_bars = 20  # Default fallback
        
        # Calculate distance-based time estimate
        if sigma_price > 0:
            distance_time_bars = calculate_time_to_target(target_price, current_price, sigma_price, volatility_factor)
        else:
            # Fallback: use distance-based estimate if no volatility
            distance_pct = abs(target_pct_move) * 100
            # Rough estimate: ~1% move per bar (conservative)
            distance_time_bars = max(5, min(100, int(distance_pct)))
        
        # Weight LSTM prediction with distance-based estimate based on confidence
        # High confidence → trust LSTM more, Low confidence → trust distance more
        expected_time_bars = int(
            model_confidence * lstm_time_bars + (1 - model_confidence) * distance_time_bars
        )
        # Ensure reasonable bounds
        expected_time_bars = max(1, min(200, expected_time_bars))
        
        return {
            'target_price': float(target_price),
            'target_pct_move': float(target_pct_move * 100),  # %
            'confidence': float(confidence.squeeze().item()),
            'expected_time_bars': int(expected_time_bars),  # Calculated from distance/volatility
            'attention_weights': attn_weights.squeeze().cpu().numpy().tolist(),
            'hod_level_probs': hod_probs.squeeze().cpu().numpy().tolist() if hod_probs is not None else None,
            'lod_level_probs': lod_probs.squeeze().cpu().numpy().tolist() if lod_probs is not None else None
        }


def monte_carlo_lstm_forecast(
    model,
    recent_features,
    current_price,
    theoretical_hod,
    theoretical_lod,
    levels,
    volume_profile,
    sigma_price,
    hurst_data=None,
    garch_regime=None,
    hmm_regime=None,
    microstructure_state=None,
    n_simulations=30,  # Reduced from 100 for production (can be overridden for higher accuracy)
    forecast_bars=30
):
    """
    Monte Carlo simulation using LSTM to generate multiple price path scenarios
    
    Uses:
    - LSTM base prediction
    - Theoretical HOD/LOD as boundaries
    - Levels as reaction points
    - Volume profile for value areas
    - Random walk with LSTM-guided drift
    
    Returns:
    --------
    dict: {
        'scenarios': list of price paths,
        'probabilities': dict of outcome probabilities,
        'expected_path': average path,
        'confidence_intervals': {50%, 80%, 95%}
    }
    """
    if not TORCH_AVAILABLE or model is None:
        return None
    
    model.eval()
    scenarios = []
    
    # Get base LSTM prediction
    base_prediction = predict_price_target(model, recent_features, current_price)
    if not base_prediction:
        return None
    
    base_drift = base_prediction['target_pct_move'] / 100.0  # Convert to decimal
    base_confidence = base_prediction['confidence']
    
    # Volatility for random walk
    sigma = sigma_price / current_price
    
    with torch.no_grad():
        X_base = torch.FloatTensor(recent_features).unsqueeze(0)
        
        for sim in range(n_simulations):
            path = [current_price]
            
            # Track actual HOD/LOD for this scenario
            scenario_hod = current_price
            scenario_lod = current_price
            hod_level = None
            lod_level = None
            
            # Add noise to features for variation
            noise = np.random.normal(0, 0.01, recent_features.shape)
            X_noisy = torch.FloatTensor(recent_features + noise).unsqueeze(0)
            
            # Get prediction with noise
            price_pred, confidence, time_pred, _ = model(X_noisy, return_attention=True)
            drift = float(price_pred.squeeze().item())
            
            # Generate path with regime-aware adjustments
            # Get regime factors
            hurst = hurst_data.get('hurst', 0.5) if hurst_data else 0.5
            vol_ratio = garch_regime.get('vol_ratio', 1.0) if garch_regime else 1.0
            micro_state = microstructure_state.get('state', 'Unknown') if microstructure_state else 'Unknown'
            
            # Adjust volatility based on regime
            regime_sigma = sigma * vol_ratio
            
            # Hurst adjustment: mean-reverting = more constrained, trending = more momentum
            if hurst < 0.4:  # Mean-reverting
                momentum_factor = 0.8  # Less momentum
            elif hurst > 0.6:  # Trending
                momentum_factor = 1.2  # More momentum
            else:
                momentum_factor = 1.0
            
            # Microstructure adjustment
            if micro_state == 'Fock':  # Jump-dominated
                jump_probability = 0.1  # 10% chance of jump
            else:
                jump_probability = 0.02  # 2% chance normally
            
            # Generate path
            for bar in range(forecast_bars):
                # Random walk with LSTM drift, adjusted by regimes
                base_shock = np.random.normal(drift / forecast_bars * momentum_factor, 
                                             regime_sigma / np.sqrt(forecast_bars))
                
                # Add jump component if in Fock state
                if np.random.random() < jump_probability:
                    jump_size = np.random.normal(0, regime_sigma * 2)  # Large jump
                    base_shock += jump_size
                
                next_price = path[-1] * (1 + base_shock)
                
                # Apply boundaries (theoretical HOD/LOD)
                next_price = np.clip(next_price, theoretical_lod, theoretical_hod)
                
                # Track HOD/LOD
                if next_price > scenario_hod:
                    scenario_hod = next_price
                    # Check if this matches a level
                    for level in levels:
                        level_price = level.get('price', 0)
                        if abs(next_price - level_price) / next_price < 0.005:
                            hod_level = level_price
                            break
                
                if next_price < scenario_lod:
                    scenario_lod = next_price
                    # Check if this matches a level
                    for level in levels:
                        level_price = level.get('price', 0)
                        if abs(next_price - level_price) / next_price < 0.005:
                            lod_level = level_price
                            break
                
                # Check for level reactions
                for level in levels:
                    level_price = level.get('price', 0)
                    if abs(next_price - level_price) / level_price < 0.002:  # Within 0.2%
                        # Small reaction at level
                        reaction = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                        next_price = level_price * (1 + reaction * 0.001)
                        break
                
                path.append(float(next_price))
            
            scenarios.append({
                'path': path,
                'hod': float(scenario_hod),
                'lod': float(scenario_lod),
                'hod_level': float(hod_level) if hod_level else None,
                'lod_level': float(lod_level) if lod_level else None
            })
    
    # Extract paths and HOD/LOD data
    paths = [s['path'] for s in scenarios]
    hods = [s['hod'] for s in scenarios]
    lods = [s['lod'] for s in scenarios]
    hod_levels = [s['hod_level'] for s in scenarios if s['hod_level'] is not None]
    lod_levels = [s['lod_level'] for s in scenarios if s['lod_level'] is not None]
    
    # Calculate statistics
    final_prices = [p[-1] for p in paths]
    final_prices_sorted = sorted(final_prices)
    
    # Confidence intervals
    ci_50 = (final_prices_sorted[int(n_simulations * 0.25)], 
             final_prices_sorted[int(n_simulations * 0.75)])
    ci_80 = (final_prices_sorted[int(n_simulations * 0.10)], 
             final_prices_sorted[int(n_simulations * 0.90)])
    ci_95 = (final_prices_sorted[int(n_simulations * 0.025)], 
             final_prices_sorted[int(n_simulations * 0.975)])
    
    # Expected path (mean) - this is the theoretical path line
    expected_path = [np.mean([p[i] for p in paths]) for i in range(forecast_bars + 1)]
    
    # Expected HOD/LOD (mean of scenario HODs/LODs)
    expected_hod = float(np.mean(hods))
    expected_lod = float(np.mean(lods))
    
    # Find which levels most often become HOD/LOD
    from collections import Counter
    hod_level_counts = Counter(hod_levels)
    lod_level_counts = Counter(lod_levels)
    
    most_likely_hod_level = hod_level_counts.most_common(1)[0][0] if hod_level_counts else None
    most_likely_lod_level = lod_level_counts.most_common(1)[0][0] if lod_level_counts else None
    
    hod_level_prob = hod_level_counts[most_likely_hod_level] / n_simulations if most_likely_hod_level else 0
    lod_level_prob = lod_level_counts[most_likely_lod_level] / n_simulations if most_likely_lod_level else 0
    
    # Outcome probabilities
    up_prob = sum(1 for p in final_prices if p > current_price) / n_simulations
    down_prob = sum(1 for p in final_prices if p < current_price) / n_simulations
    neutral_prob = 1 - up_prob - down_prob
    
    # Value area probabilities
    va_prob = 0
    if volume_profile:
        va_low = volume_profile.get('value_area_low', current_price)
        va_high = volume_profile.get('value_area_high', current_price)
        va_prob = sum(1 for p in final_prices if va_low <= p <= va_high) / n_simulations
    
    return {
        'scenarios': paths[:10],  # Return first 10 paths for visualization
        'expected_path': expected_path,  # This is the theoretical path line
        'expected_hod': expected_hod,
        'expected_lod': expected_lod,
        'most_likely_hod_level': float(most_likely_hod_level) if most_likely_hod_level else None,
        'most_likely_lod_level': float(most_likely_lod_level) if most_likely_lod_level else None,
        'hod_level_probability': float(hod_level_prob),
        'lod_level_probability': float(lod_level_prob),
        'confidence_intervals': {
            '50': ci_50,
            '80': ci_80,
            '95': ci_95
        },
        'probabilities': {
            'up': float(up_prob),
            'down': float(down_prob),
            'neutral': float(neutral_prob),
            'in_value_area': float(va_prob)
        },
        'statistics': {
            'mean_final': float(np.mean(final_prices)),
            'median_final': float(np.median(final_prices)),
            'std_final': float(np.std(final_prices)),
            'min_final': float(np.min(final_prices)),
            'max_final': float(np.max(final_prices)),
            'mean_hod': expected_hod,
            'mean_lod': expected_lod
        }
    }

# NEW ENDPOINT: LEVEL-CONSTRAINED HOD/LOD PREDICTION
# ALIAS: Keep old endpoint name for backward compatibility



# ============================================================================
# FVECM-BASED HOD/LOD PREDICTION (replaces old state-conditioned and level-constrained)
# ============================================================================

@app.route('/api/level-constrained-hod-lod', methods=['GET'])
def get_level_constrained_hod_lod():
    """
    HOD/LOD prediction using the FVECM (Fractionally Vectorized Error Correction Model).
    Replaces the old multi-layered HOD/LOD pipeline with a cleaner 2-layer approach:
      Layer 1: VECM on (mid_log, range_log) -- cointegration-based structural envelope
      Layer 2: LightGBM residual corrector on normalized residuals
    """
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']

    ticker = request.args.get('ticker', 'SPY')
    timeframe = request.args.get('timeframe', '1d').strip().lower().replace('240m','4h').replace('4hour','4h').replace('4hours','4h').replace('60m','1h')

    try:
        stock = yf.Ticker(ticker)

        # Determine fetch parameters
        is_futures = '=' in ticker
        if is_futures:
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '60m', '4h': '60m', '1d': '1d'}
        else:
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
        interval = interval_map.get(timeframe, '1d')

        if is_futures and timeframe in ['1m', '5m', '15m', '1h', '4h']:
            period_map = {'1m': '5d', '5m': '5d', '15m': '7d', '1h': '7d', '4h': '10d', '1d': '2y'}
        else:
            period_map = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '3mo', '4h': '3mo', '1d': '2y'}
        period = period_map.get(timeframe, '1y')

        # Fetch data (with resampling for 4h)
        hist = None
        if timeframe == '4h':
            try:
                hist = fetch_historical_data_with_resampling(
                    ticker=ticker, timeframe='4h', period=period, is_futures=is_futures
                )
            except Exception:
                hist = None
        elif is_futures and timeframe in ['1m', '5m', '15m', '1h']:
            attempts = [period, '5d', '3d', '2d', '1d']
            for attempt_period in attempts:
                try:
                    hist = stock.history(period=attempt_period, interval=interval)
                    if hist is not None and len(hist) > 0:
                        break
                except Exception:
                    continue
        else:
            try:
                hist = stock.history(period=period, interval=interval)
            except Exception:
                hist = None

        if hist is None or len(hist) == 0:
            return jsonify({'success': False, 'error': f'No data available for {ticker} at {timeframe}'}), 400

        closes = hist['Close'].values
        highs = hist['High'].values if 'High' in hist.columns else closes
        lows = hist['Low'].values if 'Low' in hist.columns else closes
        volumes = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(closes))
        current_price = float(closes[-1])

        # Session volatility
        session_vol_pct = 1.5
        sigma_price = (session_vol_pct / 100) * current_price
        if all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close']):
            try:
                vol_result = compute_session_volatility(hist, window=60)
                session_vol_pct = vol_result['sigma_session_pct']
                sigma_price = vol_result['sigma_price']
            except Exception:
                pass

        # Microstructure state (kept as useful context)
        returns = np.log(closes[1:] / closes[:-1]) * 100
        microstructure_state = detect_market_microstructure_state(closes, volumes, returns, highs, lows)

        # GARCH regime
        garch_vol_regime = calculate_garch_volatility_regime(closes)

        # --- FVECM Pipeline ---
        try:
            ohlcv_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            ohlcv_df = ohlcv_df.rename(columns={
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            })

            pipeline = TheoreticalBoundsPipeline(PipelineConfig(
                train_window=min(250, len(ohlcv_df) - 1),
                min_train_window=min(180, max(60, len(ohlcv_df) // 2)),
            ))
            result_df = pipeline.run(ohlcv_df)

            if result_df is not None and len(result_df) > 0 and 'HOD_corrected' in result_df.columns:
                last = result_df.iloc[-1]
                predicted_hod = float(last.get('HOD_corrected', current_price + sigma_price))
                predicted_lod = float(last.get('LOD_corrected', current_price - sigma_price))

                # Fallback if values are NaN
                if np.isnan(predicted_hod) or not np.isfinite(predicted_hod):
                    predicted_hod = current_price + sigma_price
                if np.isnan(predicted_lod) or not np.isfinite(predicted_lod):
                    predicted_lod = current_price - sigma_price

                # Safety: ensure HOD > LOD
                if predicted_hod <= predicted_lod:
                    predicted_hod = current_price + sigma_price
                    predicted_lod = current_price - sigma_price
            else:
                # Pipeline didn't produce results (insufficient data) - use volatility fallback
                predicted_hod = current_price + sigma_price
                predicted_lod = current_price - sigma_price

        except Exception as e:
            print(f"FVECM pipeline failed: {e}, using volatility fallback")
            predicted_hod = current_price + sigma_price
            predicted_lod = current_price - sigma_price

        # Level detection (kept for refinement context)
        all_levels_combined = []
        try:
            hist_data_subset = hist.tail(min(len(hist), 100))
            hdbscan_levels = calculate_hdbscan_levels(highs, lows, closes, timeframe=timeframe)
            isolation_forest_levels = find_pivot_anomalies(highs, lows, closes)
            peak_valley_levels = find_peaks_valleys_scipy(highs, lows, closes)
            pivot_levels = calculate_pivot_points(hist_data_subset, timeframe)
            fib_levels = calculate_fibonacci_levels(highs, lows)
            gap_levels = find_gap_levels(hist_data_subset)

            all_ml_levels = hdbscan_levels + isolation_forest_levels + peak_valley_levels
            all_ml_levels = agglomerative_merge_levels(
                all_ml_levels, distance_threshold_pct=None, timeframe=timeframe
            )
            confluence_levels = get_ml_confluence_levels(all_ml_levels)
            all_levels_combined = confluence_levels + all_ml_levels + pivot_levels + gap_levels
            all_levels_combined = add_fibonacci_metadata_to_levels(
                all_levels_combined, fib_levels, sigma_price, threshold_sigma=1.0
            )

            all_levels_combined, hmm_regime, hurst_data, garch_regime, micro_state = enhance_levels_with_microstructure(
                all_levels_combined, closes, volumes, current_price, garch_vol_regime, microstructure_state, sigma_price=sigma_price
            )

            # Refine FVECM predictions with detected levels
            predicted_hod, predicted_lod, refinement_debug = refine_extrema_with_levels(
                spot=current_price,
                hod_th=predicted_hod,
                lod_th=predicted_lod,
                levels=all_levels_combined,
                state=micro_state,
                timeframe=timeframe,
            )
        except Exception as e:
            print(f"Level refinement failed: {e}")

        # Confidence
        resistance_levels = [l for l in all_levels_combined if l.get('price', 0) > current_price]
        support_levels = [l for l in all_levels_combined if l.get('price', 0) < current_price]
        hod_confidence = calculate_level_confidence(predicted_hod, resistance_levels, current_price, sigma_price)
        lod_confidence = calculate_level_confidence(predicted_lod, support_levels, current_price, sigma_price)

        std_dev_decimal = session_vol_pct / 100.0

        # Compute statistical sigma bands for frontend compatibility
        base_hod_1std = current_price + 1.0 * sigma_price
        base_lod_1std = current_price - 1.0 * sigma_price
        base_hod_2std = current_price + 2.0 * sigma_price
        base_lod_2std = current_price - 2.0 * sigma_price
        base_hod_3std = current_price + 3.0 * sigma_price
        base_lod_3std = current_price - 3.0 * sigma_price

        # Find selected resistance/support from refinement
        selected_resistance = refinement_debug.get('best_hod') if 'refinement_debug' in locals() and refinement_debug else None
        selected_support = refinement_debug.get('best_lod') if 'refinement_debug' in locals() and refinement_debug else None

        return jsonify({
            'success': True,
            'ticker': ticker,
            'timeframe': timeframe,
            'currentPrice': current_price,
            'method': 'FVECM + Level Refinement',
            'sigmaDailyPct': float(session_vol_pct),
            'sigmaPrice': float(sigma_price),
            'stdDev': std_dev_decimal,
            'sigma_price': float(sigma_price),

            # Frontend expects: hod['1std'], hod['2std'], hod['3std']
            'hod': {
                '1std': float(base_hod_1std),
                '2std': float(base_hod_2std),
                '3std': float(base_hod_3std)
            },
            'lod': {
                '1std': float(base_lod_1std),
                '2std': float(base_lod_2std),
                '3std': float(base_lod_3std)
            },

            # Level-constrained predicted HOD/LOD
            'predicted': {
                'hod': float(predicted_hod),
                'lod': float(predicted_lod),
                'hod_distance_pct': float((predicted_hod - current_price) / current_price * 100),
                'lod_distance_pct': float((current_price - predicted_lod) / current_price * 100),
                'hod_confidence': float(hod_confidence),
                'lod_confidence': float(lod_confidence)
            },

            # Also keep the 'predictions' key for any code using the new format
            'predictions': {
                'hod': float(predicted_hod),
                'lod': float(predicted_lod),
                'hod_pct': (float(predicted_hod) - current_price) / current_price * 100,
                'lod_pct': (current_price - float(predicted_lod)) / current_price * 100,
            },

            # Base statistical ranges (for comparison)
            'statistical': {
                'hod_1std': float(base_hod_1std),
                'lod_1std': float(base_lod_1std),
                'hod_2std': float(base_hod_2std),
                'lod_2std': float(base_lod_2std),
                'hod_3std': float(base_hod_3std),
                'lod_3std': float(base_lod_3std)
            },

            # Selected levels (if any)
            'selectedLevels': {
                'resistance': sanitize_for_json(selected_resistance) if selected_resistance else None,
                'support': sanitize_for_json(selected_support) if selected_support else None
            },

            # Nearby levels (for visualization)
            'nearbyLevels': {
                'resistance': sanitize_for_json(resistance_levels[:5]),
                'support': sanitize_for_json(support_levels[:5])
            },

            'confidence': {
                'hod': float(hod_confidence),
                'lod': float(lod_confidence),
            },
            'microstructure': sanitize_for_json(microstructure_state),
            'garchRegime': sanitize_for_json(garch_vol_regime),
            'levels': sanitize_for_json(all_levels_combined[:30]) if all_levels_combined else [],
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/level-constrained-hod-lod: {error_trace}")
        return jsonify({'success': False, 'error': str(e) or 'Unknown error'}), 400


@app.route('/api/stdv-hod-lod', methods=['GET'])
def get_stdv_hod_lod():
    """Alias for /api/level-constrained-hod-lod - backward compatibility"""
    return get_level_constrained_hod_lod()


@app.route('/api/state-conditioned-hod-lod', methods=['GET'])
def get_state_conditioned_hod_lod():
    """
    State-conditioned HOD/LOD prediction -- now powered by FVECM.
    Delegates to the same FVECM pipeline as level-constrained-hod-lod,
    keeping the endpoint for backward compatibility.
    """
    return get_level_constrained_hod_lod()



@app.route('/api/lstm-forecast', methods=['GET'])
def get_lstm_forecast():
    """
    "Where is price going today?" - LSTM-based answer using level features
    """
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    timeframe = request.args.get('timeframe', '5m').strip().lower().replace('240m','4h').replace('4hour','4h').replace('4hours','4h').replace('60m','1h')
    lookback_window = int(request.args.get('lookback', 20))
    
    # Note: PyTorch is optional - we'll use level-based heuristic if torch is not available
    
    try:
        print(f"Generating LSTM forecast for {ticker} at {timeframe}...")
        
        # Fetch data
        stock = yf.Ticker(ticker)
        
        # For futures, use alternative interval formats that yfinance accepts better
        is_futures = '=' in ticker
        if is_futures:
            # Use minute-based intervals for futures (yfinance prefers these)
            # Note: 4h is not supported by yfinance - will use resampling from 60m
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '60m', '4h': '60m', '1d': '1d'}
        else:
            # Note: 4h is not supported by yfinance - will use resampling from 1h
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
        
        interval = interval_map.get(timeframe, '1d')
        
        # Simple fix: Use shorter periods for futures on intraday timeframes
        if is_futures and timeframe in ['1m', '5m', '15m', '1h', '4h']:
            period_map = {'1m': '5d', '5m': '5d', '15m': '7d', '1h': '7d', '4h': '10d', '1d': '2y'}
        else:
            period_map = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '3mo', '4h': '3mo', '1d': '2y'}
        
        period = period_map.get(timeframe, '1y')
        
        # For futures, try multiple approaches - especially for 1h which is problematic
        hist = None
        if is_futures and timeframe == '1h':
            # For 1h futures, yfinance is very picky - try many combinations
            attempts = [
                ('60m', '5d'),   # Most reliable for futures
                ('60m', '3d'),
                ('60m', '2d'),
                ('60m', '1d'),
                ('1h', '5d'),    # Try standard format too
                ('1h', '3d'),
                ('1h', '2d'),
                ('1h', '1d'),
            ]
            
            for attempt_interval, attempt_period in attempts:
                try:
                    print(f"Trying {ticker} 1h: interval={attempt_interval}, period={attempt_period}")
                    hist = stock.history(period=attempt_period, interval=attempt_interval)
                    if hist is not None and len(hist) > 0:
                        print(f"✓ Successfully fetched {len(hist)} bars for {ticker} 1h with interval={attempt_interval}, period={attempt_period}")
                        break
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:150]}")
                    # Continue trying other combinations
                    continue
        elif is_futures and timeframe == '15m':
            # For 15m futures, try different periods
            attempts = [
                ('15m', period),
                ('15m', '5d'),
                ('15m', '3d'),
                ('15m', '2d'),
                ('15m', '1d'),
            ]
            
            for attempt_interval, attempt_period in attempts:
                try:
                    hist = stock.history(period=attempt_period, interval=attempt_interval)
                    if hist is not None and len(hist) > 0:
                        print(f"✓ Successfully fetched {len(hist)} bars for {ticker} 15m with interval={attempt_interval}, period={attempt_period}")
                        break
                except Exception as e:
                    error_msg = str(e)
                    if "pattern" not in error_msg.lower() and "expected" not in error_msg.lower():
                        print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:100]}")
                    continue
        elif is_futures and timeframe == '4h':
            # For 4h futures, yfinance doesn't support '4h' or '240m' - must fetch 1h/60m and resample
            print(f"Fetching 4h data for {ticker} (will resample from 1h/60m)...")
            try:
                hist = fetch_historical_data_with_resampling(
                    ticker=ticker,
                    timeframe='4h',
                    period=period,
                    is_futures=True
                )
            except Exception as e:
                print(f"⚠ Resampling fetch failed: {e}")
                hist = None
        elif is_futures and timeframe in ['1m', '5m']:
            # For other futures timeframes, try with fallback periods
            attempts = [
                (interval, period),
                (interval, '5d'),
                (interval, '2d'),
                (interval, '1d'),
            ]
            
            for attempt_interval, attempt_period in attempts:
                try:
                    hist = stock.history(period=attempt_period, interval=attempt_interval)
                    if hist is not None and len(hist) > 0:
                        print(f"✓ Successfully fetched {len(hist)} bars for {ticker} {timeframe}")
                        break
                except Exception as e:
                    error_msg = str(e)
                    if "pattern" not in error_msg.lower() and "expected" not in error_msg.lower():
                        print(f"⚠ Attempt failed: {error_msg[:100]}")
                    continue
        else:
            try:
                hist = stock.history(period=period, interval=interval)
            except Exception as e:
                error_msg = str(e)
                print(f"⚠ Error fetching data for {ticker} {timeframe}: {error_msg}")
                hist = None
        
        if len(hist) < lookback_window + 10:
            return jsonify({'success': False, 'error': f'Insufficient data. Need at least {lookback_window + 10} bars.'}), 400
        
        closes = hist['Close'].values
        highs = hist['High'].values if 'High' in hist.columns else closes
        lows = hist['Low'].values if 'Low' in hist.columns else closes
        volumes = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(closes))
        current_price = closes[-1]
        
        # Get session volatility for theoretical bounds
        vol_result = compute_session_volatility(hist, window=60)
        sigma_price = vol_result.get('sigma_price', 0.0)
        
        # Fallback: if sigma_price is 0 or invalid, estimate from price range
        if sigma_price <= 0 or not np.isfinite(sigma_price):
            price_range = np.max(highs) - np.min(lows)
            sigma_price = price_range * 0.02  # Rough estimate: 2% of range
            print(f"⚠ Using fallback sigma_price: {sigma_price:.4f}")
        
        # Get microstructure state, Hurst, and regimes for level reactions
        returns = np.log(closes[1:] / closes[:-1]) * 100
        microstructure_state = detect_market_microstructure_state(closes, volumes, returns, highs, lows)
        hurst_data = calculate_hurst_exponent(closes)
        garch_regime = calculate_garch_volatility_regime(closes)
        hmm_regime = detect_market_regime_hmm(closes)
        
        # 1. Calculate Volume Profile (for value areas and directional understanding)
        print("Calculating volume profile...")
        volume_profile = calculate_volume_profile(highs, lows, closes, volumes, bins=30)
        
        # 2. Detect all levels (your existing code)
        print("Detecting levels...")
        hdbscan_levels = calculate_hdbscan_levels(highs, lows, closes, timeframe=timeframe)
        optics_levels = enhanced_optics_levels(highs, lows, closes, timeframe=timeframe)
        
        # Interaction levels
        interaction_levels = calculate_local_interaction_levels(
            closes, current_price, sigma_price, lookback=200, bins=30, max_levels=5
        )
        
        # Multiscale levels
        multiscale_levels = multiscale_hdbscan_levels(highs, lows, closes, timeframe=timeframe)
        
        # Neural Network levels (with volume profile) - INCLUDED in theoretical HOD/LOD and LSTM forecast
        print("Detecting neural network levels...")
        neural_network_levels = detect_levels_with_neural_network(hist, lookback=100, threshold=0.5)
        print(f"✓ Neural Network levels detected: {len(neural_network_levels)} levels")
        
        # DeepSupp levels
        deepsupp_levels = []
        try:
            if TORCH_AVAILABLE:
                deepsupp_levels = detect_levels_with_deepsupp(hist, model_path='deepsupp_v4.pt', device='cpu')
                print(f"✓ DeepSupp levels detected: {len(deepsupp_levels)} levels")
        except Exception as e:
            print(f"DeepSupp level detection failed: {e}")
            deepsupp_levels = []

        # ML confluence (includes neural network + deepsupp levels)
        all_ml_levels = hdbscan_levels + optics_levels + interaction_levels + neural_network_levels + deepsupp_levels
        ml_confluence_levels = get_ml_confluence_levels(all_ml_levels)
        
        # 2a. Get Multi-Timeframe Levels (for enhanced LSTM prediction)
        print("Fetching multi-timeframe levels...")
        mtf_levels = {}
        level_sequence_prediction = None
        try:
            mtf_levels = get_multi_timeframe_levels(ticker, timeframe, hist)
            print(f"✓ Multi-timeframe levels fetched: {len(mtf_levels)} timeframes")
        except Exception as e:
            print(f"⚠ Multi-timeframe level fetch failed: {e}")
            mtf_levels = {}
        
        # 3. Predict level reactions AND which levels will become actual HOD/LOD
        # NOTE: neural_network_levels are included in all_levels for theoretical HOD/LOD refinement
        print("Predicting level reactions and HOD/LOD candidates...")
        all_levels = hdbscan_levels + optics_levels + interaction_levels + ml_confluence_levels + multiscale_levels + neural_network_levels + deepsupp_levels
        start_of_move_price = closes[0] if len(closes) > 0 else current_price  # Session start
        
        level_reactions = []
        hod_lod_predictions = []
        
        for level in all_levels[:20]:  # Analyze top 20 levels
            # Predict reaction (with Hurst, GARCH, HMM regimes)
            reaction = predict_level_reaction(
                level, current_price, start_of_move_price, sigma_price,
                volume_profile, microstructure_state, hurst_data, garch_regime,
                hmm_regime, timeframe
            )
            if reaction:
                reaction['level'] = sanitize_for_json(level)
                level_reactions.append(reaction)
            
            # Predict if this level will become actual HOD/LOD
            hod_lod_pred = predict_level_as_hod_lod(
                level, current_price, all_levels, volume_profile,
                microstructure_state, sigma_price, timeframe
            )
            if hod_lod_pred:
                hod_lod_pred['level'] = sanitize_for_json(level)
                hod_lod_predictions.append(hod_lod_pred)
        
        # Sort by distance from current price
        level_reactions.sort(key=lambda x: x['distance_pct'])
        
        # Sort HOD/LOD predictions by probability
        hod_lod_predictions.sort(key=lambda x: max(x.get('hod_probability', 0), x.get('lod_probability', 0)), reverse=True)
        
        # 4. Calculate theoretical bounds (pre-market and intraday) - these are the edges
        # Ensure sigma_price is valid and non-zero
        if sigma_price <= 0 or not np.isfinite(sigma_price):
            price_range = np.max(highs) - np.min(lows) if len(highs) > 0 else current_price * 0.1
            sigma_price = max(price_range * 0.02, current_price * 0.01)  # At least 1% of price
            print(f"⚠ Theoretical bounds: Using fallback sigma_price: {sigma_price:.4f}")
        
        theoretical_hod_pm = float(current_price + 2.0 * sigma_price)
        theoretical_lod_pm = float(current_price - 2.0 * sigma_price)
        
        # Intraday bounds (updated - slightly tighter)
        theoretical_hod_id = float(current_price + 1.5 * sigma_price)
        theoretical_lod_id = float(current_price - 1.5 * sigma_price)
        
        # Ensure bounds are valid (HOD > LOD)
        if theoretical_hod_pm <= theoretical_lod_pm:
            theoretical_hod_pm = current_price * 1.02
            theoretical_lod_pm = current_price * 0.98
        if theoretical_hod_id <= theoretical_lod_id:
            theoretical_hod_id = current_price * 1.015
            theoretical_lod_id = current_price * 0.985
        
        print(f"✓ Theoretical bounds calculated: HOD={theoretical_hod_pm:.2f}, LOD={theoretical_lod_pm:.2f}")
        
        # 3. Build feature sequence (last N bars)
        print(f"Building feature sequence (lookback={lookback_window})...")
        recent_features = []
        recent_features_mtf = []  # For multi-timeframe model
        
        # Track historical level touches for MTF features
        historical_touches = []
        
        for i in range(max(0, len(hist) - lookback_window), len(hist)):
            bar = hist.iloc[i]
            bar_price = bar['Close']
            lookback_window_data = hist.iloc[max(0, i-lookback_window):i+1]
            
            # Engineer features for this timestep (with volume profile and all levels)
            features = engineer_level_features_for_lstm(
                current_price=bar_price,
                theoretical_hod_premarket=theoretical_hod_pm,
                theoretical_lod_premarket=theoretical_lod_pm,
                theoretical_hod_intraday=theoretical_hod_id,
                theoretical_lod_intraday=theoretical_lod_id,
                hdbscan_levels=hdbscan_levels,
                optics_levels=optics_levels,
                interaction_levels=interaction_levels,
                ml_confluence_levels=ml_confluence_levels,
                multiscale_levels=multiscale_levels,
                neural_network_levels=neural_network_levels,
                volume_profile=volume_profile,
                all_levels=all_levels
            )
            recent_features.append(features)
            
            # Also build MTF features if available
            if mtf_levels:
                try:
                    features_mtf = engineer_mtf_level_features(
                        current_price=bar_price,
                        current_bar_data=bar.to_dict(),
                        mtf_levels=mtf_levels,
                        historical_level_touches=historical_touches,
                        lookback_bars=lookback_window_data
                    )
                    recent_features_mtf.append(features_mtf)
                except Exception as e:
                    print(f"⚠ MTF feature engineering failed for bar {i}: {e}")
                    # Fallback: use regular features
                    if recent_features_mtf:
                        recent_features_mtf.append(recent_features_mtf[-1])
                    else:
                        recent_features_mtf.append(features)
        
        recent_features = np.array(recent_features)
        if recent_features_mtf:
            recent_features_mtf = np.array(recent_features_mtf)
        else:
            recent_features_mtf = recent_features  # Fallback
        
        # 4. Load model and run Monte Carlo simulation (if model exists and torch is available)
        model_path = 'level_lstm_best.pth'
        model_path_mtf = 'level_sequence_lstm_best.pth'  # New MTF model path
        model = None
        model_mtf = None  # Multi-timeframe level sequence model
        prediction = None
        monte_carlo_result = None
        
        # Try to load multi-timeframe level sequence model first
        if TORCH_AVAILABLE and LevelSequenceLSTM is not None and mtf_levels:
            try:
                if os.path.exists(model_path_mtf):
                    print("Loading multi-timeframe level sequence model...")
                    n_features_mtf = recent_features_mtf.shape[1] if len(recent_features_mtf.shape) > 1 else recent_features.shape[1]
                    model_mtf = LevelSequenceLSTM(n_features=n_features_mtf, max_levels_predict=5)
                    model_mtf.load_state_dict(torch.load(model_path_mtf, map_location='cpu'))
                    print("✓ Multi-timeframe level sequence model loaded")
                    
                    # Make level sequence prediction
                    level_sequence_prediction = predict_level_sequence(
                        model=model_mtf,
                        recent_features=recent_features_mtf,
                        current_price=current_price,
                        mtf_levels=mtf_levels,
                        base_timeframe=timeframe
                    )
                    if level_sequence_prediction:
                        print(f"✓ Level sequence prediction: {level_sequence_prediction['total_levels_predicted']} levels")
            except Exception as e:
                print(f"⚠ Multi-timeframe model loading/prediction failed: {e}")
                import traceback
                traceback.print_exc()
                model_mtf = None
                level_sequence_prediction = None
        
        if TORCH_AVAILABLE and LevelBasedLSTM is not None:
            try:
                if os.path.exists(model_path):
                    # Try to load model with HOD/LOD prediction capability
                    try:
                        model = LevelBasedLSTM(n_features=recent_features.shape[1], max_levels=50)
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    except:
                        # Fallback to old model format (without HOD/LOD heads)
                        model = LevelBasedLSTM(n_features=recent_features.shape[1])
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    
                    # Get base prediction (with volatility for time calculation)
                    prediction = predict_price_target(model, recent_features, current_price, sigma_price=sigma_price, volatility_factor=1.0)
                    
                    # HOD/LOD level prediction now handled by FVECM pipeline
                    hod_lod_prediction = None
                    
                    # Run Monte Carlo simulation (with regimes)
                    # Use fewer simulations for production (30 instead of 100 for speed)
                    # Can be increased for higher accuracy if needed
                    print("Running Monte Carlo LSTM simulation...")
                    monte_carlo_result = monte_carlo_lstm_forecast(
                        model=model,
                        recent_features=recent_features,
                        current_price=current_price,
                        theoretical_hod=theoretical_hod_id,
                        theoretical_lod=theoretical_lod_id,
                        levels=all_levels[:20],  # Top 20 levels for reactions
                        volume_profile=volume_profile,
                        sigma_price=sigma_price,
                        hurst_data=hurst_data,
                        garch_regime=garch_regime,
                        hmm_regime=hmm_regime,
                        microstructure_state=microstructure_state,
                        n_simulations=30,  # Reduced from 100 for production performance
                        forecast_bars=30
                    )
                    
                    if prediction:
                        print(f"✓ LSTM prediction: target={prediction['target_price']:.2f}, confidence={prediction['confidence']:.2f}")
                    if monte_carlo_result:
                        print(f"✓ Monte Carlo: {monte_carlo_result['probabilities']['up']*100:.1f}% up, {monte_carlo_result['probabilities']['down']*100:.1f}% down")
                else:
                    print(f"⚠ Model file not found: {model_path}. Using level-based estimate.")
            except Exception as e:
                print(f"⚠ Model loading/prediction failed: {e}. Using level-based estimate.")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠ PyTorch not available. Using level-based heuristic estimate.")
        
        # 5. Fallback: If no model, use level-based heuristic with volume profile
        if prediction is None:
            # Use volume profile POC as directional guide
            if volume_profile:
                poc = volume_profile.get('poc', current_price)
                va_high = volume_profile.get('value_area_high', current_price)
                va_low = volume_profile.get('value_area_low', current_price)
                
                # If current price is below POC, likely to move toward POC
                if current_price < poc:
                    target_price = min(poc, va_high)
                elif current_price > poc:
                    target_price = max(poc, va_low)
                else:
                    target_price = current_price * 1.01
            else:
                # Find nearest levels above/below
                levels_above = [l for l in all_levels if l.get('price', 0) > current_price]
                levels_below = [l for l in all_levels if l.get('price', 0) < current_price]
                
                if levels_above:
                    nearest_resistance = min(levels_above, key=lambda x: x.get('price', float('inf')))
                    target_price = nearest_resistance.get('price', current_price * 1.01)
                else:
                    target_price = theoretical_hod_id
            
            target_pct_move = (target_price - current_price) / current_price * 100
            confidence = 0.6  # Moderate confidence for heuristic
            
            prediction = {
                'target_price': float(target_price),
                'target_pct_move': float(target_pct_move),
                'confidence': float(confidence),
                'expected_time_bars': 20,  # Default estimate
                'attention_weights': None
            }
        
        # 6. Find which level the model is targeting
        target_price = prediction['target_price']
        all_levels = hdbscan_levels + optics_levels + interaction_levels + ml_confluence_levels + neural_network_levels
        
        # Find closest level to predicted target
        closest_level = None
        if all_levels:
            closest_level = min(all_levels, key=lambda l: abs(l.get('price', current_price) - target_price))
        
        # Calculate timeframe multiplier for time estimate
        timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}.get(timeframe, 5)
        
        # Build response with all new features
        response_data = {
            'success': True,
            'question': 'Where is price going today?',
            'answer': {
                'target_price': prediction['target_price'],
                'target_pct_move': prediction['target_pct_move'],
                'confidence': prediction['confidence'],
                'expected_time_minutes': prediction['expected_time_bars'] * timeframe_minutes,
                'expected_time_bars': prediction['expected_time_bars'],
                'expected_time_description': f"Estimated {prediction['expected_time_bars']} bars ({prediction['expected_time_bars'] * timeframe_minutes} minutes) until price reaches target",
                'closest_level': sanitize_for_json(closest_level) if closest_level else None,
                'attention_focus': {
                    'most_important_bar': int(np.argmax(prediction['attention_weights'])) if prediction.get('attention_weights') else None,
                    'weights': prediction.get('attention_weights')
                } if prediction.get('attention_weights') else None
            },
            'theoretical_bounds': {
                'hod_premarket': float(theoretical_hod_pm),
                'lod_premarket': float(theoretical_lod_pm),
                'hod_intraday': float(theoretical_hod_id),
                'lod_intraday': float(theoretical_lod_id)
            },
            'volume_profile': sanitize_for_json(volume_profile) if volume_profile else None,
            'level_reactions': sanitize_for_json(level_reactions[:10]) if level_reactions else [],  # Top 10 closest
            'hod_lod_predictions': sanitize_for_json(hod_lod_predictions[:5]) if hod_lod_predictions else [],  # Top 5 most likely
            'lstm_hod_lod_prediction': sanitize_for_json(hod_lod_prediction) if 'hod_lod_prediction' in locals() and hod_lod_prediction else None,  # LSTM prediction of which level becomes HOD/LOD
            'level_sequence_prediction': sanitize_for_json(level_sequence_prediction) if 'level_sequence_prediction' in locals() and level_sequence_prediction else None,  # Multi-timeframe level sequence prediction
            'monte_carlo': sanitize_for_json(monte_carlo_result) if monte_carlo_result else None,
            'model_used': 'MTF Level Sequence LSTM' if level_sequence_prediction else ('LSTM + Monte Carlo' if monte_carlo_result else ('LSTM' if model is not None else 'Level-based heuristic')),
            'levels_detected': {
                'hdbscan': len(hdbscan_levels),
                'optics': len(optics_levels),
                'interaction': len(interaction_levels),
                'ml_confluence': len(ml_confluence_levels),
                'multiscale': len(multiscale_levels),
                'neural_network': len(neural_network_levels),
                'deepsupp': len(deepsupp_levels)
            },
            'microstructure_state': sanitize_for_json(microstructure_state) if microstructure_state else None
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/lstm-forecast: {error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/train-level-detector', methods=['POST'])
def api_train_level_detector():
    """
    Train the neural network level detector
    
    POST body (JSON):
    {
        "ticker": "SPY" (optional, default: "SPY"),
        "timeframe": "1d" (optional, default: "1d"),
        "lookback": 100 (optional, default: 100),
        "epochs": 50 (optional, default: 50),
        "batch_size": 32 (optional, default: 32)
    }
    """
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    try:
        data = request.get_json() or {}
        ticker = data.get('ticker', 'SPY')
        timeframe = data.get('timeframe', '1d')
        lookback = int(data.get('lookback', 100))
        epochs = int(data.get('epochs', 50))
        batch_size = int(data.get('batch_size', 32))
        
        result = train_level_detection_network(
            ticker=ticker,
            timeframe=timeframe,
            lookback=lookback,
            epochs=epochs,
            batch_size=batch_size
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/train-level-detector: {error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/train-deepsupp-levels', methods=['POST'])
def api_train_deepsupp_levels():
    """
    Train the DeepSupp v4 structural level model.

    POST body (JSON):
    {
        "ticker": "SPY"           (optional, default: "SPY"),
        "timeframe": "1d"         (optional, default: "1d"),
        "vol_lookback": 20        (optional, default: 20),
        "corr_window": 20         (optional, default: 20),
        "seq_len": 16             (optional, default: 16),
        "epochs": 50              (optional, default: 50),
        "batch_size": 32          (optional, default: 32),
        "model_path": "deepsupp_v4.pt"  (optional, default: "deepsupp_v4.pt")
    }
    """
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']

    try:
        data = request.get_json() or {}
        ticker = data.get('ticker', 'SPY')
        timeframe = data.get('timeframe', '1d')
        vol_lookback = int(data.get('vol_lookback', 20))
        corr_window = int(data.get('corr_window', 20))
        seq_len = int(data.get('seq_len', 16))
        epochs = int(data.get('epochs', 50))
        batch_size = int(data.get('batch_size', 32))
        model_path = data.get('model_path', 'deepsupp_v4.pt')

        result = train_deepsupp_level_model(
            ticker=ticker,
            timeframe=timeframe,
            vol_lookback=vol_lookback,
            corr_window=corr_window,
            seq_len=seq_len,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path,
        )

        status = 200 if result.get('success') else 400
        return jsonify(result), status
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/train-deepsupp-levels: {error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 400





# Ensure DB exists on startup (works with Gunicorn)
@app.before_request
def ensure_db():
    """Ensure database is initialized before any request"""
    global DB_PATH
    try:
        # Quick check if table exists
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not c.fetchone():
            conn.close()
            init_db()
        else:
            conn.close()
    except Exception as e:
        print(f"⚠ Database check error: {e}")
        # Fallback to users.db if there's an error
        try:
            if DB_PATH != 'users.db':
                DB_PATH = 'users.db'
            init_db()
        except Exception as e2:
            print(f"⚠ Database initialization failed: {e2}")
            # Don't crash the app, just log the error

# Initialize database on module load
# Wrap in try-except to prevent startup failure
try:
    init_db()
except Exception as e:
    print(f"⚠ Warning: Database initialization failed on startup: {e}")
    print("⚠ Will retry on first request via ensure_db()")
    # Don't crash - let the app start and retry on first request

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)                                                                                                                                                                                                                                


