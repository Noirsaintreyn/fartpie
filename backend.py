from flask import Flask, jsonify, request, session, render_template, redirect, url_for
from functools import wraps
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import hdbscan
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, OSError):
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    MarkovRegression = None

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
from datetime import datetime, timedelta, timezone
import sqlite3
import json
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple

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
            # yfinance's 1h-data 730-day cap rejects period="24mo" even
            # though it's nominally the same span as "2y" (date-math
            # quirk) - "2y" fetches fine, "24mo" returns zero bars with a
            # "must be within the last 730 days" error. Use "2y" whenever
            # the cap is hit instead of the equivalent "24mo" string.
            period = "2y" if adjusted_months >= 24 else f"{adjusted_months}mo"
    
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
import joblib
import contextlib
from arch import arch_model
from arch.univariate import SkewStudent
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

# ============================================================================
# MOTIVEWAVE DATA STORE — uploaded CSVs take priority over yfinance
# ============================================================================

import io
MOTIVEWAVE_DATA: dict = {}  # key: "TICKER_TIMEFRAME" → pd.DataFrame

def mw_key(ticker: str, timeframe: str) -> str:
    return f"{ticker.upper()}_{timeframe.lower()}"



def get_motivewave_data(ticker: str, timeframe: str) -> pd.DataFrame | None:
    """Return uploaded MotiveWave data for this ticker/timeframe combo, or None."""
    key = mw_key(ticker, timeframe)
    df = MOTIVEWAVE_DATA.get(key)
    if df is not None and not df.empty:
        return df
    # Also check if 1h data was uploaded and we need 4h (resample on the fly)
    if timeframe == '4h':
        df1h = MOTIVEWAVE_DATA.get(mw_key(ticker, '1h'))
        if df1h is not None and not df1h.empty:
            return resample_ohlcv(df1h, '4h')
    return None

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
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://degencap.uk,https://www.degencap.uk,https://peepeepoop.workers.dev,https://peepeepoop.pages.dev').split(',')
CORS(app, supports_credentials=True, origins=CORS_ORIGINS)

@app.route("/api/health")
def health():
    return {"status": "backend live", "zone_target_pipeline": "corridor_buffer_0.3"}


# ============================================================================
# MACRO REGIME READ - "is today likely to trend or chop", per instrument,
# from cross-asset macro signals (VIX/yields/USD + sector-specific tilt).
# Models trained offline (backtest_macro_regime.py), walk-forward validated,
# loaded here from the committed .pkl files - NOT retrained live. Only NQ
# is actually validated (z=2.74, p=0.006); ES/GC/CL/SI/AMD/AAPL either
# showed no significant edge or were never backtested - the endpoint says
# so explicitly rather than presenting every instrument as equally trustworthy.
# ============================================================================
_MACRO_MODELS_CACHE = {}
_MACRO_INSTRUMENT_MAP = {
    'NQ=F': 'NQ', 'NQ': 'NQ', '^NDX': 'NQ',
    'ES=F': 'ES', 'ES': 'ES', '^GSPC': 'ES', 'SPY': 'ES',
    'GC=F': 'GC', 'GC': 'GC',
    'CL=F': 'CL', 'CL': 'CL',
    'SI=F': 'SI', 'SI': 'SI',
    'AMD': 'AMD',
    'AAPL': 'AAPL',
}
_MACRO_PRICE_TICKER = {'NQ': 'NQ=F', 'ES': 'ES=F', 'GC': 'GC=F', 'CL': 'CL=F',
                        'SI': 'SI=F', 'AMD': 'AMD', 'AAPL': 'AAPL'}






# name aliases: feature names use short forms ("yield", "dollar") that
# don't match the raw series dict keys ("yield10y", "dollar") 1:1 in
# every case - kept explicit rather than guessing string transforms
_MACRO_SERIES_ALIAS = {'yield': 'yield10y'}
_MACRO_DIFF_NOT_PCT = {'yield10y'}  # yields move in absolute bps, not %, everything else uses % change













# ============================================================================
# MOTIVEWAVE CSV UPLOAD ROUTES
# ============================================================================



  
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
            
            # --- ML eval table: predictions + realized + intermediates ---
            c.execute('''
            CREATE TABLE IF NOT EXISTS hodlod_eval (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                ticker TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                pred_ts TEXT NOT NULL,          -- ISO timestamp of prediction moment
                session_date TEXT NOT NULL,     -- YYYY-MM-DD (the session this prediction belongs to)

                spot REAL,

                sigma_daily_pct REAL,
                sigma_price REAL,

                micro_state TEXT,
                micro_conf REAL,
                garch_regime TEXT,
                lss REAL,

                -- Intermediate pipeline outputs (so we can blame modules)
                base_hod REAL,
                base_lod REAL,

                lss_hod REAL,
                lss_lod REAL,
                lss_meta TEXT,

                oi_hod REAL,
                oi_lod REAL,
                oi_meta TEXT,

                rf_hod REAL,
                rf_lod REAL,
                rf_meta TEXT,

                final_hod REAL,
                final_lod REAL,

                -- Realized
                realized_hod REAL,
                realized_lod REAL,
                realized_ts TEXT,

                -- Feature snapshot used by ML_FEATURES
                features_json TEXT
            )
            ''')

            # --- optional: store learned calibration knobs per regime bucket ---
            c.execute('''
            CREATE TABLE IF NOT EXISTS hodlod_calibration (
                key TEXT PRIMARY KEY,           -- e.g. "Fock|highLSS|highVol"
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                params_json TEXT                -- {"tail_mult":1.1,"oi_clip_mult":0.6,"rf_clip":1.7}
            )
            ''')

            # --- shadow-mode v3 filter scoring log: v2's real live decision
            # next to v3's score/decision on the SAME candidate, for every
            # candidate v2 scores at the 4 real-time call sites (not the
            # ou-zone-history bulk-scan endpoint - that loop calls the
            # filter per historical bar and would multiply cost heavily).
            # Never read by any live request path - purely for offline
            # v2-vs-v3 comparison once enough rows/outcomes accrue. The
            # realized_* columns are filled in later by a separate backfill
            # job (not yet built) that revisits each logged candidate after
            # its outcome window closes - NULL until then.
            c.execute('''
            CREATE TABLE IF NOT EXISTS shadow_v3_scores (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                endpoint TEXT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_ts TEXT,

                category TEXT NOT NULL,
                price REAL NOT NULL,
                current_price REAL,

                features_json TEXT,
                model_version TEXT,

                v2_decision INTEGER,
                v3_raw_proba REAL,
                v3_calibrated_proba REAL,
                calibration_source TEXT,
                v3_shadow_decision INTEGER,

                state_age_seconds REAL,
                snapshot_cache_hit INTEGER,
                scoring_latency_ms REAL,

                realized_bounced INTEGER,
                realized_mfe_atr REAL,
                realized_mae_atr REAL,
                realized_ts TEXT
            )
            ''')

            conn.commit()
            
            # Admin account: rey / admin
            rey_password = hash_password('admin')
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
            
            # Remove old admin and test accounts if they exist
            c.execute("DELETE FROM users WHERE username = 'admin' AND is_admin = 1")
            c.execute("DELETE FROM users WHERE username IN ('test1', 'test2')")
            conn.commit()
            
            # Account 2: user1 / pw
            user1_password = hash_password('pw')
            try:
                c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                          ('user1', 'user1@degendiscovery.com', user1_password, 0))
                conn.commit()
                print("✓ Account (user1) created")
            except sqlite3.IntegrityError:
                c.execute("UPDATE users SET password = ? WHERE username = 'user1'", (user1_password,))
                conn.commit()
                print("✓ Account (user1) already exists, password updated")
            
            # Account 3: user2 / 67
            user2_password = hash_password('67')
            try:
                c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                          ('user2', 'user2@degendiscovery.com', user2_password, 0))
                conn.commit()
                print("✓ Account (user2) created")
            except sqlite3.IntegrityError:
                c.execute("UPDATE users SET password = ? WHERE username = 'user2'", (user2_password,))
                conn.commit()
                print("✓ Account (user2) already exists, password updated")
        # Connection auto-closes here via context manager
        
        conn.close()
        print(f"✓ Database initialized at: {DB_PATH}")
    except Exception as e:
        print(f"⚠ Database initialization error: {e}")
        # Try fallback to current directory
        if DB_PATH != 'users.db':
            DB_PATH = 'users.db'
            init_db()


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
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    # Handle native Python float with inf/nan
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
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
    
    # NOTE: LSS adjustments are applied in adjust_hod_lod_usage, not here
    # This prevents double-adjustment of permeability and overshoot bias
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


def fit_gjr_garch_vol_forecast_pct(returns_pct):
    """
    GJR-GARCH(1,1,1) 1-step vol forecast, in the same percent units as
    returns_pct. The o=1 asymmetry term captures the leverage effect
    (vol reacts more to down moves than up moves) that plain GARCH
    (fit_garch_model) misses - used by score_and_filter_levels_v2's
    vwap_distance_norm/vol_forecast_pct_of_price/gjr_vol_regime_ratio
    features. Returns None on failure (too little data, non-convergence
    treated as best-effort by the arch package itself).
    """
    try:
        if len(returns_pct) < 50:
            return None
        model = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=1)
        return float(np.sqrt(forecast.variance.values[-1, 0]))
    except Exception:
        return None












V3_FEATURE_COLS = ['vwap_distance_norm', 'vol_forecast_pct_of_price', 'atr_distance',
                    'confluence', 'hurst', 'hmm_state_confidence', 'hmm_recent_flip',
                    'garman_klass_vol_pct', 'gjr_vol_regime_ratio', 'vwap_bias_alignment',
                    'ou_zone_distance_atr']

_MARKET_STATE_V3_CACHE = {}
_MARKET_STATE_V3_CACHE_MAX = 64  # bound memory - simple oldest-evicted cap, not a real LRU




_LEVEL_FILTER_V3_ARTIFACT = None








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
    current_vol = garch_vol_regime['current_vol']
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
        atm_vol_pct = garch_vol_regime['current_vol']  # Percentage (e.g., 20.0 = 20%)
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

def trend_reversion_scenario_forecast(prices, forecast_periods=10, num_scenarios=3):
    """Rolling-mean trend extraction + mean-reversion term + injected noise,
    branched into bullish/base/bearish scenarios via trend/vol multipliers.
    NOT an N-BEATS model (no basis expansion, no neural net, no training) -
    previously misleadingly named nbeats_forecast."""
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

def multiscale_weighted_average_forecast(prices, volumes, forecast_periods=10):
    """Exponentially-weighted average across 4 lookback scales (5/10/20/40
    bars) + linear trend extrapolation + damped noise. NOT a TCN (no
    convolutions, no dilated layers, no learned weights) - previously
    misleadingly named tcn_style_forecast."""
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
    nbeats = trend_reversion_scenario_forecast(closes, forecast_periods=forecast_periods, num_scenarios=3)
    if nbeats:
        forecasts['scenarios'] = nbeats
    tcn = multiscale_weighted_average_forecast(closes, volumes, forecast_periods=forecast_periods)
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
        # raw state IDs are arbitrary (label-switching - EM doesn't order
        # states by mean return), so map by fitted mean return instead of
        # raw index, or 'Bullish'/'Bearish' get coin-flip-attached to the
        # wrong state on a fresh fit
        ordered_states = np.argsort(model.means_[:, 0])
        regime_names = {int(ordered_states[0]): 'Bearish', int(ordered_states[1]): 'Neutral', int(ordered_states[2]): 'Bullish'}
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




def fractional_brownian_adjustment(base_hod, base_lod, hurst, sigma):
    """
    Adjust predictions based on Hurst exponent
    H > 0.5: trending (wider range)
    H < 0.5: mean-reverting (narrower range)
    """
    # Fractional scaling
    # σ_fBm(t) = σ × t^H  (vs. σ × √t for Brownian)
    
    if hurst > 0.6:  # Trending
        # Expect larger moves
        multiplier = 1.0 + 0.3 * (hurst - 0.5) / 0.5  # Up to 1.3x
    elif hurst < 0.4:  # Mean-reverting
        # Expect smaller moves
        multiplier = 1.0 - 0.2 * (0.5 - hurst) / 0.5  # Down to 0.8x
    else:
        multiplier = 1.0
    
    mid = (base_hod + base_lod) / 2
    hod_dist = base_hod - mid
    lod_dist = mid - base_lod
    
    adj_hod = mid + hod_dist * multiplier
    adj_lod = mid - lod_dist * multiplier
    
    return float(adj_hod), float(adj_lod)

# [ALL THE LEVEL DETECTION FUNCTIONS - KEEPING THEM EXACTLY AS BEFORE]

def _volume_weighted_price_array(highs, lows, closes, volumes, max_replication=5):
    """Approximate volume-weighted density input for the clustering-based
    level detectors, which otherwise treat every high/low/close point as
    equal mass regardless of how much actually traded there. Repeats each
    bar's three price points a number of times proportional to that bar's
    volume relative to the window's mean (so a bar that's ~1x average
    volume gets ~1x weight, ~3x average gets ~3x, etc.), clipped to
    max_replication so a single outlier volume spike can't blow up the
    point count. Mean replication is ~1x by construction, so the resulting
    array stays close in scale to the unweighted np.concatenate([highs,
    lows, closes]) - existing adaptive parameters that key off n_samples
    (e.g. HDBSCAN's min_cluster_size) don't need retuning.

    volumes=None -> returns the plain unweighted concatenation, identical
    to every function's original behavior (backward compatible - no
    current live caller passes volumes, so nothing live changes)."""
    if volumes is None:
        return np.concatenate([highs, lows, closes])
    vol = np.asarray(volumes, dtype=float)
    vol_norm = vol / (vol.mean() + 1e-9)
    reps = np.clip(np.round(vol_norm), 1, max_replication).astype(int)
    return np.concatenate([np.repeat(highs, reps), np.repeat(lows, reps), np.repeat(closes, reps)])


def calculate_meanshift_levels(highs, lows, closes, volumes=None):
    all_prices = _volume_weighted_price_array(highs, lows, closes, volumes).reshape(-1, 1)
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


# Same clustering algorithms used at the live call sites, callable
# generically as (highs, lows, closes, timeframe) -> levels, so
# extend_thin_side_levels() can re-run the SAME validated detectors on a
# longer window instead of inventing a different method for the thin side.
# Excludes categories with a different call signature (Neural Net, Local
# Interaction, Wyckoff, Time-Weighted HDBSCAN) - out of scope for a first
# pass, most of the level count in practice comes from these 8.
_THIN_SIDE_ALGORITHMS = {
    'GMM': lambda h, l, c, tf: calculate_gmm_levels(h, l, c),
    'TDA': lambda h, l, c, tf: persistent_homology_levels(h, l, c, max_levels=8),
    'HDBSCAN': lambda h, l, c, tf: calculate_hdbscan_levels(h, l, c, timeframe=tf),
    'OPTICS': lambda h, l, c, tf: enhanced_optics_levels(h, l, c, timeframe=tf),
    'KDE': lambda h, l, c, tf: kde_based_levels(h, l, c, n_levels=10),
    'Isolation-Forest': lambda h, l, c, tf: find_pivot_anomalies(h, l, c),
    'MeanShift': lambda h, l, c, tf: calculate_meanshift_levels(h, l, c),
    'Multiscale-HDBSCAN': lambda h, l, c, tf: multiscale_hdbscan_levels(h, l, c, timeframe=tf),
}

# Deliberately much longer than the default live fetch (5-10 days for
# intraday futures, per the period_map at get_data()) - the whole point is
# to look further back than the normal window to recover real prior highs/
# lows that a fresh breakout has simply moved outside of.
_THIN_SIDE_EXTENDED_PERIOD = {
    ('1m', True): '1mo', ('5m', True): '1mo', ('15m', True): '2mo',
    ('1h', True): '3mo', ('4h', True): '6mo',
    ('1m', False): '1mo', ('5m', False): '2mo', ('15m', False): '3mo',
    ('1h', False): '1y', ('4h', False): '1y',
}


def extend_thin_side_levels(ticker, timeframe, highs, lows, closes, current_price,
                             levels_by_category, is_futures, min_levels_per_side=3, side_band_atr=15):
    """
    If one side (resistance above current_price, support below) has fewer
    than min_levels_per_side raw candidates within side_band_atr*ATR of
    current price, fetches a LONGER historical window (same source, same
    ticker, just further back) and re-runs the SAME clustering algorithms
    already used at this call site against it, merging in only the NEW
    candidates that land on the thin side.

    This does not invent anything - it recovers real prior price structure
    (e.g. a swing high from 6 weeks ago) that the short default live window
    (5-10 days for intraday futures) simply doesn't include, which is
    exactly what happens on a breakout day: the balanced side has plenty of
    recent consolidation to cluster, the breakout side has almost no
    trading history within the short window because price is making new
    highs/lows relative to it. Never touches the side that already has
    enough coverage. Merged candidates still go through the same ML filter
    as everything else - nothing here bypasses validation, it only adds
    more raw candidates for the filter to judge.

    Fails safe: any error (bad ticker, no data, algorithm exception)
    returns levels_by_category unchanged, never raises into the caller.
    """
    try:
        atr = _atr_for_ml_filter(highs, lows, closes)
        if atr is None or atr <= 0:
            return levels_by_category

        resistance_n = sum(1 for lvls in levels_by_category.values() for lvl in (lvls or [])
                            if lvl.get('price') is not None and current_price < lvl['price'] <= current_price + side_band_atr * atr)
        support_n = sum(1 for lvls in levels_by_category.values() for lvl in (lvls or [])
                         if lvl.get('price') is not None and current_price - side_band_atr * atr <= lvl['price'] < current_price)

        if resistance_n < min_levels_per_side and support_n >= min_levels_per_side:
            thin_side = 'resistance'
        elif support_n < min_levels_per_side and resistance_n >= min_levels_per_side:
            thin_side = 'support'
        else:
            return levels_by_category  # balanced, or both thin (not this mechanism's job)

        period = _THIN_SIDE_EXTENDED_PERIOD.get((timeframe, is_futures))
        if period is None:
            return levels_by_category

        if timeframe == '4h':
            wide_hist = fetch_historical_data_with_resampling(ticker=ticker, timeframe='4h', period=period, is_futures=is_futures)
        else:
            interval = '60m' if (is_futures and timeframe == '1h') else timeframe
            wide_hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if wide_hist is None or len(wide_hist) < 60:
            return levels_by_category
        w_h, w_l, w_c = wide_hist['High'].values, wide_hist['Low'].values, wide_hist['Close'].values

        extended = {cat: list(lvls) if lvls else [] for cat, lvls in levels_by_category.items()}
        added = 0
        for category in levels_by_category.keys():
            fn = _THIN_SIDE_ALGORITHMS.get(category)
            if fn is None:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    wide_levels = fn(w_h, w_l, w_c, timeframe) or []
            except Exception:
                continue
            existing_prices = {round(lvl['price'], 4) for lvl in extended[category] if lvl.get('price') is not None}
            for lvl in wide_levels:
                price = lvl.get('price')
                if price is None or round(price, 4) in existing_prices:
                    continue
                if thin_side == 'resistance' and not (current_price < price <= current_price + side_band_atr * atr):
                    continue
                if thin_side == 'support' and not (current_price - side_band_atr * atr <= price < current_price):
                    continue
                lvl = dict(lvl)
                lvl['from_extended_lookback'] = True
                lvl['extended_lookback_period'] = period
                extended[category].append(lvl)
                added += 1

        if added:
            print(f"extend_thin_side_levels: {ticker} {timeframe} thin_side={thin_side} "
                  f"(resistance={resistance_n}, support={support_n}) -> added {added} candidates from {period} lookback")
        return extended
    except Exception as e:
        print(f"extend_thin_side_levels failed (non-fatal, using original levels): {e}")
        return levels_by_category


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


def calculate_hdbscan_levels(highs, lows, closes, timeframe='1d', volumes=None):
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
    all_prices = _volume_weighted_price_array(highs, lows, closes, volumes)
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

def enhanced_optics_levels(highs, lows, closes, timeframe='1d', volumes=None):
    """
    OPTICS with reachability-based strength scoring
    Reachability distance = "how dense is this cluster?"
    Better than HDBSCAN for some patterns
    """
    if len(closes) < 20:
        return []

    all_prices = _volume_weighted_price_array(highs, lows, closes, volumes).reshape(-1, 1)
    
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

def kde_based_levels(highs, lows, closes, n_levels=10, volumes=None):
    """
    Find levels using kernel density estimation
    Peaks in density = strong levels
    """
    all_prices = np.concatenate([highs, lows, closes])

    # Adaptive bandwidth (Scott's rule). gaussian_kde supports weights
    # natively (unlike sklearn's clustering estimators) - each bar's
    # volume, normalized, weighted directly rather than approximated via
    # point replication like the other volume-weighted detectors.
    kde_weights = None
    if volumes is not None:
        vol = np.asarray(volumes, dtype=float)
        vol_norm = vol / (vol.mean() + 1e-9)
        kde_weights = np.concatenate([vol_norm, vol_norm, vol_norm])
    kde = gaussian_kde(all_prices, bw_method='scott', weights=kde_weights)
    
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


def _atr_for_ml_filter(highs, lows, closes, period=14):
    """Same ATR formula used in backtest_levels.py's compute_atr - must match
    exactly, since the ML filter below was validated against that specific
    computation."""
    if len(closes) < 2:
        return 0.0
    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    return float(np.mean(tr[-period:]))


_ML_FILTER_INTERCEPT = -0.28262707263138526
_ML_FILTER_COEF_VWAP_DIST = -0.030765693587465632
_ML_FILTER_COEF_VOL_FORECAST_PCT = -0.002548414498073197
_ML_FILTER_COEF_ATR_DIST = -0.036833278406937744
_ML_FILTER_OFFSETS = {
    'GMM': -0.021134471847448096,
    'HDBSCAN': -0.04392717204472848,
    'Isolation-Forest': -0.06601702304382738,
    'KDE': -0.02285343878326555,
    'MeanShift': -0.061615015155393416,
    'OPTICS': -0.04047284239466215,
    'TDA': -0.026448081888226352,
}
_ML_FILTER_THRESHOLDS = {
    'GMM': 0.427390,
    'HDBSCAN': 0.420976,
    'Isolation-Forest': 0.423303,
    'KDE': 0.427252,
    'MeanShift': 0.419031,
    'OPTICS': 0.421409,
    'TDA': 0.429716,
}




# ============================================================================
# ML FILTER v2 - Hurst, HMM regime-change, GJR-GARCH, Garman-Klass,
# confluence, VWAP bias/stretch added on top of the v1 filter above.
# ============================================================================
# Coefficients: logistic regression fit on the FULL v2 dataset
# (backtest_ml_filter_v2_events.csv, ~150k events, 12yr NQ/ES 1H+4H), same
# 7 production categories as v1. Walk-forward validated (16 folds,
# validate_ml_filter_walkforward.py --feature-set v2): 6/7 categories at
# 100% fold consistency (Isolation-Forest 93.8%), all 7 clear Bonferroni
# correction. Filtered accuracy 0.483-0.504 (v1 was 0.480-0.497) - the
# real improvement isn't the headline accuracy, it's that pred_proba now
# discriminates WITHIN the filtered population (pooled walk-forward
# holdout, top-25%-of-filtered vs rest-of-filtered accuracy spread):
# GMM +0.056, HDBSCAN +0.065, Isolation-Forest +0.041, KDE +0.053,
# MeanShift +0.064, OPTICS +0.060, TDA +0.062. v1 had essentially none of
# this (flat scores regardless of outcome), which is exactly what made the
# trade simulator's "take the single highest-scoring candidate" rule
# meaningless - see simulate_prop_challenge.py history.
_ML_FILTER_V2_INTERCEPT = -0.09550514020085478
_ML_FILTER_V2_COEF_VWAP_DIST = -0.03117995003161389
_ML_FILTER_V2_COEF_VOL_FORECAST_PCT = -0.0013540725253570474
_ML_FILTER_V2_COEF_ATR_DIST = -0.03223406859338196
_ML_FILTER_V2_COEF_CONFLUENCE = -0.017667877441859152
_ML_FILTER_V2_COEF_HURST = 0.3346999532374335
_ML_FILTER_V2_COEF_HMM_CONFIDENCE = 0.04804291584223122
_ML_FILTER_V2_COEF_HMM_FLIP = -0.16367347474479166
_ML_FILTER_V2_COEF_GK_VOL_PCT = -0.13773557187129407
_ML_FILTER_V2_COEF_GJR_REGIME_RATIO = -0.1410323984314109
_ML_FILTER_V2_COEF_VWAP_BIAS = -0.001663097004885809
_ML_FILTER_V2_OFFSETS = {
    'GMM': 0.009732512315720157,
    'HDBSCAN': -0.025645667487374405,
    'Isolation-Forest': -0.04837356634313406,
    'KDE': 0.021995201026009485,
    'MeanShift': -0.033440097487212564,
    'OPTICS': -0.013081112628211837,
    'TDA': -0.006270496723109859,
}
_ML_FILTER_V2_THRESHOLDS = {
    'GMM': 0.427264,
    'HDBSCAN': 0.419601,
    'Isolation-Forest': 0.418986,
    'KDE': 0.427638,
    'MeanShift': 0.418313,
    'OPTICS': 0.421419,
    'TDA': 0.427575,
}


def score_and_filter_levels_v2(levels_by_category, highs, lows, opens, closes, volumes,
                                current_price, timestamps=None, confluence_atr_mult=0.5):
    """
    v2 of score_and_filter_levels: same 7 validated categories and same
    "return only levels clearing their category's threshold" contract, but
    scored with the expanded, walk-forward-validated feature set (see
    module comment above _ML_FILTER_V2_INTERCEPT).

    Needs `opens` (not required by v1) for the Garman-Klass realized-vol
    feature - pass the same window's open prices alongside highs/lows/closes.

    Latency note: this fits a GJR-GARCH model AND an HMM per call (on top
    of the plain-GARCH v1 already fit elsewhere), roughly 2-4x the compute
    of score_and_filter_levels. Fine for backtesting; for a live request
    path, profile before assuming it's free.
    """
    all_raw = [lvl for lvls in levels_by_category.values() for lvl in (lvls or [])]
    if len(closes) < 60:
        return all_raw

    try:
        vwap_result = calculate_vwap(highs, lows, closes, volumes, timestamps=timestamps)
        vwap = vwap_result['vwap'] if vwap_result else current_price
    except Exception:
        vwap = current_price

    returns_pct = np.diff(np.log(closes)) * 100
    gjr_vol_pct = fit_gjr_garch_vol_forecast_pct(returns_pct)
    if gjr_vol_pct is None or gjr_vol_pct <= 0:
        return all_raw
    vol_forecast = gjr_vol_pct / 100.0 * current_price
    vol_forecast_pct = vol_forecast / current_price

    atr = _atr_for_ml_filter(highs, lows, closes)
    if atr <= 0:
        return all_raw

    try:
        gk_vol_pct = garman_klass_daily_volatility(opens, highs, lows, closes) * 100
    except Exception:
        gk_vol_pct = None
    if gk_vol_pct is None or gk_vol_pct <= 0:
        return all_raw
    gjr_vol_regime_ratio = gjr_vol_pct / (gk_vol_pct + 1e-9)

    try:
        hurst = calculate_hurst_exponent(closes)['hurst']
    except Exception:
        hurst = 0.5  # neutral fallback (random walk)

    hmm_confidence, hmm_flip = 0.5, 0
    if HMMLEARN_AVAILABLE and GaussianHMM is not None and len(closes) >= 60:
        try:
            hmm_returns = np.diff(np.log(closes)).reshape(-1, 1)
            hmm_model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
            hmm_model.fit(hmm_returns)
            states = hmm_model.predict(hmm_returns)
            post = hmm_model.predict_proba(hmm_returns)
            hmm_confidence = float(post[-1, states[-1]])
            hmm_flip = int(states[-1] != states[max(0, len(states) - 1 - 5)])
        except Exception:
            hmm_confidence, hmm_flip = 0.5, 0

    # confluence needs every candidate across all categories up front
    all_candidates = [(lvl.get('price'), cat) for cat, lvls in levels_by_category.items()
                       for lvl in (lvls or []) if lvl.get('price') is not None]
    by_type = {}
    for price, cat in all_candidates:
        by_type.setdefault(cat, []).append(price)
    tol = confluence_atr_mult * atr

    def _confluence_count(price, own_type):
        count = 0
        for cat, prices in by_type.items():
            if cat == own_type:
                continue
            if any(abs(p - price) <= tol for p in prices):
                count += 1
        return count

    def _score(price, category, offset):
        atr_dist = (price - current_price) / atr
        confluence = _confluence_count(price, category)

        side = 'support' if price < current_price else 'resistance'
        trade_direction = 'long' if side == 'support' else 'short'
        price_vs_vwap = current_price - vwap
        aligned_with_bias = (trade_direction == 'long' and price_vs_vwap > 0) or \
                             (trade_direction == 'short' and price_vs_vwap < 0)
        vwap_stretch = abs(price_vs_vwap) / (vol_forecast + 1e-9)
        vwap_bias_alignment = vwap_stretch if aligned_with_bias else -vwap_stretch
        vwap_dist_norm = (price - vwap) / (vol_forecast + 1e-9)

        logit = (_ML_FILTER_V2_INTERCEPT + offset +
                 _ML_FILTER_V2_COEF_VWAP_DIST * vwap_dist_norm +
                 _ML_FILTER_V2_COEF_VOL_FORECAST_PCT * vol_forecast_pct +
                 _ML_FILTER_V2_COEF_ATR_DIST * atr_dist +
                 _ML_FILTER_V2_COEF_CONFLUENCE * confluence +
                 _ML_FILTER_V2_COEF_HURST * hurst +
                 _ML_FILTER_V2_COEF_HMM_CONFIDENCE * hmm_confidence +
                 _ML_FILTER_V2_COEF_HMM_FLIP * hmm_flip +
                 _ML_FILTER_V2_COEF_GK_VOL_PCT * gk_vol_pct +
                 _ML_FILTER_V2_COEF_GJR_REGIME_RATIO * gjr_vol_regime_ratio +
                 _ML_FILTER_V2_COEF_VWAP_BIAS * vwap_bias_alignment)
        return 1.0 / (1.0 + np.exp(-logit))

    filtered = []
    for category, lvls in levels_by_category.items():
        offset = _ML_FILTER_V2_OFFSETS.get(category)
        threshold = _ML_FILTER_V2_THRESHOLDS.get(category)
        if offset is None or threshold is None:
            filtered.extend(lvls or [])
            continue
        for lvl in (lvls or []):
            price = lvl.get('price')
            if price is None:
                continue
            prob = _score(price, category, offset)
            if prob >= threshold:
                lvl = dict(lvl)
                lvl['ml_filter_score'] = float(prob)
                filtered.append(lvl)
    return filtered


def calculate_gmm_levels(highs, lows, closes, min_components=3, max_components=10, min_frac=0.03, volumes=None):
    """
    Gaussian Mixture Model: soft/probabilistic alternative to HDBSCAN.
    Clusters raw prices same as HDBSCAN, but each level's confidence comes
    from the component's posterior probability (graded) instead of a hard
    cluster assignment. Component count picked by BIC.

    Backtested (backtest_levels.py, 12yr NQ/ES 1H+4H): tied for the highest
    weighted-average support accuracy of any method tested (0.434), ahead
    of HDBSCAN (0.427) on the same data - see backtest_summary_fixed.csv.
    """
    if len(closes) < 20:
        return []
    all_prices = _volume_weighted_price_array(highs, lows, closes, volumes)
    X = all_prices.reshape(-1, 1)
    n = len(X)

    best_gmm, best_bic = None, np.inf
    for k in range(min_components, min(max_components, n // 5) + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200, n_init=1)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        except Exception:
            continue
    if best_gmm is None:
        return []

    labels = best_gmm.predict(X)
    probs = best_gmm.predict_proba(X)
    levels = []
    for k in range(best_gmm.n_components):
        mask = labels == k
        count = int(mask.sum())
        if count < max(5, n * min_frac):
            continue
        center = float(best_gmm.means_[k][0])
        if center <= 0:
            continue
        confidence = float(np.clip(probs[mask, k].mean(), 0, 0.95))
        levels.append({
            'price': center, 'type': 'GMM Cluster', 'touches': count,
            'strength': confidence, 'breakoutProb': float(1 - confidence),
            'reversionProb': confidence, 'category': 'GMM',
        })
    return levels

def multiscale_hdbscan_levels(highs, lows, closes, timeframe='1d', volumes=None):
    """
    Run HDBSCAN at multiple scales to catch both major and minor levels
    """
    all_prices = _volume_weighted_price_array(highs, lows, closes, volumes).reshape(-1, 1)
    
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
    
    # Calculate time weights (exponential decay). Use np.timedelta64 division
    # rather than .days - a raw numpy datetime64 array (e.g. hist.index.values)
    # yields numpy.timedelta64 on subtraction, which has no .days attribute
    # (only pandas.Timedelta does) - this silently zeroed out every weight
    # and made this algorithm return 0 levels everywhere it was called.
    current_time = pd.to_datetime(timestamps[-1]) if isinstance(timestamps[-1], (int, float)) else pd.to_datetime(timestamps[-1])
    all_times_dt = pd.to_datetime(all_times)
    time_diffs = ((current_time - all_times_dt) / pd.Timedelta(days=1)).to_numpy()
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
    Use 0-dimensional persistent homology (connected-component merging) to
    find price clusters that persist across multiple distance scales.

    FIXED: the previous version treated the raw (birth, death) pair from the
    ripser diagram as if it were a price coordinate - for 0-dim persistence
    on a 1D point cloud, birth is always 0 and death is the MERGE DISTANCE
    between two point-clusters, not a price. That produced nonsense "prices"
    like 23.6 on a $25,000 instrument. The correct level price is the mean
    price of the points in the cluster that persists - recovered here via
    scipy's single-linkage hierarchy, which is mathematically equivalent to
    0-dim persistent homology's merge tree (same persistence/"death"
    distances), but actually gives you cluster membership to compute a real
    price from.
    """
    if len(closes) < 20:
        return []

    all_prices = np.concatenate([highs, lows, closes])
    points = all_prices.reshape(-1, 1)

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(points, method='single')
        merge_distances = Z[:, 2]
        max_persistence = float(merge_distances.max()) if len(merge_distances) else 0.0
        if max_persistence <= 0:
            return []

        levels = []
        seen_prices = set()
        # Walk merge events from most persistent (largest gap closed) down,
        # cutting the dendrogram just below each merge distance to recover
        # the cluster that existed right before that merge.
        for dist in sorted(set(merge_distances), reverse=True)[:max_levels * 2]:
            cluster_ids = fcluster(Z, t=dist - 1e-9, criterion='distance')
            for cid in set(cluster_ids):
                member_prices = all_prices[cluster_ids == cid]
                if len(member_prices) < 3:
                    continue
                level_price = float(np.mean(member_prices))
                price_key = round(level_price, 2)
                if price_key in seen_prices:
                    continue

                persistence = dist
                strength = persistence / (max_persistence + 1e-9)
                if strength < 0.3:
                    continue

                touches = int(np.sum(np.abs(all_prices - level_price) < level_price * 0.005))
                levels.append({
                    'price': level_price,
                    'type': 'Persistent Homology',
                    'strength': float(strength),
                    'persistence': float(persistence),
                    'touches': touches,
                    'category': 'TDA',
                    'breakoutProb': float(1 - strength),
                    'reversionProb': float(strength)
                })
                seen_prices.add(price_key)

        return sorted(levels, key=lambda x: x['persistence'], reverse=True)[:max_levels]
    except Exception as e:
        print(f"Persistent Homology failed: {e}")
        return []








if TORCH_AVAILABLE and nn is not None:
    import torch.nn.functional as F

    class CausalConv1d(nn.Module):
        """Conv1d that only looks at past and present — no future leakage."""
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  padding=0, dilation=dilation)

        def forward(self, x):
            # Pad left side only: (left, right)
            x = F.pad(x, (self.padding, 0))
            return self.conv(x)

    class LevelDetectionNet(nn.Module):
        """
        Causal CNN + LSTM + MLP for S/R level detection.

        Based on Khairov et al. (2025) — causal convolutions prevent future
        data leakage, LSTM captures temporal dependencies, and an MLP head
        classifies each bar as level / not-level.

        Input : raw OHLCV per bar  (5 features)
        Output: per-bar logit (before sigmoid) indicating S/R probability
        """
        def __init__(self, lookback=100, in_channels=5, hidden_dim=64,
                     lstm_hidden=128, lstm_layers=2, dropout=0.2):
            super().__init__()
            self.lookback = lookback

            # Causal CNN feature extractor
            self.conv1 = CausalConv1d(in_channels, hidden_dim, kernel_size=5)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = CausalConv1d(hidden_dim, hidden_dim * 2, kernel_size=5, dilation=2)
            self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
            self.conv3 = CausalConv1d(hidden_dim * 2, hidden_dim, kernel_size=3, dilation=4)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            self.dropout_cnn = nn.Dropout(dropout)

            # Unidirectional LSTM (no future information)
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )

            # MLP classification head
            self.mlp = nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden // 2, lstm_hidden // 4),
                nn.ReLU(),
                nn.Linear(lstm_hidden // 4, 1),
            )

        def forward(self, x):
            """
            x: [batch, lookback, in_channels]  (OHLCV)
            returns: [batch, lookback]  logits
            """
            # Conv1d expects [batch, channels, seq]
            h = x.transpose(1, 2)

            h = self.dropout_cnn(torch.relu(self.bn1(self.conv1(h))))
            h = self.dropout_cnn(torch.relu(self.bn2(self.conv2(h))))
            h = self.dropout_cnn(torch.relu(self.bn3(self.conv3(h))))

            # Back to [batch, seq, features]
            h = h.transpose(1, 2)

            # LSTM temporal modelling
            h, _ = self.lstm(h)

            # Per-bar classification
            logits = self.mlp(h)  # [batch, seq, 1]
            return logits.squeeze(-1)  # [batch, seq]
else:
    # Dummy class when torch is not available
    class LevelDetectionNet:
        def __init__(self, *args, **kwargs):
            pass




def detect_levels_with_neural_network(hist, lookback=100, threshold=0.7):
    """
    Use the trained Causal-CNN + LSTM + MLP model to detect S/R levels.

    Falls back to scipy local-extrema detection if the trained model
    (level_detector.pth) is not available.
    """
    if not TORCH_AVAILABLE or len(hist) < lookback:
        return []

    try:
        opens   = hist['Open'].values[-lookback:].astype(np.float32)
        highs   = hist['High'].values[-lookback:].astype(np.float32)
        lows    = hist['Low'].values[-lookback:].astype(np.float32)
        closes  = hist['Close'].values[-lookback:].astype(np.float32)
        volumes = (hist['Volume'].values[-lookback:].astype(np.float32)
                   if 'Volume' in hist.columns else np.ones(lookback, dtype=np.float32))

        # Build OHLCV tensor and normalise per channel
        ohlcv = np.stack([opens, highs, lows, closes, volumes], axis=-1)
        ch_mean = ohlcv.mean(axis=0, keepdims=True)
        ch_std  = ohlcv.std(axis=0, keepdims=True) + 1e-9
        ohlcv_norm = (ohlcv - ch_mean) / ch_std

        input_tensor = torch.FloatTensor(ohlcv_norm).unsqueeze(0)  # [1, lookback, 5]

        # --- try loading the trained model ---
        model_path = 'level_detector.pth'
        try:
            model = LevelDetectionNet(lookback=lookback, in_channels=5)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs  = torch.sigmoid(logits)

                level_indices = (probs[0] > threshold).nonzero(as_tuple=False).flatten()

                # Collect candidate levels and merge nearby prices
                levels = []
                for idx in level_indices:
                    iv = idx.item()
                    if iv < len(closes):
                        price = float(closes[iv])
                        prob  = float(probs[0][iv])
                        levels.append({
                            'price': price,
                            'type': 'Neural Network',
                            'strength': prob,
                            'category': 'Neural-Network',
                            'breakoutProb': float(1 - prob),
                            'reversionProb': prob,
                            'touches': 1,
                        })

                # De-duplicate nearby prices (within 0.15 %)
                levels.sort(key=lambda l: l['price'])
                merged = []
                for lv in levels:
                    if merged and abs(lv['price'] - merged[-1]['price']) / merged[-1]['price'] < 0.0015:
                        if lv['strength'] > merged[-1]['strength']:
                            merged[-1] = lv
                    else:
                        merged.append(lv)

                return sorted(merged, key=lambda x: x['strength'], reverse=True)[:10]
            else:
                print(f"Neural network model not found at {model_path}. "
                      f"Use /api/train-level-detector to train it.")
        except Exception as model_error:
            print(f"Could not load neural network model: {model_error}, using fallback")

        # --- fallback: scipy local extrema (NOT the neural net - category
        # is kept as 'Neural-Network' so this still occupies its slot in the
        # ml_stat ensemble family, but 'type'/'usedFallback' are honest about
        # what actually ran, since this used to silently claim to be NN
        # output even when the model file was missing) ---
        print(f"⚠ Neural network fallback active: using scipy local-extrema, not {model_path}")
        from scipy.signal import argrelextrema
        high_indices = argrelextrema(highs, np.greater, order=5)[0]
        low_indices  = argrelextrema(lows,  np.less,    order=5)[0]

        levels = []
        for idx in high_indices:
            levels.append({
                'price': float(highs[idx]),
                'type': 'Local Extrema Fallback (High)',
                'strength': 0.65,
                'category': 'Neural-Network',
                'breakoutProb': 0.35,
                'reversionProb': 0.65,
                'touches': 1,
                'usedFallback': True,
            })
        for idx in low_indices:
            levels.append({
                'price': float(lows[idx]),
                'type': 'Local Extrema Fallback (Low)',
                'strength': 0.65,
                'category': 'Neural-Network',
                'breakoutProb': 0.35,
                'reversionProb': 0.65,
                'touches': 1,
                'usedFallback': True,
            })

        seen = set()
        unique = []
        for lv in levels:
            pk = round(lv['price'], 2)
            if pk not in seen:
                seen.add(pk)
                unique.append(lv)

        return sorted(unique, key=lambda x: x['strength'], reverse=True)[:10]

    except Exception as e:
        print(f"Neural Network level detection failed: {e}")
        return []


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

def score_levels_by_historical_outcome(levels, current_price, sigma_price):
    """
    NOT reinforcement learning (no policy/reward/environment) - previously
    misleadingly named validate_levels_with_rl. Filter levels using a
    composite score that prioritises OUTCOME-BASED signals.

    Priority order (highest → lowest weight):
      1. historical_reaction_score  — did price actually bounce here? (OUTCOME-BASED)
      2. historical_touch_count     — how many times has price visited this zone?
      3. historical_volume_at_zone  — was there elevated volume on those visits?
      4. independent_families       — how many distinct algorithm types agree?
      5. algorithm strength         — algorithm's own confidence (lowest weight, circular)

    NOTE: The pre-trained model (level_validator.pth) path is kept for future use,
    but the rule-based fallback is now non-circular — it uses historical outcome
    data that is computed BEFORE this function is called (in the pipeline).
    """
    if not levels:
        return levels

    try:
        validated_levels = []

        model = None
        if TORCH_AVAILABLE:
            try:
                model_path = 'level_validator.pth'
                if os.path.exists(model_path):
                    model = LevelValidator(n_features=10)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
            except Exception:
                model = None

        for level in levels:
            price      = level.get('price', current_price)
            strength   = level.get('strength', 0.5)
            category   = level.get('category', 'Unknown')
            distance_sigma = abs(price - current_price) / (sigma_price + 1e-9)

            # --- OUTCOME-BASED signals (non-circular) ---
            hist_score   = level.get('historical_reaction_score', None)
            touch_count  = level.get('historical_touch_count', 0)
            vol_at_zone  = level.get('historical_volume_at_zone', 1.0)
            freshness    = level.get('historical_freshness_bars', 9999)
            ind_families = level.get('independent_families', 1)

            has_history = hist_score is not None and touch_count > 0

            if model is not None:
                # If a trained model exists, use it (future path)
                features = np.array([
                    hist_score if hist_score is not None else 0.5,
                    min(touch_count / 10.0, 1.0),
                    min(vol_at_zone / 3.0, 1.0),
                    min(ind_families / 3.0, 1.0),
                    strength,
                    min(distance_sigma / 3.0, 1.0),
                    1.0 if category in ['Density (HDBSCAN)', 'HDBSCAN', 'ML-Confluence'] else 0.5,
                    min(freshness / 100.0, 1.0),
                    level.get('reversionProb', 0.5),
                    1.0 if distance_sigma < 2.0 else 0.5,
                ])
                with torch.no_grad():
                    ft = torch.FloatTensor(features).unsqueeze(0)
                    out = model(ft)
                    accept_score = float(torch.softmax(out, dim=1)[0][1].item())
            elif has_history:
                # Outcome-first scoring — historical evidence dominates
                reaction_component = hist_score  # 0-1, is this level proven?
                touch_component    = min(touch_count / 5.0, 1.0)  # caps at 5+ touches
                vol_component      = min((vol_at_zone - 1.0) / 2.0 + 0.5, 1.0)  # >1.0 vol = better
                freshness_bonus    = 1.0 if freshness < 20 else (0.8 if freshness < 60 else 0.6)
                family_bonus       = min(ind_families / 2.0, 1.0)  # 2 families = max

                accept_score = (
                    reaction_component * 0.45 +
                    touch_component    * 0.20 +
                    vol_component      * 0.15 +
                    freshness_bonus    * 0.10 +
                    family_bonus       * 0.10
                )
            else:
                # No historical data available — use conservative rule-based fallback
                # Bias towards keeping levels (lower threshold) since we have no evidence
                accept_score = (
                    strength          * 0.40 +
                    min(ind_families / 2.0, 1.0) * 0.30 +
                    level.get('reversionProb', 0.5) * 0.20 +
                    (1.0 if distance_sigma < 2.0 else 0.4) * 0.10
                )

            # Hard reject: levels with strong historical evidence of being NON-levels
            if has_history and hist_score < 0.30 and touch_count >= 3:
                # Price visited this zone 3+ times and broke through most of them
                level['rl_validation_score'] = float(accept_score)
                level['rl_rejected'] = True
                level['rejection_reason'] = f'Low reaction rate ({hist_score:.2f}) over {touch_count} touches'
                continue  # Skip this level

            threshold = 0.38 if has_history else 0.42
            if accept_score >= threshold:
                level['rl_validation_score'] = float(accept_score)
                validated_levels.append(level)

        print(f"  RL validator: {len(levels)} → {len(validated_levels)} levels "
              f"({'history-based' if any(l.get('historical_reaction_score') is not None for l in levels) else 'rule-based'})")
        return validated_levels

    except Exception as e:
        print(f"RL validation failed: {e}")
        return levels

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

    IMPORTANT - the real finding from this whole backtest program: a random
    price drawn uniformly from the recent trading range scores 0.4227
    support accuracy. Every price-clustering method tested (HDBSCAN, OPTICS,
    GMM, KDE, MeanShift, Isolation-Forest, Peak-Valley, Wavelet, HMM-based
    levels, and a trained neural network) landed in the 0.42-0.43 band -
    statistically indistinguishable from that random baseline AND from each
    other, once corrected for testing multiple methods (Bonferroni). The
    small differences between them (e.g. GMM 0.434 vs HDBSCAN 0.427) are
    noise, not real skill differences - don't read anything into them.

    Only four methods have shown REAL, statistically significant edge over
    the random baseline (p < 0.05/n after Bonferroni correction for every
    method tested this session): VWAP, TDA (persistent homology - fixed a
    bug where it output nonsense prices; the corrected version tests
    genuinely well), Session-Open, and Prior-Day-High/Low. All four are
    reference points real participants actually watch, not statistically-
    discovered price clusters - see backtest_reference_levels.py /
    backtest_reference_summary.csv and backtest_summary_tda.csv.

    Interaction and ML-Confluence are still NOT backed by any measured
    backtest at all - hand-set placeholders from before this program
    started, kept only so old callers don't break.
    """
    if category == 'VWAP' or source == 'VWAP':
        return 0.437  # Proven: p=0.0001 vs random baseline
    elif category == 'TDA' or source == 'TDA':
        return 0.437  # Proven: p=0.0010 vs random baseline (post bug-fix)
    elif category == 'Session-Open' or source == 'Session-Open':
        return 0.445  # Proven: p=0.0011 vs random baseline
    elif category == 'Prior-Day-HL' or source == 'Prior-Day-HL':
        return 0.436  # Proven: p=0.0060 vs random baseline
    elif category == 'Density (HDBSCAN)' or source == 'HDBSCAN' or category == 'HDBSCAN':
        return 0.427  # NOT proven vs random - see docstring
    elif category == 'GMM' or source == 'GMM':
        return 0.434  # NOT proven vs random - see docstring
    elif category == 'Isolation-Forest' or source == 'Isolation Forest':
        return 0.431  # NOT proven vs random - see docstring
    elif category == 'KDE' or source == 'KDE':
        return 0.433  # NOT proven vs random - see docstring
    elif category == 'MeanShift' or source == 'MeanShift':
        return 0.427  # NOT proven vs random - see docstring
    elif category == 'Neural-Network' or source == 'Neural Network':
        return 0.426  # NOT proven vs random - see docstring
    elif category == 'Interaction' or source == 'Local Density':
        return 0.55  # Not backtested at all - placeholder
    elif category == 'ML-Confluence':
        return 0.60  # Not backtested at all - placeholder
    else:
        return 0.4227  # Default: the measured random baseline itself

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





def calculate_historical_reaction_score(level_price, highs, lows, closes, volumes, sigma_price, recency_halflife=30):
    """
    Backtest a candidate level against historical price data.
    Measures: how often did price bounce vs break when it visited this zone?

    This is the PRIMARY quality signal - it answers whether the level has
    actually WORKED historically, not just whether algorithms detected it.

    Returns a dict with:
      reaction_rate     - fraction of touches that resulted in a bounce (0-1)
      touch_count       - number of times price entered the zone
      weighted_touches  - recency-weighted touch count
      freshness_bars    - how many bars since last touch (lower = fresher)
      volume_at_zone    - avg relative volume when price was in the zone
    """
    if len(closes) < 10 or sigma_price <= 0:
        return {'reaction_rate': 0.5, 'touch_count': 0, 'weighted_touches': 0.0,
                'freshness_bars': 9999, 'volume_at_zone': 1.0}

    zone_half_width = max(sigma_price * 0.4, level_price * 0.001)
    n = len(closes)
    avg_vol = float(np.mean(volumes)) if len(volumes) == n else 1.0

    bounces = 0.0
    breaks_ = 0.0
    touch_count = 0
    weighted_touches = 0.0
    freshness_bars = n  # default: never touched
    volume_at_zone_sum = 0.0

    in_zone = False
    for i in range(n - 5):
        price_in_zone = (lows[i] <= level_price + zone_half_width and
                         highs[i] >= level_price - zone_half_width)

        if price_in_zone and not in_zone:
            in_zone = True
            touch_count += 1
            recency_weight = np.exp(-((n - 1 - i) / (recency_halflife * 5)))
            weighted_touches += recency_weight
            if i < freshness_bars:
                freshness_bars = n - 1 - i
            vol_i = volumes[i] if len(volumes) > i else avg_vol
            volume_at_zone_sum += vol_i / (avg_vol + 1e-9)

            # Check outcome over next 3-8 bars
            look_forward = min(8, n - i - 1)
            if look_forward >= 3:
                future_highs = highs[i+1:i+1+look_forward]
                future_lows  = lows[i+1:i+1+look_forward]
                move_up   = float(np.max(future_highs)) - level_price
                move_down = level_price - float(np.min(future_lows))

                # Bounce: price moved >0.5 sigma away and reversed
                bounce_threshold = sigma_price * 0.5
                if closes[i] >= level_price:  # approaching from above → support test
                    if move_up > bounce_threshold and move_down < zone_half_width * 2:
                        bounces += recency_weight
                    elif move_down > bounce_threshold * 1.5:
                        breaks_ += recency_weight
                else:  # approaching from below → resistance test
                    if move_down > bounce_threshold and move_up < zone_half_width * 2:
                        bounces += recency_weight
                    elif move_up > bounce_threshold * 1.5:
                        breaks_ += recency_weight
        elif not price_in_zone:
            in_zone = False

    total = bounces + breaks_
    reaction_rate = float(bounces / total) if total > 0.01 else 0.5
    volume_at_zone = float(volume_at_zone_sum / touch_count) if touch_count > 0 else 1.0

    return {
        'reaction_rate': reaction_rate,
        'touch_count': touch_count,
        'weighted_touches': float(weighted_touches),
        'freshness_bars': int(freshness_bars),
        'volume_at_zone': volume_at_zone
    }


def get_ml_confluence_levels(all_algorithm_levels):
    """
    ML Confluence: Identifies levels where INDEPENDENT algorithm families agree.

    FIX: Previously counted total algorithms, which caused algorithm incest —
    HDBSCAN, OPTICS, KDE, multiscale HDBSCAN, and time-weighted HDBSCAN are all
    density estimators. 5 agreeing density algorithms = 1 signal, not 5 signals.

    Now groups algorithms into independent families and requires 2+ DISTINCT
    families to agree before calling something a confluence level.

    Families:
      density   → HDBSCAN, OPTICS, KDE, MeanShift, multiscale, time-weighted
      structure → Wyckoff, Gap, Pivot, Fibonacci
      ml_stat   → Neural Network, Persistent Homology, Isolation Forest
      extrema   → Peak-Valley
    """
    FAMILY_MAP = {
        'Density (HDBSCAN)': 'density', 'HDBSCAN': 'density',
        'OPTICS': 'density', 'Enhanced OPTICS': 'density',
        'KDE': 'density', 'Kernel Density': 'density',
        'MeanShift': 'density', 'Multiscale HDBSCAN': 'density',
        'Time-Weighted HDBSCAN': 'density', 'Agglomerative-Merged': 'density',
        'Wyckoff': 'structure', 'Gap': 'structure', 'Pivot': 'structure',
        'Fibonacci': 'structure', 'Classical': 'structure',
        'Neural-Network': 'ml_stat', 'Neural Network': 'ml_stat',
        'Persistent Homology': 'ml_stat', 'TDA': 'ml_stat',
        'Isolation-Forest': 'ml_stat', 'Isolation Forest': 'ml_stat',
        'Peak-Valley': 'extrema', 'PeakValley': 'extrema',
    }

    final_levels = []
    used = set()
    for level in sorted(all_algorithm_levels, key=lambda x: x['price']):
        price_key = round(level['price'], 4)
        if price_key in used:
            continue

        similar = [l for l in all_algorithm_levels
                   if abs(l['price'] - level['price']) / max(level['price'], 1e-9) < 0.012
                   and round(l['price'], 4) not in used]

        if len(similar) < 2:
            continue

        # Map each algorithm to its family
        families_seen = set()
        for l in similar:
            cat = l.get('category', l.get('source', 'Unknown'))
            family = FAMILY_MAP.get(cat, 'other')
            families_seen.add(family)

        independent_families = len(families_seen)

        # Require 2+ INDEPENDENT families — same-family agreement doesn't count
        if independent_families < 2:
            for l in similar:
                used.add(round(l['price'], 4))
            continue

        avg_price = float(np.mean([l['price'] for l in similar]))
        avg_strength = float(np.mean([l.get('strength', 0.5) for l in similar]))

        # Scale confluence boost by number of INDEPENDENT families (max 3)
        independence_multiplier = min(independent_families / 2.0, 1.5)
        confluence_strength = min(avg_strength * independence_multiplier, 0.95)

        sources = [l.get('source', l.get('category', 'Unknown')) for l in similar]
        primary_source = ('HDBSCAN' if any('HDBSCAN' in str(s) for s in sources)
                          else sources[0] if sources else 'Unknown')

        # Carry historical quality if already computed
        hist_scores = [l.get('historical_reaction_score') for l in similar
                       if l.get('historical_reaction_score') is not None]
        hist_score = float(np.mean(hist_scores)) if hist_scores else None

        entry = {
            'price': avg_price,
            'type': 'ML Confluence',
            'strength': confluence_strength,
            'algorithms': [l.get('category', 'Unknown') for l in similar],
            'families': list(families_seen),
            'independent_families': independent_families,
            'source': primary_source,
            'confluence_count': len(similar),
            'breakoutProb': float(1 - confluence_strength),
            'reversionProb': float(confluence_strength),
            'category': 'ML-Confluence',
        }
        if hist_score is not None:
            entry['historical_reaction_score'] = hist_score

        final_levels.append(entry)
        for l in similar:
            used.add(round(l['price'], 4))

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

        # ── MotiveWave priority check ──────────────────────────────────────
        mw_hist = get_motivewave_data(ticker, timeframe)
        if mw_hist is not None and not mw_hist.empty:
            print(f"✓ Using MotiveWave data for {ticker} {timeframe} ({len(mw_hist)} bars)")
            hist = mw_hist
        else:
            hist = None

        if hist is not None and not hist.empty:
            # Skip all yfinance fetching — jump straight to processing
            pass
        else:
            # Fall through to yfinance below
            hist = None

        stock = yf.Ticker(ticker) if hist is None else None
        
        # For futures, use alternative interval formats that yfinance accepts better
        is_futures = '=' in ticker
        
        # ── yfinance fetching (skipped if MotiveWave data already loaded) ────
        if hist is None:
          # Special handling for 1h timeframe - yfinance has issues with it for futures
          if is_futures and timeframe == '1h':
            interval = '60m'
            print(f"⚠ Futures 1h timeframe detected for {ticker}, using 60m interval")
          elif is_futures:
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '60m', '4h': '60m', '1d': '1d'}
            interval = interval_map.get(timeframe, '1d')
          else:
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
            interval = interval_map.get(timeframe, '1d')

        if hist is None:
            # ── yfinance fetch (only when no MotiveWave data available) ──────
            interval = locals().get('interval', '1d')
            if start_date and end_date:
                if is_futures and timeframe == '4h':
                    print(f"Fetching 4h data for {ticker} with date range (will resample from 1h/60m)...")
                    try:
                        hist = fetch_historical_data_with_resampling(
                            ticker=ticker, timeframe='4h',
                            start_date=start_date, end_date=end_date, is_futures=True
                        )
                    except Exception as e:
                        print(f"⚠ Resampling fetch failed: {e}")
                        hist = None
                else:
                    hist = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                if is_futures and timeframe in ['1m', '5m', '15m', '1h', '4h']:
                    period_map = {'1m': '5d', '5m': '5d', '15m': '7d', '1h': '7d', '4h': '10d', '1d': '2y'}
                else:
                    period_map = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '3mo', '4h': '3mo', '1d': '2y'}
                period = period_map.get(timeframe, '1y')

                if is_futures and timeframe == '4h':
                    print(f"Fetching 4h data for {ticker} (will resample from 1h/60m)...")
                    try:
                        hist = fetch_historical_data_with_resampling(
                            ticker=ticker, timeframe='4h', period=period, is_futures=True
                        )
                    except Exception as e:
                        print(f"⚠ Resampling fetch failed: {e}")
                        hist = None

                elif is_futures and timeframe == '1h':
                    attempts = [
                        ('60m', '5d'), ('60m', '3d'), ('60m', '2d'), ('60m', '1d'),
                        ('1h', '5d'), ('1h', '3d'), ('1h', '2d'), ('1h', '1d'),
                    ]
                    for attempt_interval, attempt_period in attempts:
                        try:
                            print(f"Trying {ticker} 1h: interval={attempt_interval}, period={attempt_period}")
                            hist = stock.history(period=attempt_period, interval=attempt_interval)
                            if hist is not None and len(hist) > 0:
                                print(f"✓ Fetched {len(hist)} bars for {ticker} 1h")
                                break
                        except Exception as e:
                            print(f"⚠ Attempt failed: {attempt_interval}/{attempt_period}: {str(e)[:80]}")
                            continue

                elif is_futures and timeframe in ['1m', '5m', '15m']:
                    attempts = [period, '5d', '3d', '2d', '1d'] if timeframe == '15m' else [period, '5d', '2d', '1d']
                    for attempt_period in attempts:
                        try:
                            hist = stock.history(period=attempt_period, interval=interval)
                            if hist is not None and len(hist) > 0:
                                print(f"✓ Fetched {len(hist)} bars for {ticker} {timeframe}")
                                break
                        except Exception as e:
                            if "pattern" not in str(e).lower() and "expected" not in str(e).lower():
                                print(f"⚠ Attempt failed {attempt_period}: {str(e)[:80]}")
                            continue

                elif timeframe == '4h':
                    print(f"Fetching 4h data for {ticker} (will resample from 1h)...")
                    try:
                        hist = fetch_historical_data_with_resampling(
                            ticker=ticker, timeframe='4h', period=period, is_futures=False
                        )
                    except Exception as e:
                        print(f"⚠ Resampling fetch failed: {e}")
                        hist = None

                else:
                    try:
                        hist = stock.history(period=period, interval=interval)
                    except Exception as e:
                        error_msg = str(e)
                        print(f"⚠ Error fetching {ticker} {timeframe}: {error_msg}")
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
                error_msg += '. yfinance futures data may be stale or unavailable. Use the MotiveWave Import panel to upload a CSV export from MotiveWave for this ticker/timeframe.'
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
        opens = hist['Open'].values if 'Open' in hist.columns else closes
        volumes = hist['Volume'].values
        current_price = closes[-1]

        print(f"Current: ${current_price:.2f} | Bars: {len(closes)}")

        if historical_mode:
            lookback_idx = int(len(closes) * 0.8)
            lookback_idx = min(max(lookback_idx, 50), len(closes))
            hist_closes = closes[:lookback_idx]
            hist_highs = highs[:lookback_idx]
            hist_lows = lows[:lookback_idx]
            hist_opens = opens[:lookback_idx]
            hist_volumes = volumes[:lookback_idx]
            hist_data_subset = hist.iloc[:lookback_idx]
        else:
            hist_closes = closes
            hist_highs = highs
            hist_lows = lows
            hist_opens = opens
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
        
        # NEW: Persistent Homology (TDA) - raw, filtered below alongside GMM
        persistent_homology_levels_result = []
        try:
            persistent_homology_levels_result = persistent_homology_levels(hist_highs, hist_lows, hist_closes, max_levels=8)
            print(f"Persistent Homology: Generated {len(persistent_homology_levels_result) if persistent_homology_levels_result else 0} raw levels")
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
        
        # SECONDARY: IsolationForest (event pivot candidates)
        isolation_forest_levels = find_pivot_anomalies(hist_highs, hist_lows, hist_closes)

        # GMM: soft/probabilistic clustering, ML-filtered below alongside TDA
        gmm_levels_result = calculate_gmm_levels(hist_highs, hist_lows, hist_closes)
        print(f"GMM: Generated {len(gmm_levels_result) if gmm_levels_result else 0} raw levels")

        # MeanShift: was previously dropped from the ML-filter category set
        # entirely (not just here - no live endpoint fed it in, before this
        # fix). v2's own validation docs list it as one of the 7 Bonferroni-
        # significant categories with one of the LARGER discrimination
        # improvements (+0.064) - real signal was being left on the table.
        # It's also still used separately as a validator against HDBSCAN
        # levels via enhance_levels_with_microstructure() below - that's a
        # different mechanism (post-hoc confidence boost), not mutually
        # exclusive with also scoring it here as its own candidate source.
        meanshift_levels_result = calculate_meanshift_levels(hist_highs, hist_lows, hist_closes)

        # ML FILTER v2: keep only levels the validated logistic regression
        # scores above threshold - all 7 categories walk-forward validated,
        # 6/7 at 100% fold consistency (see score_and_filter_levels_v2
        # docstring/module comment). Raw/unfiltered candidates are discarded,
        # not exposed.
        try:
            levels_by_category_v2 = {
                'GMM': gmm_levels_result, 'TDA': persistent_homology_levels_result,
                'HDBSCAN': hdbscan_levels, 'OPTICS': enhanced_optics_levels_result,
                'KDE': kde_levels_result, 'Isolation-Forest': isolation_forest_levels,
                'MeanShift': meanshift_levels_result,
            }
            levels_by_category_v2 = extend_thin_side_levels(
                ticker, timeframe, hist_highs, hist_lows, hist_closes, current_price,
                levels_by_category_v2, is_futures,
            )
            filtered = score_and_filter_levels_v2(
                levels_by_category_v2,
                hist_highs, hist_lows, hist_opens, hist_closes, hist_volumes, current_price,
                timestamps=hist_data_subset.index.values,
            )
            gmm_levels_result = [l for l in filtered if l.get('category') == 'GMM']
            persistent_homology_levels_result = [l for l in filtered if l.get('category') == 'TDA']
            hdbscan_levels = [l for l in filtered if l.get('category') in ('HDBSCAN', 'Density (HDBSCAN)')]
            enhanced_optics_levels_result = [l for l in filtered if l.get('category') == 'OPTICS']
            kde_levels_result = [l for l in filtered if l.get('category') == 'KDE']
            isolation_forest_levels = [l for l in filtered if l.get('category') == 'Isolation-Forest']
            meanshift_levels_result = [l for l in filtered if l.get('category') == 'MeanShift']
            print(f"ML filter v2: kept {len(gmm_levels_result)} GMM, {len(persistent_homology_levels_result)} TDA, "
                  f"{len(hdbscan_levels)} HDBSCAN, {len(enhanced_optics_levels_result)} OPTICS, "
                  f"{len(kde_levels_result)} KDE, {len(isolation_forest_levels)} Isolation-Forest, "
                  f"{len(meanshift_levels_result)} MeanShift levels")
        except Exception as e:
            print(f"ML filter failed, falling back to unfiltered GMM/TDA: {e}")

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

        # Fibonacci for metadata enrichment only (not primary levels)
        fib_levels = calculate_fibonacci_levels(hist_highs, hist_lows)

        # ---- HARD GUARD: ensure all level outputs are lists ----
        hdbscan_levels = hdbscan_levels or []
        enhanced_optics_levels_result = enhanced_optics_levels_result or []
        kde_levels_result = kde_levels_result or []
        multiscale_hdbscan_levels_result = multiscale_hdbscan_levels_result or []
        time_weighted_levels_result = time_weighted_levels_result or []
        wyckoff_levels_result = wyckoff_levels_result or []
        persistent_homology_levels_result = persistent_homology_levels_result or []
        neural_network_levels_result = neural_network_levels_result or []
        isolation_forest_levels = isolation_forest_levels or []
        gmm_levels_result = gmm_levels_result or []
        meanshift_levels_result = meanshift_levels_result or []
        fib_levels = fib_levels or []

        # ML LEVELS: Primary discovery algorithms only
        all_ml_levels = (hdbscan_levels + enhanced_optics_levels_result + kde_levels_result +
                        multiscale_hdbscan_levels_result + time_weighted_levels_result +
                        wyckoff_levels_result + persistent_homology_levels_result +
                        neural_network_levels_result + isolation_forest_levels + gmm_levels_result +
                        meanshift_levels_result)
        
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

        # ── HISTORICAL REACTION SCORING ────────────────────────────────────────
        # For each candidate level, backtest against real price history:
        # "When price visited this zone before, did it bounce or break?"
        # This is the only truly non-circular quality signal — it uses OUTCOMES,
        # not the algorithm's own confidence score.
        # Adds: historical_reaction_score, historical_touch_count,
        #       historical_volume_at_zone, historical_freshness_bars to each level.
        try:
            _vol_arr = np.array(hist_volumes.values, dtype=float) if hasattr(hist_volumes, 'values') else np.array(hist_volumes, dtype=float)
            _hi_arr  = np.array(hist_highs,  dtype=float)
            _lo_arr  = np.array(hist_lows,   dtype=float)
            _cl_arr  = np.array(hist_closes, dtype=float)
            for _lvl in all_ml_levels:
                _score = calculate_historical_reaction_score(
                    _lvl['price'], _hi_arr, _lo_arr, _cl_arr, _vol_arr, sigma_price
                )
                _lvl['historical_reaction_score']  = _score['reaction_rate']
                _lvl['historical_touch_count']     = _score['touch_count']
                _lvl['historical_volume_at_zone']  = _score['volume_at_zone']
                _lvl['historical_freshness_bars']  = _score['freshness_bars']
            proven = sum(1 for l in all_ml_levels if l.get('historical_touch_count', 0) >= 2)
            print(f"✓ Historical reaction scored {len(all_ml_levels)} levels — {proven} have 2+ historical touches")
        except Exception as _e:
            print(f"⚠ Historical reaction scoring failed: {_e}")
        # ──────────────────────────────────────────────────────────────────────

        confluence_levels = get_ml_confluence_levels(all_ml_levels)
        confluence_levels = confluence_levels or []

        # Combine ML levels (no gap/pivot/peak-valley/VbP — pure ML discovery)
        all_levels_combined = confluence_levels + all_ml_levels

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
                all_levels_combined = score_levels_by_historical_outcome(all_levels_combined, current_price, sigma_price)
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
        
        # DEBUG: Log new level counts
        print(f"🔍 NEW LEVEL DETECTION METHODS:")
        print(f"   OPTICS: {len(enhanced_optics_ml)} levels")
        print(f"   KDE: {len(kde_ml)} levels")
        print(f"   Multi-Scale HDBSCAN: {len(multiscale_hdbscan_ml)} levels")
        print(f"   Time-Weighted HDBSCAN: {len(time_weighted_ml)} levels")
        print(f"   Wyckoff: {len(wyckoff_ml)} levels")
        print(f"   Persistent Homology (TDA): {len(persistent_homology_ml)} levels")
        print(f"   Neural Network: {len(neural_network_ml)} levels")
        if len(neural_network_ml) > 0:
            print(f"   ✓ Neural Network levels found: {[l.get('price') for l in neural_network_ml[:3]]}")
        
        # Combine all structural density-based levels
        hdbscan_ml = hdbscan_ml + enhanced_optics_ml + kde_ml + multiscale_hdbscan_ml + time_weighted_ml + wyckoff_ml + persistent_homology_ml + neural_network_ml
        
        isolation_forest_ml = [l for l in all_levels_combined if l['category'] == 'Isolation-Forest']

        # DEBUG: Log level counts before building response
        print(f"Level organization - HDBSCAN: {len(hdbscan_ml)}, Confluence: {len(ml_confluence)}, Event: {len(isolation_forest_ml)}, Interaction: {len(local_interaction_levels)}")
        
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
            'structural': hdbscan_ml,

            # EVENT / PIVOT LEVELS (behavioral, fast-decay)
            'event': isolation_forest_ml,

            # INTERACTION LEVELS (local density, near price, short memory)
            'interaction': local_interaction_levels,

            # Backward compatibility: empty lists for removed algorithm types
            'mlConfluence': ml_confluence,
            'peakValley': [],
            'fallback': [],
            'classicalStructural': {'pivots': [], 'gaps': []},
            'meanshift': [],
            'dbscan': [],
            'gmm': [],
            'kmeans': [],
            'volatility': [],
            'pivots': [],
            'fibonacci': [],
            'gaps': []
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
            'mostProbablePath': most_probable_path,
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

# NEW ENDPOINT: 3D PHASE SPACE DATA

# NEW ENDPOINT: DETAILED GARCH ANALYSIS

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
# EQUATION ARCHITECTURE: HARD PHYSICS vs LEARNABLE
# ============================================================================
#
# HARD PHYSICS (never learned, always fixed):
# - GARCH volatility estimation (fit_garch_model, calculate_garch_volatility_regime)
#   Reason: Volatility is a fundamental market property, not a calibration parameter
# - State machine detection (detect_market_microstructure_state)
#   Reason: State classification is structural, calibration happens via multipliers
# - Liquidity stress scoring (liquidity_stress_score)
#   Reason: LSS is an observation, not a tunable parameter
# - OI wall computation (compute_oi_walls)
#   Reason: Wall positions are market data, not learnable
#
# LEARNABLE (can be calibrated per regime):
# - Tail usage multiplier (tail_usage_multiplier_from_lss → adjust_hod_lod_usage)
#   Calibration: override_tail_mult in hodlod_calibration table
#   Bounds: [0.70, 1.80] (hard constraint)
# - OI clipping intensity (apply_oi_walls_to_hod_lod)
#   Calibration: override_oi_clip_mult in hodlod_calibration table
#   Bounds: [0.10, 1.50] (hard constraint)
# - RF adjustment strength (rf_adjust_hod_lod)
#   Calibration: rf_clip (future: can clip RF adjustments)
#   Bounds: [0.50, 2.50] (hard constraint)
#
# STRUCTURAL CONSTRAINTS (enforced, never learned):
# - HOD > LOD (always enforced)
# - OI walls clip, not predict (structural boundaries)
# - RF applied after structural constraints (ordering preserved)
# - Residual correction applied last (learns remaining bias)
#
# ============================================================================
# SELF-LEARNING ML FRAMEWORK - HOD/LOD PREDICTION
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
# VOLATILITY ESTIMATORS - ENHANCED
# ============================================================================


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


# GARMAN-KLASS VOLATILITY ESTIMATOR (backward compatibility)

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
    
    # 1. Calculate DAILY (non-annualized) volatility using Garman-Klass
    # Ensure all values are positive and valid
    opens = np.maximum(opens, 1e-9)
    highs = np.maximum(highs, opens * 0.99)
    lows = np.maximum(lows, opens * 0.99)
    closes = np.maximum(closes, lows)
    
    log_hl = np.log(highs / (lows + 1e-9))
    log_co = np.log(closes / (opens + 1e-9))
    variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    # Handle negative variance (can happen if close-open correlation is high)
    variance = np.maximum(variance, 1e-9)  # Ensure non-negative
    
    # Session volatility (for next period)
    mean_variance = np.mean(variance)
    if mean_variance <= 0 or not np.isfinite(mean_variance):
        # Fallback: use simple close-to-close volatility
        returns = np.diff(np.log(closes))
        mean_variance = np.var(returns)
        if mean_variance <= 0 or not np.isfinite(mean_variance):
            # Last resort: use price range
            price_range = np.max(highs) - np.min(lows)
            mean_variance = (price_range / current_price) ** 2 / len(closes)
    
    sigma_session = np.sqrt(mean_variance)  # Decimal (e.g., 0.015 = 1.5%)
    
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
        'sigma_session': float(sigma_session),  # Next period vol (decimal)
        'sigma_session_pct': float(sigma_session * 100),  # Next period vol (%)
        'sigma_annual_pct': float(sigma_annual * 100),  # Annualized (%)
        'sigma_price': float(sigma_price),  # Expected $ range
        'method': 'garman_klass_session'
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


# ============================================================================
# SIMPLIFIED API FOR YOUR EXISTING CODE
# ============================================================================


# ============================================================================
# STATE MACHINE ENHANCEMENTS - FOR IMPROVED HOD/LOD PREDICTIONS
# ============================================================================








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
# OU ZONE MODEL (standalone - independently validated, NOT the level detector)
# ============================================================================
#
# Zone construction: fit an Ornstein-Uhlenbeck process to price-minus-VWAP
# (a mean-reverting quantity by construction), zone = VWAP + mu +/- k*std.
# Validated on the full 12-year NQ/ES 1H/4H history (see conversation/
# backtest_ou_zones.py): ~61-64% reject rate at the zone edge (vs 50%
# random for a symmetric edge, z=5.7-14.6, highly significant), stable
# across all four 12-year time-quartiles, and the rejection rate holds up
# whether the PRIOR zone rejected or broke (a genuine recurring regime
# read, not a one-shot fit). Adding Hurst/HMM/GJR-GARCH/EGARCH/
# Garman-Klass as extra context features was tested and showed NO
# improvement (near-zero/insignificant discrimination, walk-forward
# validated) - so this stays deliberately simple: two inputs (price,
# VWAP), no other model.
#
# Separately: only ~19-21% of touches are a genuinely DURABLE rejection
# (a real move that doesn't later get run over) - the flat reject rate
# overstates how much a single touch can be trusted to hold.

_OU_ZONE_K = 0.25  # zone half-width in stationary-std units, tuned during validation
_OU_ZONE_BACKTEST_SUMMARY = {
    'note': 'Independently validated on 12yr NQ/ES 1H/4H history - not the level detector.',
    'flat_reject_rate': {'ES_1h': 0.641, 'ES_4h': 0.633, 'NQ_1h': 0.639, 'NQ_4h': 0.615},
    'touch_rate': {'ES_1h': 0.632, 'ES_4h': 0.610, 'NQ_1h': 0.632, 'NQ_4h': 0.619},
    'strong_durable_reject_rate': {'ES_1h': 0.208, 'ES_4h': 0.187, 'NQ_1h': 0.209, 'NQ_4h': 0.209},
    'median_zone_width_atr': {'ES_1h': 1.17, 'ES_4h': 1.22, 'NQ_1h': 1.26, 'NQ_4h': 1.27},
    # Target pipeline (k=0.25, corridor_buffer_atr=0.3) - of rejected zones
    # that had a real KDE target on the path to fair value: does price ever
    # reach it (target_reach_rate), and separately, does it get there
    # before a 1-ATR adverse move against a position entered at the edge
    # (clean_reach_rate) - i.e. would a real trade with a 1-ATR stop have
    # survived to see the target. From backtest_zone_target.py.
    'target_availability': {'ES_1h': 0.313, 'ES_4h': 0.371, 'NQ_1h': 0.322, 'NQ_4h': 0.358},
    'target_reach_rate': {'ES_1h': 0.957, 'ES_4h': 0.934, 'NQ_1h': 0.948, 'NQ_4h': 0.952},
    'target_clean_reach_rate': {'ES_1h': 0.851, 'ES_4h': 0.771, 'NQ_1h': 0.838, 'NQ_4h': 0.822},
}

# ============================================================================
# LEVEL-DETECTION ALGORITHM BACKTEST (per-algorithm, NOT the OU zone model
# above - this validates the actual structural level detectors: HDBSCAN,
# GMM, TDA, OPTICS, KDE, MeanShift, Multiscale-HDBSCAN, Isolation-Forest)
# ============================================================================
# Walk-forward (70/30 train/holdout split), pooled across NQ/ES 1H/4H,
# ~175k candidate-touch events - see backtest_ml_filter_v3_events.py.
# "unconditional" = raw accuracy of that algorithm's candidates with no
# filtering; "filtered" = accuracy of the top-half by the validated ML
# filter's predicted P(bounce); "lift" is the improvement filtering adds.
# All except KDE clear Bonferroni correction (alpha=0.05/8=0.00625).
# Regenerated after today's system review: OPTICS now reflects
# enhanced_optics_levels (the function actually live in production - the
# original version of this backtest referenced optics_multi_density_levels,
# which had zero live callers and was removed as dead code), and
# Multiscale-HDBSCAN was added since it's part of the current live
# algorithm set. Wyckoff and time-weighted-HDBSCAN aren't included yet -
# different call signature (DataFrame / timestamps array) this backtest
# script doesn't support without separate handling.
_LEVEL_ALGO_BACKTEST_SUMMARY = {
    'note': 'Walk-forward holdout, ~175k candidate-touch events - same trained filter, broken out per instrument/timeframe rather than pooled. VWAP excluded (not a discovery algorithm, showed no/negative lift). realized_rr = avg max-favorable-excursion / avg max-adverse-excursion within the reaction window (ATR units) - the ACTUAL price behavior, notably closer to 1:1 than the fixed 2:1 target/stop the accuracy labels are defined against.',
    'random_baseline': 0.4227,
    'by_instrument': {
        'NQ_1h': {
            'TDA': {'unconditional': 0.4278, 'filtered': 0.4798, 'lift': 0.052, 'z': 4.27, 'n_holdout': 1436, 'avg_mfe_atr': 3.043, 'avg_mae_atr': 2.797, 'realized_rr': 1.088},
            'Multiscale-HDBSCAN': {'unconditional': 0.4267, 'filtered': 0.4592, 'lift': 0.0326, 'z': 3.09, 'n_holdout': 1864, 'avg_mfe_atr': 2.993, 'avg_mae_atr': 2.871, 'realized_rr': 1.042},
            'GMM': {'unconditional': 0.4288, 'filtered': 0.4573, 'lift': 0.0285, 'z': 2.09, 'n_holdout': 914, 'avg_mfe_atr': 3.026, 'avg_mae_atr': 2.871, 'realized_rr': 1.054},
            'MeanShift': {'unconditional': 0.4229, 'filtered': 0.4533, 'lift': 0.0305, 'z': 2.05, 'n_holdout': 1136, 'avg_mfe_atr': 3.05, 'avg_mae_atr': 2.91, 'realized_rr': 1.048},
            'OPTICS': {'unconditional': 0.4112, 'filtered': 0.4522, 'lift': 0.0409, 'z': 2.24, 'n_holdout': 1484, 'avg_mfe_atr': 2.885, 'avg_mae_atr': 2.858, 'realized_rr': 1.009},
            'HDBSCAN': {'unconditional': 0.4185, 'filtered': 0.4471, 'lift': 0.0286, 'z': 2.02, 'n_holdout': 1769, 'avg_mfe_atr': 2.983, 'avg_mae_atr': 2.883, 'realized_rr': 1.035},
            'KDE': {'unconditional': 0.4155, 'filtered': 0.4467, 'lift': 0.0312, 'z': 1.13, 'n_holdout': 553, 'avg_mfe_atr': 3.157, 'avg_mae_atr': 2.91, 'realized_rr': 1.085},
            'Isolation-Forest': {'unconditional': 0.4038, 'filtered': 0.4228, 'lift': 0.019, 'z': 0.01, 'n_holdout': 745, 'avg_mfe_atr': 2.932, 'avg_mae_atr': 2.966, 'realized_rr': 0.988},
        },
        'NQ_4h': {
            'OPTICS': {'unconditional': 0.4738, 'filtered': 0.5967, 'lift': 0.123, 'z': 6.7, 'n_holdout': 367, 'avg_mfe_atr': 2.956, 'avg_mae_atr': 1.93, 'realized_rr': 1.531},
            'TDA': {'unconditional': 0.4269, 'filtered': 0.5471, 'lift': 0.1202, 'z': 4.61, 'n_holdout': 340, 'avg_mfe_atr': 2.872, 'avg_mae_atr': 2.095, 'realized_rr': 1.371},
            'GMM': {'unconditional': 0.4805, 'filtered': 0.5426, 'lift': 0.0621, 'z': 3.61, 'n_holdout': 223, 'avg_mfe_atr': 2.986, 'avg_mae_atr': 2.14, 'realized_rr': 1.396},
            'Isolation-Forest': {'unconditional': 0.4532, 'filtered': 0.5381, 'lift': 0.085, 'z': 3.57, 'n_holdout': 236, 'avg_mfe_atr': 2.693, 'avg_mae_atr': 2.277, 'realized_rr': 1.183},
            'KDE': {'unconditional': 0.4535, 'filtered': 0.5227, 'lift': 0.0692, 'z': 2.32, 'n_holdout': 132, 'avg_mfe_atr': 2.867, 'avg_mae_atr': 2.124, 'realized_rr': 1.349},
            'Multiscale-HDBSCAN': {'unconditional': 0.4462, 'filtered': 0.518, 'lift': 0.0717, 'z': 4.16, 'n_holdout': 473, 'avg_mfe_atr': 2.793, 'avg_mae_atr': 2.199, 'realized_rr': 1.27},
            'MeanShift': {'unconditional': 0.458, 'filtered': 0.5034, 'lift': 0.0454, 'z': 2.78, 'n_holdout': 292, 'avg_mfe_atr': 2.797, 'avg_mae_atr': 2.231, 'realized_rr': 1.253},
            'HDBSCAN': {'unconditional': 0.4272, 'filtered': 0.5, 'lift': 0.0728, 'z': 3.36, 'n_holdout': 470, 'avg_mfe_atr': 2.749, 'avg_mae_atr': 2.23, 'realized_rr': 1.233},
        },
        'ES_1h': {
            'Isolation-Forest': {'unconditional': 0.4199, 'filtered': 0.4872, 'lift': 0.0673, 'z': 3.51, 'n_holdout': 741, 'avg_mfe_atr': 3.156, 'avg_mae_atr': 2.934, 'realized_rr': 1.076},
            'TDA': {'unconditional': 0.4365, 'filtered': 0.4859, 'lift': 0.0494, 'z': 4.75, 'n_holdout': 1451, 'avg_mfe_atr': 2.994, 'avg_mae_atr': 2.781, 'realized_rr': 1.076},
            'Multiscale-HDBSCAN': {'unconditional': 0.4307, 'filtered': 0.4762, 'lift': 0.0455, 'z': 4.54, 'n_holdout': 1873, 'avg_mfe_atr': 2.975, 'avg_mae_atr': 2.741, 'realized_rr': 1.085},
            'OPTICS': {'unconditional': 0.4036, 'filtered': 0.4695, 'lift': 0.0659, 'z': 3.68, 'n_holdout': 1591, 'avg_mfe_atr': 3.036, 'avg_mae_atr': 2.987, 'realized_rr': 1.016},
            'MeanShift': {'unconditional': 0.4265, 'filtered': 0.4654, 'lift': 0.0389, 'z': 2.92, 'n_holdout': 1186, 'avg_mfe_atr': 2.957, 'avg_mae_atr': 2.677, 'realized_rr': 1.104},
            'HDBSCAN': {'unconditional': 0.4249, 'filtered': 0.4602, 'lift': 0.0353, 'z': 3.16, 'n_holdout': 1836, 'avg_mfe_atr': 2.947, 'avg_mae_atr': 2.856, 'realized_rr': 1.032},
            'KDE': {'unconditional': 0.4221, 'filtered': 0.4547, 'lift': 0.0326, 'z': 1.54, 'n_holdout': 574, 'avg_mfe_atr': 3.0, 'avg_mae_atr': 2.813, 'realized_rr': 1.067},
            'GMM': {'unconditional': 0.4227, 'filtered': 0.4487, 'lift': 0.026, 'z': 1.55, 'n_holdout': 896, 'avg_mfe_atr': 2.927, 'avg_mae_atr': 2.879, 'realized_rr': 1.017},
        },
        'ES_4h': {
            'Multiscale-HDBSCAN': {'unconditional': 0.4441, 'filtered': 0.5279, 'lift': 0.0838, 'z': 4.56, 'n_holdout': 466, 'avg_mfe_atr': 2.928, 'avg_mae_atr': 2.398, 'realized_rr': 1.221},
            'HDBSCAN': {'unconditional': 0.4416, 'filtered': 0.5147, 'lift': 0.0731, 'z': 3.89, 'n_holdout': 443, 'avg_mfe_atr': 2.912, 'avg_mae_atr': 2.403, 'realized_rr': 1.212},
            'TDA': {'unconditional': 0.445, 'filtered': 0.4933, 'lift': 0.0482, 'z': 2.73, 'n_holdout': 371, 'avg_mfe_atr': 2.887, 'avg_mae_atr': 2.188, 'realized_rr': 1.32},
            'MeanShift': {'unconditional': 0.4248, 'filtered': 0.4801, 'lift': 0.0554, 'z': 1.93, 'n_holdout': 277, 'avg_mfe_atr': 3.002, 'avg_mae_atr': 2.274, 'realized_rr': 1.32},
            'Isolation-Forest': {'unconditional': 0.3972, 'filtered': 0.4663, 'lift': 0.0691, 'z': 1.27, 'n_holdout': 208, 'avg_mfe_atr': 2.979, 'avg_mae_atr': 2.539, 'realized_rr': 1.173},
            'OPTICS': {'unconditional': 0.453, 'filtered': 0.4659, 'lift': 0.013, 'z': 1.67, 'n_holdout': 367, 'avg_mfe_atr': 2.78, 'avg_mae_atr': 2.466, 'realized_rr': 1.127},
            'GMM': {'unconditional': 0.4233, 'filtered': 0.4413, 'lift': 0.018, 'z': 0.55, 'n_holdout': 213, 'avg_mfe_atr': 3.016, 'avg_mae_atr': 2.49, 'realized_rr': 1.211},
            'KDE': {'unconditional': 0.4022, 'filtered': 0.4074, 'lift': 0.0052, 'z': -0.36, 'n_holdout': 135, 'avg_mfe_atr': 3.0, 'avg_mae_atr': 2.379, 'realized_rr': 1.261},
        },
        'GC_1h': {
            'OPTICS': {'unconditional': 0.4745, 'filtered': 0.5258, 'lift': 0.0513, 'z': 6.0, 'n_holdout': 854, 'avg_mfe_atr': 2.23, 'avg_mae_atr': 1.821, 'realized_rr': 1.225},
            'Multiscale-HDBSCAN': {'unconditional': 0.462, 'filtered': 0.5151, 'lift': 0.053, 'z': 5.98, 'n_holdout': 1062, 'avg_mfe_atr': 2.401, 'avg_mae_atr': 2.103, 'realized_rr': 1.141},
            'TDA': {'unconditional': 0.4612, 'filtered': 0.5135, 'lift': 0.0523, 'z': 5.17, 'n_holdout': 814, 'avg_mfe_atr': 2.388, 'avg_mae_atr': 2.126, 'realized_rr': 1.123},
            'MeanShift': {'unconditional': 0.4539, 'filtered': 0.5125, 'lift': 0.0586, 'z': 4.4, 'n_holdout': 599, 'avg_mfe_atr': 2.399, 'avg_mae_atr': 1.97, 'realized_rr': 1.218},
            'HDBSCAN': {'unconditional': 0.4612, 'filtered': 0.5094, 'lift': 0.0482, 'z': 5.49, 'n_holdout': 1013, 'avg_mfe_atr': 2.292, 'avg_mae_atr': 2.121, 'realized_rr': 1.081},
            'Isolation-Forest': {'unconditional': 0.4344, 'filtered': 0.498, 'lift': 0.0635, 'z': 3.34, 'n_holdout': 490, 'avg_mfe_atr': 2.392, 'avg_mae_atr': 2.069, 'realized_rr': 1.156},
            'GMM': {'unconditional': 0.442, 'filtered': 0.4653, 'lift': 0.0233, 'z': 1.86, 'n_holdout': 475, 'avg_mfe_atr': 2.337, 'avg_mae_atr': 2.181, 'realized_rr': 1.071},
            'KDE': {'unconditional': 0.4371, 'filtered': 0.4563, 'lift': 0.0192, 'z': 1.19, 'n_holdout': 309, 'avg_mfe_atr': 2.177, 'avg_mae_atr': 2.241, 'realized_rr': 0.971},
        },
        'GC_4h': {
            'Multiscale-HDBSCAN': {'unconditional': 0.4344, 'filtered': 0.4801, 'lift': 0.0458, 'z': 2.09, 'n_holdout': 327, 'avg_mfe_atr': 2.454, 'avg_mae_atr': 2.004, 'realized_rr': 1.224},
            'HDBSCAN': {'unconditional': 0.4364, 'filtered': 0.4795, 'lift': 0.043, 'z': 1.95, 'n_holdout': 292, 'avg_mfe_atr': 2.528, 'avg_mae_atr': 2.018, 'realized_rr': 1.252},
            'TDA': {'unconditional': 0.4167, 'filtered': 0.4755, 'lift': 0.0588, 'z': 1.73, 'n_holdout': 265, 'avg_mfe_atr': 2.368, 'avg_mae_atr': 2.224, 'realized_rr': 1.065},
            'KDE': {'unconditional': 0.4526, 'filtered': 0.475, 'lift': 0.0224, 'z': 0.95, 'n_holdout': 80, 'avg_mfe_atr': 2.525, 'avg_mae_atr': 1.964, 'realized_rr': 1.286},
            'OPTICS': {'unconditional': 0.4262, 'filtered': 0.4568, 'lift': 0.0306, 'z': 1.07, 'n_holdout': 243, 'avg_mfe_atr': 2.329, 'avg_mae_atr': 1.966, 'realized_rr': 1.185},
            'MeanShift': {'unconditional': 0.412, 'filtered': 0.4343, 'lift': 0.0223, 'z': 0.31, 'n_holdout': 175, 'avg_mfe_atr': 2.266, 'avg_mae_atr': 2.222, 'realized_rr': 1.02},
            'Isolation-Forest': {'unconditional': 0.4023, 'filtered': 0.432, 'lift': 0.0297, 'z': 0.24, 'n_holdout': 169, 'avg_mfe_atr': 2.371, 'avg_mae_atr': 2.126, 'realized_rr': 1.116},
            'GMM': {'unconditional': 0.4404, 'filtered': 0.4275, 'lift': -0.0128, 'z': 0.11, 'n_holdout': 138, 'avg_mfe_atr': 2.296, 'avg_mae_atr': 2.128, 'realized_rr': 1.079},
        },
        'SI_1h': {
            'TDA': {'unconditional': 0.4488, 'filtered': 0.5247, 'lift': 0.0759, 'z': 5.35, 'n_holdout': 688, 'avg_mfe_atr': 2.383, 'avg_mae_atr': 2.038, 'realized_rr': 1.169},
            'OPTICS': {'unconditional': 0.4725, 'filtered': 0.5201, 'lift': 0.0476, 'z': 5.4, 'n_holdout': 773, 'avg_mfe_atr': 2.503, 'avg_mae_atr': 2.06, 'realized_rr': 1.215},
            'Isolation-Forest': {'unconditional': 0.4647, 'filtered': 0.5083, 'lift': 0.0437, 'z': 3.53, 'n_holdout': 421, 'avg_mfe_atr': 2.419, 'avg_mae_atr': 2.056, 'realized_rr': 1.176},
            'Multiscale-HDBSCAN': {'unconditional': 0.4507, 'filtered': 0.4819, 'lift': 0.0312, 'z': 3.51, 'n_holdout': 884, 'avg_mfe_atr': 2.441, 'avg_mae_atr': 2.012, 'realized_rr': 1.213},
            'HDBSCAN': {'unconditional': 0.4502, 'filtered': 0.4814, 'lift': 0.0312, 'z': 3.48, 'n_holdout': 887, 'avg_mfe_atr': 2.487, 'avg_mae_atr': 2.138, 'realized_rr': 1.163},
            'KDE': {'unconditional': 0.4366, 'filtered': 0.476, 'lift': 0.0394, 'z': 1.7, 'n_holdout': 250, 'avg_mfe_atr': 2.261, 'avg_mae_atr': 2.041, 'realized_rr': 1.108},
            'MeanShift': {'unconditional': 0.4112, 'filtered': 0.4496, 'lift': 0.0384, 'z': 1.27, 'n_holdout': 556, 'avg_mfe_atr': 2.29, 'avg_mae_atr': 2.04, 'realized_rr': 1.123},
            'GMM': {'unconditional': 0.4221, 'filtered': 0.4334, 'lift': 0.0114, 'z': 0.42, 'n_holdout': 383, 'avg_mfe_atr': 2.237, 'avg_mae_atr': 2.096, 'realized_rr': 1.067},
        },
        'SI_4h': {
            'GMM': {'unconditional': 0.4811, 'filtered': 0.5465, 'lift': 0.0654, 'z': 2.32, 'n_holdout': 86, 'avg_mfe_atr': 2.593, 'avg_mae_atr': 2.354, 'realized_rr': 1.102},
            'OPTICS': {'unconditional': 0.4778, 'filtered': 0.4855, 'lift': 0.0077, 'z': 1.96, 'n_holdout': 241, 'avg_mfe_atr': 2.395, 'avg_mae_atr': 2.391, 'realized_rr': 1.002},
            'KDE': {'unconditional': 0.3904, 'filtered': 0.4853, 'lift': 0.0949, 'z': 1.04, 'n_holdout': 68, 'avg_mfe_atr': 2.329, 'avg_mae_atr': 2.263, 'realized_rr': 1.029},
            'MeanShift': {'unconditional': 0.4158, 'filtered': 0.4754, 'lift': 0.0596, 'z': 1.18, 'n_holdout': 122, 'avg_mfe_atr': 2.309, 'avg_mae_atr': 2.341, 'realized_rr': 0.986},
            'Isolation-Forest': {'unconditional': 0.3868, 'filtered': 0.437, 'lift': 0.0502, 'z': 0.31, 'n_holdout': 119, 'avg_mfe_atr': 2.234, 'avg_mae_atr': 2.257, 'realized_rr': 0.99},
            'HDBSCAN': {'unconditional': 0.4134, 'filtered': 0.4318, 'lift': 0.0184, 'z': 0.27, 'n_holdout': 220, 'avg_mfe_atr': 2.366, 'avg_mae_atr': 2.166, 'realized_rr': 1.093},
            'Multiscale-HDBSCAN': {'unconditional': 0.405, 'filtered': 0.4198, 'lift': 0.0149, 'z': -0.08, 'n_holdout': 212, 'avg_mfe_atr': 2.393, 'avg_mae_atr': 2.433, 'realized_rr': 0.983},
            'TDA': {'unconditional': 0.3842, 'filtered': 0.3949, 'lift': 0.0107, 'z': -0.7, 'n_holdout': 157, 'avg_mfe_atr': 2.261, 'avg_mae_atr': 2.23, 'realized_rr': 1.014},
        },
        'CL_1h': {
            'Isolation-Forest': {'unconditional': 0.4569, 'filtered': 0.5147, 'lift': 0.0578, 'z': 3.06, 'n_holdout': 272, 'avg_mfe_atr': 2.77, 'avg_mae_atr': 2.125, 'realized_rr': 1.304},
            'OPTICS': {'unconditional': 0.424, 'filtered': 0.4667, 'lift': 0.0426, 'z': 2.21, 'n_holdout': 630, 'avg_mfe_atr': 2.78, 'avg_mae_atr': 2.219, 'realized_rr': 1.253},
            'TDA': {'unconditional': 0.427, 'filtered': 0.4631, 'lift': 0.0361, 'z': 1.89, 'n_holdout': 542, 'avg_mfe_atr': 2.751, 'avg_mae_atr': 2.211, 'realized_rr': 1.244},
            'Multiscale-HDBSCAN': {'unconditional': 0.4277, 'filtered': 0.4537, 'lift': 0.0259, 'z': 1.65, 'n_holdout': 712, 'avg_mfe_atr': 2.663, 'avg_mae_atr': 2.178, 'realized_rr': 1.223},
            'HDBSCAN': {'unconditional': 0.417, 'filtered': 0.4525, 'lift': 0.0355, 'z': 1.58, 'n_holdout': 705, 'avg_mfe_atr': 2.742, 'avg_mae_atr': 2.16, 'realized_rr': 1.27},
            'GMM': {'unconditional': 0.417, 'filtered': 0.4381, 'lift': 0.0211, 'z': 0.54, 'n_holdout': 299, 'avg_mfe_atr': 2.579, 'avg_mae_atr': 2.308, 'realized_rr': 1.117},
            'MeanShift': {'unconditional': 0.4115, 'filtered': 0.4319, 'lift': 0.0204, 'z': 0.38, 'n_holdout': 426, 'avg_mfe_atr': 2.536, 'avg_mae_atr': 2.247, 'realized_rr': 1.128},
            'KDE': {'unconditional': 0.4381, 'filtered': 0.4227, 'lift': -0.0154, 'z': -0.0, 'n_holdout': 194, 'avg_mfe_atr': 2.588, 'avg_mae_atr': 2.28, 'realized_rr': 1.135},
        },
        'CL_4h': {
            'KDE': {'unconditional': 0.4906, 'filtered': 0.5319, 'lift': 0.0413, 'z': 1.51, 'n_holdout': 47, 'avg_mfe_atr': 2.292, 'avg_mae_atr': 1.534, 'realized_rr': 1.495},
            'Isolation-Forest': {'unconditional': 0.4976, 'filtered': 0.5275, 'lift': 0.0299, 'z': 2.02, 'n_holdout': 91, 'avg_mfe_atr': 2.297, 'avg_mae_atr': 1.866, 'realized_rr': 1.231},
            'MeanShift': {'unconditional': 0.4978, 'filtered': 0.5045, 'lift': 0.0067, 'z': 1.74, 'n_holdout': 111, 'avg_mfe_atr': 2.193, 'avg_mae_atr': 1.802, 'realized_rr': 1.217},
            'OPTICS': {'unconditional': 0.5196, 'filtered': 0.4904, 'lift': -0.0292, 'z': 1.39, 'n_holdout': 104, 'avg_mfe_atr': 2.292, 'avg_mae_atr': 1.491, 'realized_rr': 1.538},
            'Multiscale-HDBSCAN': {'unconditional': 0.4197, 'filtered': 0.4581, 'lift': 0.0384, 'z': 0.89, 'n_holdout': 155, 'avg_mfe_atr': 2.268, 'avg_mae_atr': 1.806, 'realized_rr': 1.256},
            'HDBSCAN': {'unconditional': 0.4442, 'filtered': 0.4321, 'lift': -0.0121, 'z': 0.24, 'n_holdout': 162, 'avg_mfe_atr': 2.231, 'avg_mae_atr': 1.711, 'realized_rr': 1.304},
            'GMM': {'unconditional': 0.4379, 'filtered': 0.4211, 'lift': -0.0168, 'z': -0.03, 'n_holdout': 76, 'avg_mfe_atr': 2.14, 'avg_mae_atr': 1.752, 'realized_rr': 1.222},
            'TDA': {'unconditional': 0.3973, 'filtered': 0.3985, 'lift': 0.0012, 'z': -0.56, 'n_holdout': 133, 'avg_mfe_atr': 2.083, 'avg_mae_atr': 1.757, 'realized_rr': 1.186},
        },    },
}










# ============================================================================
# VOLUME PROFILE & LEVEL REACTION ANALYSIS
# ============================================================================

def calculate_vwap(highs, lows, closes, volumes, timestamps=None, n_sigma_bands=(1, 2, 3)):
    """
    VWAP with standard-deviation bands, anchored to session (day) if
    timestamps are available, otherwise a single anchor over the whole
    window (rolling-window callers should pass just the trailing slice
    they want anchored from).

    Uses typical price (H+L+C)/3, the standard VWAP convention, weighted by
    volume. Bands use the volume-weighted variance of typical price around
    the running VWAP, so they widen/narrow with actual dispersion instead
    of a fixed percentage.

    Returns
    -------
    dict: {
        'vwap': float,               # current (session-to-date) VWAP
        'vwap_series': np.ndarray,   # running VWAP at every bar
        'bands': {sigma: {'upper': float, 'lower': float}, ...},  # current bands
        'band_series': {sigma: {'upper': np.ndarray, 'lower': np.ndarray}, ...},
        'session_start_idx': int,    # index the current session's anchor starts at
    }
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(closes)
    if n == 0:
        return None

    typical_price = (highs + lows + closes) / 3.0

    # Determine session anchor: reset at each new calendar day if we have
    # timestamps, otherwise anchor once at the start of the whole array
    # (caller's responsibility to pass a session-sliced window in that case).
    if timestamps is not None:
        ts = pd.to_datetime(pd.Series(timestamps))
        session_ids = ts.dt.date
        session_start_idx = int(np.where(session_ids.values == session_ids.values[-1])[0][0])
    else:
        session_start_idx = 0

    pv = typical_price * volumes
    cum_pv = np.zeros(n)
    cum_vol = np.zeros(n)
    running_pv, running_vol = 0.0, 0.0
    for i in range(n):
        if i == session_start_idx:
            running_pv, running_vol = 0.0, 0.0
        running_pv += pv[i]
        running_vol += volumes[i]
        cum_pv[i] = running_pv
        cum_vol[i] = running_vol if running_vol > 0 else 1e-9

    vwap_series = cum_pv / cum_vol

    # Volume-weighted variance of typical price around running VWAP, session-to-date
    band_series = {s: {'upper': np.zeros(n), 'lower': np.zeros(n)} for s in n_sigma_bands}
    running_sq_dev_vol = 0.0
    for i in range(n):
        if i == session_start_idx:
            running_sq_dev_vol = 0.0
        dev = typical_price[i] - vwap_series[i]
        running_sq_dev_vol += (dev ** 2) * volumes[i]
        variance = running_sq_dev_vol / cum_vol[i]
        sigma = np.sqrt(max(variance, 0))
        for s in n_sigma_bands:
            band_series[s]['upper'][i] = vwap_series[i] + s * sigma
            band_series[s]['lower'][i] = vwap_series[i] - s * sigma

    return {
        'vwap': float(vwap_series[-1]),
        'vwap_series': vwap_series,
        'bands': {s: {'upper': float(band_series[s]['upper'][-1]),
                      'lower': float(band_series[s]['lower'][-1])} for s in n_sigma_bands},
        'band_series': band_series,
        'session_start_idx': session_start_idx,
    }








# ============================================================================
# MULTI-TIMEFRAME LEVEL-BASED LSTM FORECASTING
# Predicts: Which levels will be touched, in what order, and when
# ============================================================================





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






# ============================================================================
# LEVEL-BASED LSTM FORECAST: "Where is price going today?"
# ============================================================================


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






# NEW ENDPOINT: LEVEL-CONSTRAINED HOD/LOD PREDICTION
@app.route('/api/level-constrained-hod-lod', methods=['GET'])
def get_level_constrained_hod_lod():
    """
    Enhanced HOD/LOD prediction using your level detection as constraints
    Instead of pure statistical ranges, this finds the most probable HOD/LOD
    by weighting detected levels with volatility expectations
    """
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    timeframe = request.args.get('timeframe', '1d').strip().lower().replace('240m','4h').replace('4hour','4h').replace('4hours','4h').replace('60m','1h')
    
    try:
        print(f"Calculating level-constrained HOD/LOD for {ticker}...")
        
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
            # Futures have limited intraday data - use shorter periods
            # 15m and 1h get slightly longer periods as they're more reliable
            period_map = {'1m': '5d', '5m': '5d', '15m': '7d', '1h': '7d', '4h': '10d', '1d': '2y'}
        else:
            period_map = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '3mo', '4h': '3mo', '1d': '2y'}
        
        period = period_map.get(timeframe, '1y')
        
        # Try to get data, with fallback to shorter periods if needed
        # More aggressive fallback for 15m, 1h, and 4h
        hist = None
        if is_futures and timeframe == '1h':
            # For 1h futures, try many combinations
            attempts = [
                ('60m', '5d'), ('60m', '3d'), ('60m', '2d'), ('60m', '1d'),
                ('1h', '5d'), ('1h', '3d'), ('1h', '2d'), ('1h', '1d'),
            ]
            for attempt_interval, attempt_period in attempts:
                try:
                    print(f"Trying {ticker} 1h: interval={attempt_interval}, period={attempt_period}")
                    hist = stock.history(period=attempt_period, interval=attempt_interval)
                    if hist is not None and len(hist) > 0:
                        print(f"✓ Successfully fetched {len(hist)} bars for {ticker} 1h")
                        break
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:150]}")
                    continue
        elif timeframe == '4h':
            # yfinance doesn't support '4h'/'240m' natively for any ticker -
            # must fetch 1h/60m and resample, regardless of futures/crypto
            print(f"Fetching 4h data for {ticker} (will resample from 1h/60m)...")
            try:
                hist = fetch_historical_data_with_resampling(
                    ticker=ticker,
                    timeframe='4h',
                    period=period,
                    is_futures=is_futures
                )
            except Exception as e:
                print(f"⚠ Resampling fetch failed: {e}")
                hist = None
        elif is_futures and timeframe in ['1m', '5m', '15m']:
            if timeframe in ['15m']:
                attempts = [period, '5d', '3d', '2d', '1d']
            else:
                attempts = [period, '5d', '2d', '1d']
            
            for attempt_period in attempts:
                interval_options = [interval]
                if timeframe == '15m':
                    interval_options = ['15m']
                
                for attempt_interval in interval_options:
                    try:
                        hist = stock.history(period=attempt_period, interval=attempt_interval)
                        if hist is not None and len(hist) > 0:
                            print(f"✓ Successfully fetched {len(hist)} bars for {ticker} at {timeframe}")
                            break
                    except Exception as e:
                        error_msg = str(e)
                        if "pattern" not in error_msg.lower() and "expected" not in error_msg.lower():
                            print(f"⚠ Attempt failed: interval={attempt_interval}, period={attempt_period}, error={error_msg[:100]}")
                        continue
                
                if hist is not None and len(hist) > 0:
                    break
        else:
            attempts = [period]
            for attempt_period in attempts:
                try:
                    hist = stock.history(period=attempt_period, interval=interval)
                    if hist is not None and len(hist) > 0:
                        break
                except Exception as e:
                    continue
        
        if hist is None or len(hist) == 0:
            return jsonify({'success': False, 'error': f'No data available for {ticker} at {timeframe}. Futures have limited intraday data availability.'}), 400
        
        closes = hist['Close'].values
        highs = hist['High'].values if 'High' in hist.columns else closes
        lows = hist['Low'].values if 'Low' in hist.columns else closes
        opens = hist['Open'].values if 'Open' in hist.columns else closes
        volumes = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(closes))
        current_price = closes[-1]

        # 1. Get session volatility (for next-period prediction, not annualized)
        garch_vol_regime = calculate_garch_volatility_regime(closes)
        
        # Use session volatility for accuracy
        if all(col in hist.columns for col in ['Open', 'High', 'Low', 'Close']):
            try:
                vol_result = compute_session_volatility(hist, window=60)
                session_vol_pct = vol_result['sigma_session_pct']  # Next session % (1-3%)
                sigma_price = vol_result['sigma_price']  # Expected $ range
                sigma_annual_pct = vol_result['sigma_annual_pct']  # For logging/comparison
                method = 'Session Volatility + Levels'
                
                # Validate
                if np.isnan(session_vol_pct) or not np.isfinite(session_vol_pct) or session_vol_pct <= 0:
                    raise ValueError("Invalid session volatility")
                
                print(f"✓ Session vol: {session_vol_pct:.2f}% (annualized: {sigma_annual_pct:.1f}%, σ_price: ${sigma_price:.2f})")
                
            except Exception as e:
                print(f"⚠ Session vol failed: {e}, using fallback")
                returns = np.log(closes[1:] / closes[:-1])
                if len(returns) > 0:
                    sigma_session = np.std(returns)
                    session_vol_pct = sigma_session * 100
                    sigma_price = sigma_session * current_price
                else:
                    session_vol_pct = 1.5  # 1.5% default session vol
                    sigma_price = (session_vol_pct / 100) * current_price
                method = 'Fallback Session Vol + Levels'
        else:
            # No OHLC data, use close-to-close returns
            returns = np.log(closes[1:] / closes[:-1])
            if len(returns) > 0:
                sigma_session = np.std(returns)
                session_vol_pct = sigma_session * 100
                sigma_price = sigma_session * current_price
            else:
                session_vol_pct = 1.5  # 1.5% default session vol
                sigma_price = (session_vol_pct / 100) * current_price
            method = 'Fallback Session Vol + Levels'
        
        # Final validation
        if np.isnan(session_vol_pct) or not np.isfinite(session_vol_pct) or session_vol_pct <= 0:
            print(f"⚠ Invalid session_vol_pct: {session_vol_pct}, using default 1.5%")
            session_vol_pct = 1.5
        
        # Recalculate sigma_price if needed
        if np.isnan(sigma_price) or not np.isfinite(sigma_price) or sigma_price <= 0:
            print(f"⚠ Invalid sigma_price: {sigma_price}, recalculating")
            sigma_price = (session_vol_pct / 100) * current_price
        
        # 2. Get microstructure state (affects how we weight levels)
        returns = np.log(closes[1:] / closes[:-1]) * 100
        microstructure_state = detect_market_microstructure_state(closes, volumes, returns, highs, lows)
        
        # 3. DETECT ALL YOUR LEVELS (using your existing functions)
        print("Running level detection algorithms...")
        
        hist_data_subset = hist.tail(min(len(hist), 100))
        
        # PRIMARY: HDBSCAN (state-of-the-art density clustering)
        hdbscan_levels = calculate_hdbscan_levels(highs, lows, closes, timeframe=timeframe)

        # SECONDARY: IsolationForest (event pivot candidates)
        isolation_forest_levels = find_pivot_anomalies(highs, lows, closes)

        # Enhanced OPTICS, Multi-scale HDBSCAN, Time-weighted HDBSCAN, Wyckoff -
        # brought in line with /api/data's algorithm set (was previously
        # missing here, organic drift between endpoints built at different times)
        try:
            optics_levels_result = enhanced_optics_levels(highs, lows, closes, timeframe=timeframe)
        except Exception as e:
            print(f"Enhanced OPTICS failed: {e}")
            optics_levels_result = []

        try:
            multiscale_hdbscan_levels_result = multiscale_hdbscan_levels(highs, lows, closes, timeframe=timeframe)
        except Exception as e:
            print(f"Multi-scale HDBSCAN failed: {e}")
            multiscale_hdbscan_levels_result = []

        time_weighted_levels_result = []
        try:
            if hasattr(hist.index, 'values'):
                time_weighted_levels_result = time_weighted_hdbscan(highs, lows, closes, hist.index.values, half_life_days=30)
        except Exception as e:
            print(f"Time-weighted HDBSCAN failed: {e}")
            time_weighted_levels_result = []

        try:
            wyckoff_levels_result = detect_wyckoff_zones(hist_data_subset, lookback=50)
        except Exception as e:
            print(f"Wyckoff zones failed: {e}")
            wyckoff_levels_result = []

        # GMM, TDA, KDE: ML-filtered below alongside HDBSCAN/Isolation-Forest/OPTICS/MeanShift
        gmm_levels_result = calculate_gmm_levels(highs, lows, closes)
        tda_levels_result = persistent_homology_levels(highs, lows, closes, max_levels=8)
        kde_levels_result = kde_based_levels(highs, lows, closes, n_levels=10)
        # MeanShift: scored through the validated v2 filter as its own
        # candidate source (one of the 7 Bonferroni-significant categories,
        # +0.064 discrimination improvement in v2) - separate from, and not
        # mutually exclusive with, its other role as a validator against
        # HDBSCAN levels via enhance_levels_with_microstructure() below.
        meanshift_levels_result = calculate_meanshift_levels(highs, lows, closes)

        try:
            levels_by_category_v2 = {
                'GMM': gmm_levels_result, 'TDA': tda_levels_result,
                'HDBSCAN': hdbscan_levels, 'Isolation-Forest': isolation_forest_levels,
                'KDE': kde_levels_result, 'OPTICS': optics_levels_result,
                'MeanShift': meanshift_levels_result,
            }
            levels_by_category_v2 = extend_thin_side_levels(
                ticker, timeframe, highs, lows, closes, current_price, levels_by_category_v2, is_futures,
            )
            filtered = score_and_filter_levels_v2(
                levels_by_category_v2,
                highs, lows, opens, closes, volumes, current_price, timestamps=hist.index.values,
            )
            gmm_levels_result = [l for l in filtered if l.get('category') == 'GMM']
            tda_levels_result = [l for l in filtered if l.get('category') == 'TDA']
            hdbscan_levels = [l for l in filtered if l.get('category') in ('HDBSCAN', 'Density (HDBSCAN)')]
            isolation_forest_levels = [l for l in filtered if l.get('category') == 'Isolation-Forest']
            kde_levels_result = [l for l in filtered if l.get('category') == 'KDE']
            optics_levels_result = [l for l in filtered if l.get('category') == 'OPTICS']
            meanshift_levels_result = [l for l in filtered if l.get('category') == 'MeanShift']
        except Exception as e:
            print(f"ML filter failed, falling back to unfiltered levels: {e}")

        # Neural Network levels (with volume profile)
        try:
            neural_network_levels_result = detect_levels_with_neural_network(hist_data_subset, lookback=100, threshold=0.5)
            print(f"Neural Network: Generated {len(neural_network_levels_result) if neural_network_levels_result else 0} levels")
        except Exception as e:
            print(f"Neural Network level detection failed: {e}")
            neural_network_levels_result = []

        # Fibonacci for metadata enrichment only (not primary levels)
        fib_levels = calculate_fibonacci_levels(highs, lows)

        # ML LEVELS: Primary discovery algorithms only
        all_ml_levels = (hdbscan_levels + isolation_forest_levels + gmm_levels_result + tda_levels_result +
                        kde_levels_result + optics_levels_result + multiscale_hdbscan_levels_result +
                        time_weighted_levels_result + wyckoff_levels_result + meanshift_levels_result +
                        (neural_network_levels_result if neural_network_levels_result else []))
        
        # NEW: Agglomerative merge BEFORE confluence (prevents probability fragmentation)
        # Use timeframe-aware threshold (cleaner than regime-aware for this step)
        all_ml_levels = agglomerative_merge_levels(
            all_ml_levels,
            distance_threshold_pct=None,  # Will use timeframe-aware default
            timeframe=timeframe
        )
        
        confluence_levels = get_ml_confluence_levels(all_ml_levels)

        # Combine ML levels (no gap/pivot/peak-valley — volume-based only)
        all_levels_combined = confluence_levels + all_ml_levels

        # Add Fibonacci as metadata/confluence to nearby levels (not as primary levels)
        all_levels_combined = add_fibonacci_metadata_to_levels(
            all_levels_combined, fib_levels, sigma_price, threshold_sigma=1.0
        )
        
        # 4. ENHANCE LEVELS with microstructure
        all_levels_combined, hmm_regime, hurst_data, garch_regime, micro_state = enhance_levels_with_microstructure(
            all_levels_combined, closes, volumes, current_price, garch_vol_regime, microstructure_state, sigma_price=sigma_price
        )
        
        print(f"✓ Detected {len(all_levels_combined)} total levels")
        
        # NEW: Apply Fractional Brownian Motion adjustment if Hurst is available
        if hurst_data and 'hurst' in hurst_data:
            try:
                # Get base sigma first (will adjust after fractional Brownian)
                base_sigma = sigma_price
                
                # Calculate base predictions from volatility
                base_hod_2std_temp = current_price + 2.0 * base_sigma
                base_lod_2std_temp = current_price - 2.0 * base_sigma
                
                # Apply fractional Brownian adjustment
                adj_hod_2std, adj_lod_2std = fractional_brownian_adjustment(
                    base_hod_2std_temp, base_lod_2std_temp, hurst_data['hurst'], base_sigma
                )
                
                # Recalculate sigma_price if adjustment was significant
                if abs(adj_hod_2std - base_hod_2std_temp) > base_sigma * 0.1:
                    sigma_price = (adj_hod_2std - adj_lod_2std) / 4.0  # Recalculate sigma from adjusted range
                    print(f"✓ Applied fractional Brownian adjustment (Hurst={hurst_data['hurst']:.3f})")
            except Exception as e:
                print(f"⚠ Fractional Brownian adjustment failed: {e}")
        
        # 5. FIND MOST PROBABLE HOD/LOD using levels as attractors
        
        # Separate into resistance (above current) and support (below current)
        resistance_levels = [l for l in all_levels_combined if l['price'] > current_price]
        support_levels = [l for l in all_levels_combined if l['price'] < current_price]
        
        # Sort by distance from current price
        resistance_levels.sort(key=lambda x: x['price'])
        support_levels.sort(key=lambda x: -x['price'])
        
        # Calculate base predictions from volatility (your sigma ranges)
        base_hod_1std = current_price + 1.0 * sigma_price
        base_lod_1std = current_price - 1.0 * sigma_price
        base_hod_2std = current_price + 2.0 * sigma_price
        base_lod_2std = current_price - 2.0 * sigma_price
        base_hod_3std = current_price + 3.0 * sigma_price
        base_lod_3std = current_price - 3.0 * sigma_price
        
        # Get lower timeframe theoretical LOD for validation
        lower_tf_lod = None
        try:
            if timeframe in ['1h', '4h', '1d']:
                lower_tf_hist = stock.history(period='5d', interval='15m')
                if len(lower_tf_hist) > 0:
                    lower_tf_vol = compute_session_volatility(lower_tf_hist, window=60)
                    lower_tf_sigma = lower_tf_vol['sigma_price']
                    lower_tf_lod = current_price - 1.5 * lower_tf_sigma
        except:
            pass
        
        # FIND MOST PROBABLE HOD/LOD using your refined approach
        predicted_hod, predicted_lod, refinement_debug = refine_extrema_with_levels(
            spot=current_price,
            hod_th=base_hod_2std,  # Use 2σ as envelope bound
            lod_th=base_lod_2std,
            levels=all_levels_combined,
            state=micro_state,
            timeframe=timeframe,
            lower_tf_lod=lower_tf_lod
        )
        
        # Find which levels were selected
        selected_resistance = refinement_debug.get('best_hod')
        selected_support = refinement_debug.get('best_lod')
        
        # Calculate confidence scores
        hod_confidence = calculate_level_confidence(predicted_hod, resistance_levels, current_price, sigma_price)
        lod_confidence = calculate_level_confidence(predicted_lod, support_levels, current_price, sigma_price)
        
        # Multi-timeframe confluence (soft structural ceilings/floors)
        mtf_confluence = compute_mtf_confluence(
            ticker=ticker,
            spot=current_price,
            sigma_price=sigma_price,
            micro_state=micro_state.get('state', 'Unknown'),
            lookback=20
        )
        
        # Apply MTF structural constraints (soft caps, not hard limits)
        if mtf_confluence['apply']:
            # Soft structural ceilings/floors - improve confidence, not expand range
            if mtf_confluence['resistance'] and predicted_hod > mtf_confluence['resistance']:
                # Predicted HOD exceeds MTF resistance - cap it softly
                predicted_hod = min(predicted_hod, mtf_confluence['resistance'] * 1.02)  # Allow 2% overshoot
                print(f"✓ MTF resistance at ${mtf_confluence['resistance']:.2f} → capped HOD")
            
            if mtf_confluence['support'] and predicted_lod < mtf_confluence['support']:
                # Predicted LOD below MTF support - cap it softly
                predicted_lod = max(predicted_lod, mtf_confluence['support'] * 0.98)  # Allow 2% undershoot
                print(f"✓ MTF support at ${mtf_confluence['support']:.2f} → capped LOD")
            
            # Boost confidence if MTF structure is confirmed
            hod_confidence = min(1.0, hod_confidence + mtf_confluence['confidence_boost'])
            lod_confidence = min(1.0, lod_confidence + mtf_confluence['confidence_boost'])
            print(f"✓ MTF confluence confirmed → confidence boost: {mtf_confluence['confidence_boost']:.1%}")
        
        # Convert session vol pct to decimal for stdDev (frontend expects decimal)
        std_dev_decimal = session_vol_pct / 100.0
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'timeframe': timeframe,
            'currentPrice': float(current_price),
            'sigmaDailyPct': float(session_vol_pct),  # Session vol for next period
            'sigmaPrice': float(sigma_price),
            'stdDev': float(std_dev_decimal),  # Frontend expects decimal (will multiply by 100)
                                'method': method,
            
            # Frontend expects: hod['1std'], hod['2std'], hod['3std']
                                'hod': {
                '1std': float(base_hod_1std),
                '2std': float(base_hod_2std),
                '3std': float(base_hod_3std)
            },
            
            # Frontend expects: lod['1std'], lod['2std'], lod['3std']
                                'lod': {
                '1std': float(base_lod_1std),
                '2std': float(base_lod_2std),
                '3std': float(base_lod_3std)
            },
            
            # Additional data (for advanced use)
            'predicted': {
                'hod': float(predicted_hod),
                'lod': float(predicted_lod),
                'hod_distance_pct': float((predicted_hod - current_price) / current_price * 100),
                'lod_distance_pct': float((current_price - predicted_lod) / current_price * 100),
                'hod_confidence': float(hod_confidence),
                'lod_confidence': float(lod_confidence)
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
            
            # All nearby levels (for visualization)
            'nearbyLevels': {
                'resistance': sanitize_for_json(resistance_levels[:5]),
                'support': sanitize_for_json(support_levels[:5])
            },
            
            # Refinement debug info
            'refinement': sanitize_for_json(refinement_debug),
            
            'microstructure': sanitize_for_json(micro_state),
            'garchRegime': sanitize_for_json(garch_regime),
            'mtfConfluence': sanitize_for_json(mtf_confluence) if 'mtf_confluence' in locals() else None
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/level-constrained-hod-lod: {error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 400

# ALIAS: Keep old endpoint name for backward compatibility
@app.route('/api/stdv-hod-lod', methods=['GET'])
def get_stdv_hod_lod():
    """Alias for /api/level-constrained-hod-lod - backward compatibility"""
    return get_level_constrained_hod_lod()

# NEW ENDPOINT: STATE-CONDITIONED HOD/LOD





_EQUITY_CURVE_CACHE = None












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

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Main routes
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/login')
def login_page():
    if 'username' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout_page():
    session.clear()
    return redirect(url_for('login_page'))


# ============================================================================
# VALIANT SELECTION STACK + LOOKUP TOOL
#
# Two new, independent pieces added alongside the pruned level-detection
# system (the levels + theoretical HOD/LOD above): the frozen long-only
# Valiant selection stack (fundamentals gate + quality z-rank, sector cap,
# inverse-vol weight, flat exposure - validated in-sample, now paper-
# tracked prospectively from 2026-09-15 with no further tuning) and the
# per-stock diagnostic lookup tool (state description, not a prediction).
# NOTE: deliberately NOT imported at module level. backtest_levels.py
# (part of the separate research pipeline) does `import backend` to reuse
# this file's level-detection engine - importing anything from that
# pipeline back into backend.py at load time creates a circular import
# that breaks the entire pipeline. Each route below imports lazily
# instead, at request time, after every module has already finished
# loading.
# ============================================================================


@app.route('/api/valiant-lookup', methods=['GET'])
def get_valiant_lookup():
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']

    ticker = request.args.get('ticker', 'AAPL').strip().upper()
    timeframe = request.args.get('timeframe', 'daily').strip().lower()
    if timeframe not in ('daily', 'weekly', 'monthly'):
        return jsonify({'success': False, 'error': 'timeframe must be daily, weekly, or monthly'}), 400

    try:
        from valiant_stock_lookup import build_report
        report, err = build_report(ticker, timeframe)
        if report is None:
            return jsonify({'success': False, 'error': err}), 404
        return jsonify({'success': True, 'report': sanitize_for_json(report)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/valiant-status', methods=['GET'])
def get_valiant_status():
    """Current paper-tracking ledger: every recorded month's frozen-stack
    picks, plus forward return so far for each (flat exposure - the
    validated variant; the correlation exposure dial is logged for
    comparison only, not applied). Read-only."""
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']

    try:
        import os, json as json_module
        import valiant_paper_tracking
        if not os.path.exists(valiant_paper_tracking.LEDGER_PATH):
            return jsonify({'success': True, 'ledger': [], 'note': 'No paper-tracking entries recorded yet.'})
        with open(valiant_paper_tracking.LEDGER_PATH) as f:
            ledger = json_module.load(f)

        all_tickers = sorted({p['ticker'] for e in ledger for p in e['picks']})
        raw = valiant_paper_tracking.batch_download_cached(all_tickers) if all_tickers else {}

        entries = []
        for entry in ledger:
            picks_out = []
            month_ret_flat = 0.0
            for p in entry['picks']:
                df = raw.get(p['ticker'])
                cur_price = float(df['close'].iloc[-1]) if df is not None and len(df) else None
                fwd_ret = (cur_price / p['entry_price'] - 1) if cur_price else None
                month_ret_flat += (fwd_ret or 0.0) * p['weight_flat']
                picks_out.append({**p, 'current_price': cur_price, 'forward_return': fwd_ret})
            entries.append({**entry, 'picks': picks_out, 'basket_forward_return_flat': month_ret_flat})

        return jsonify({
            'success': True,
            'methodology': ('FROZEN as of 2026-09-15: real-time S&P 500 membership, fundamentals-gated '
                             '(SEC EDGAR strict pass) recent-bullish-flip quality z-rank, top 5, 30% sector '
                             'cap, inverse-vol weighted, flat exposure. In-sample backtest showed a '
                             '99th-percentile total-return result vs a sector+count-matched random null - '
                             'not yet validated out of sample. This ledger is the out-of-sample record.'),
            'ledger': sanitize_for_json(entries),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/valiant-record-month', methods=['POST'])
def post_valiant_record_month():
    """Admin-only: records this calendar month's frozen-stack picks into
    the paper-tracking ledger. Intended to be called once per month -
    calling it again in the same month overwrites that month's entry
    rather than duplicating it (see valiant_paper_tracking.record_month)."""
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Admin only'}), 403

    try:
        import valiant_paper_tracking
        entry = valiant_paper_tracking.record_month()
        return jsonify({'success': True, 'entry': sanitize_for_json(entry)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


