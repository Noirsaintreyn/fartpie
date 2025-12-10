from flask import Flask, jsonify, request, session
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime, timedelta
import sqlite3
import hashlib
import warnings
import requests
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'degen-discovery-secret-key-2024'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS

# Allow CORS from anywhere for development
CORS(app, supports_credentials=True, origins=['*'])

FRED_API_KEY = '024452292701539abb68abc50276eb70'

# Simple password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed):
    return hashlib.sha256(password.encode()).hexdigest() == hashed

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
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
    
    admin_password = hash_password('admin123')
    try:
        c.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                  ('admin', 'admin@degendiscovery.com', admin_password, 1))
        conn.commit()
        print("✓ Admin account created")
    except sqlite3.IntegrityError:
        print("✓ Admin account already exists")
    conn.close()

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
        conn = sqlite3.connect('users.db')
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
    print(f"Login attempt received: {request.json}")
    
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Missing credentials'}), 400
    
    conn = sqlite3.connect('users.db')
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
        'is_admin': is_admin
    })

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT is_active FROM users WHERE id = ?", (session['user_id'],))
        result = c.fetchone()
        conn.close()
        
        if result and result[0]:
            return jsonify({
                'authenticated': True,
                'username': session.get('username'),
                'is_admin': session.get('is_admin', False)
            })
    return jsonify({'authenticated': False}), 401

@app.route('/api/admin/users', methods=['GET'])
def get_users():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect('users.db')
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
    
    conn = sqlite3.connect('users.db')
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
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET is_active = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User enabled'})

def require_auth():
    if 'user_id' not in session:
        return {'error': 'Not authenticated', 'code': 401}
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT is_active FROM users WHERE id = ?", (session['user_id'],))
    result = c.fetchone()
    conn.close()
    
    if not result or not result[0]:
        session.clear()
        return {'error': 'Account disabled', 'code': 403}
    return None

# FRED API
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

# Forecasting
def nbeats_forecast(prices, forecast_periods=10, num_scenarios=3):
    if len(prices) < 50:
        return None
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    window = min(20, len(prices) // 3)
    if window < 5:
        window = 5
    trend = pd.Series(prices_scaled).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
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

# Regime detection
def detect_market_regime_hmm(closes, n_states=3):
    try:
        from hmmlearn.hmm import GaussianHMM
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

# Level detection functions
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

def calculate_dbscan_levels(highs, lows):
    all_prices = np.concatenate([highs, lows]).reshape(-1, 1)
    eps = (all_prices.max() - all_prices.min()) * 0.02
    db = DBSCAN(eps=eps, min_samples=5).fit(all_prices)
    labels = db.labels_
    core_samples = np.unique(labels[labels != -1])
    levels = []
    for cluster_id in core_samples:
        cluster_points = all_prices[labels == cluster_id].flatten()
        center = np.median(cluster_points)
        strength = min(len(cluster_points) / 50, 0.85)
        levels.append({'price': float(center), 'type': 'DBSCAN', 'touches': len(cluster_points),
                      'strength': strength, 'breakoutProb': float(1 - strength),
                      'reversionProb': float(strength), 'category': 'DBSCAN'})
    return sorted(levels, key=lambda x: x['strength'], reverse=True)[:5]

def calculate_gmm_levels(closes, highs, lows, n_components=4):
    all_prices = np.concatenate([highs, lows, closes]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
    gmm.fit(all_prices)
    means = gmm.means_.flatten()
    weights = gmm.weights_
    levels = []
    for i, (mean, weight) in enumerate(zip(means, weights)):
        levels.append({'price': float(mean), 'type': f'GMM-{i+1}', 'strength': float(weight),
                      'breakoutProb': float(1 - weight), 'reversionProb': float(weight), 'category': 'GMM'})
    return levels

def calculate_kmeans_levels(highs, lows):
    points = np.concatenate([highs, lows])
    k = 8
    if len(points) < k:
        return []
    centers = np.random.choice(points, k, replace=False)
    for _ in range(15):
        clusters = [[] for _ in range(k)]
        for p in points:
            idx = np.argmin(np.abs(centers - p))
            clusters[idx].append(p)
        for i in range(k):
            if clusters[i]:
                centers[i] = np.mean(clusters[i])
    levels = []
    for center in centers:
        touches = np.sum((np.abs(highs - center) < (highs - lows) * 0.5) | 
                        (np.abs(lows - center) < (highs - lows) * 0.5))
        if touches > 2:
            strength = min(touches / 10, 0.85)
            levels.append({'price': float(center), 'type': 'K-Means', 'breakoutProb': float(1 - strength),
                          'reversionProb': float(strength), 'category': 'K-Means', 
                          'touches': int(touches), 'strength': strength})
    return sorted(levels, key=lambda x: x['touches'], reverse=True)[:5]

def calculate_vol_levels(closes, current):
    returns = np.log(closes[1:] / closes[:-1])
    vol = np.std(returns) * np.sqrt(252)
    levels = []
    for d in [1, 5, 10]:
        for s in [1, 2]:
            factor = np.exp(s * vol * np.sqrt(d / 252))
            prob = 0.16 if s == 1 else 0.025
            levels.append({'price': float(current * factor), 'type': f'Vol +{s}σ {d}d',
                          'breakoutProb': prob, 'reversionProb': 1 - prob, 
                          'category': 'Volatility', 'strength': 1 - prob})
            levels.append({'price': float(current / factor), 'type': f'Vol -{s}σ {d}d',
                          'breakoutProb': prob, 'reversionProb': 1 - prob,
                          'category': 'Volatility', 'strength': 1 - prob})
    return levels

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
    if len(highs) < 20:
        return []
    recent_high = np.max(highs[-50:])
    recent_low = np.min(lows[-50:])
    range_val = recent_high - recent_low
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    levels = []
    for ratio in fib_ratios:
        level_from_high = recent_high - (range_val * ratio)
        levels.append({'price': float(level_from_high), 'type': f'Fib {ratio:.3f}',
                      'strength': 0.7, 'breakoutProb': 0.3, 'reversionProb': 0.7, 'category': 'Fibonacci'})
    return levels

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

def calculate_short_term_iv(closes, window=5):
    if len(closes) < window + 1:
        window = max(2, len(closes) - 1)
    returns = np.log(closes[1:] / closes[:-1])
    recent_returns = returns[-window:]
    daily_vol = np.std(recent_returns)
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol * 100

def calculate_volatility_regime(closes, short_window=5, long_window=20):
    short_iv = calculate_short_term_iv(closes, short_window)
    long_iv = calculate_short_term_iv(closes, long_window)
    ratio = short_iv / long_iv if long_iv > 0 else 1.0
    if ratio > 1.3:
        regime = "High Vol Spike"
        regime_factor = 1.5
    elif ratio > 1.1:
        regime = "Elevated Vol"
        regime_factor = 1.2
    elif ratio < 0.8:
        regime = "Low Vol Compression"
        regime_factor = 0.7
    else:
        regime = "Normal Vol"
        regime_factor = 1.0
    return {'regime': regime, 'short_iv': float(short_iv), 'long_iv': float(long_iv),
            'ratio': float(ratio), 'regime_factor': regime_factor}

def enhance_levels_with_regime_detection(levels, closes, current_price):
    hmm_regime = detect_market_regime_hmm(closes)
    hurst_data = calculate_hurst_exponent(closes)
    for level in levels:
        original_strength = level.get('strength', 0.5)
        if hmm_regime['state'] == 0:
            if level['price'] > current_price:
                level['strength'] = min(original_strength * 1.25, 0.98)
            else:
                level['strength'] = min(original_strength * 1.15, 0.98)
        elif hmm_regime['state'] == 2:
            if level['price'] < current_price:
                level['strength'] = min(original_strength * 1.25, 0.98)
            else:
                level['strength'] = min(original_strength * 1.15, 0.98)
        level['strength'] = min(level['strength'] * hurst_data['level_multiplier'], 0.98)
        level['hmm_regime'] = hmm_regime['regime']
        level['hmm_confidence'] = hmm_regime['confidence']
        level['hurst_exponent'] = hurst_data['hurst']
        level['hurst_regime'] = hurst_data['regime']
        level['breakoutProb'] = float(1 - level['strength'])
        level['reversionProb'] = float(level['strength'])
        distance_dollars = abs(level['price'] - current_price)
        distance_pct = (distance_dollars / current_price) * 100
        level['distance_dollars'] = float(distance_dollars)
        level['distance_pct'] = float(distance_pct)
    return levels, hmm_regime, hurst_data

def get_ml_confluence_levels(all_algorithm_levels):
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
            final_levels.append({'price': float(avg_price), 'type': 'ML Confluence',
                                'strength': confluence_strength, 'algorithms': [l['category'] for l in similar],
                                'confluence_count': len(similar), 'breakoutProb': float(1 - confluence_strength),
                                'reversionProb': float(confluence_strength), 'category': 'ML-Confluence'})
            for l in similar:
                used.add(l['price'])
    return sorted(final_levels, key=lambda x: x['strength'], reverse=True)

@app.route('/api/data', methods=['GET'])
def get_data():
    auth_error = require_auth()
    if auth_error:
        return jsonify({'success': False, 'error': auth_error['error']}), auth_error['code']
    
    ticker = request.args.get('ticker', 'SPY')
    timeframe = request.args.get('timeframe', '1d')
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    historical_mode = request.args.get('historical_mode', 'false').lower() == 'true'
    
    try:
        print(f"\n{'='*60}")
        print(f"Analysis: {ticker} - User: {session.get('username')}")
        print(f"{'='*60}")
        
        stock = yf.Ticker(ticker)
        interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
        interval = interval_map.get(timeframe, '1d')
        
        if start_date and end_date:
            hist = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            period_map = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '3mo', '4h': '3mo', '1d': '2y'}
            period = period_map.get(timeframe, '1y')
            hist = stock.history(period=period, interval=interval)
        
        if len(hist) == 0:
            return jsonify({'success': False, 'error': 'No data available'}), 400
        
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
        
        print("Running algorithms...")
        
        forecasts = generate_price_forecast(hist_closes, hist_highs, hist_lows, hist_volumes, forecast_periods=20)
        macro_indicators = get_macro_indicators()
        
        peak_valley_levels = find_peaks_valleys_scipy(hist_highs, hist_lows, hist_closes)
        meanshift_levels = calculate_meanshift_levels(hist_highs, hist_lows, hist_closes)
        dbscan_levels = calculate_dbscan_levels(hist_highs, hist_lows)
        gmm_levels = calculate_gmm_levels(hist_closes, hist_highs, hist_lows)
        pivot_levels = calculate_pivot_points(hist_data_subset, timeframe)
        fib_levels = calculate_fibonacci_levels(hist_highs, hist_lows)
        gap_levels = find_gap_levels(hist_data_subset)
        kmeans_levels = calculate_kmeans_levels(hist_highs, hist_lows)
        vol_levels = calculate_vol_levels(hist_closes, current_price)
        
        vol_regime = calculate_volatility_regime(closes)
        
        all_ml_levels = (peak_valley_levels + meanshift_levels + dbscan_levels + 
                        gmm_levels + fib_levels + kmeans_levels)
        confluence_levels = get_ml_confluence_levels(all_ml_levels)
        
        all_levels_combined = (confluence_levels + peak_valley_levels + meanshift_levels + 
                              dbscan_levels + gmm_levels + pivot_levels + fib_levels + 
                              gap_levels + kmeans_levels + vol_levels)
        
        all_levels_combined, hmm_regime, hurst_data = enhance_levels_with_regime_detection(
            all_levels_combined, closes, current_price
        )
        
        print(f"✓ Complete")
        
        confluence_levels = [l for l in all_levels_combined if l['category'] == 'ML-Confluence']
        peak_valley_levels = [l for l in all_levels_combined if l['category'] == 'Peak-Valley']
        meanshift_levels = [l for l in all_levels_combined if l['category'] == 'MeanShift']
        dbscan_levels = [l for l in all_levels_combined if l['category'] == 'DBSCAN']
        gmm_levels = [l for l in all_levels_combined if l['category'] == 'GMM']
        pivot_levels = [l for l in all_levels_combined if l['category'] == 'Pivot']
        fib_levels = [l for l in all_levels_combined if l['category'] == 'Fibonacci']
        gap_levels = [l for l in all_levels_combined if l['category'] == 'Gap']
        kmeans_levels = [l for l in all_levels_combined if l['category'] == 'K-Means']
        vol_levels = [l for l in all_levels_combined if l['category'] == 'Volatility']
        
        levels = {
            'mlConfluence': confluence_levels,
            'peakValley': peak_valley_levels,
            'meanshift': meanshift_levels,
            'dbscan': dbscan_levels,
            'gmm': gmm_levels,
            'pivots': pivot_levels,
            'fibonacci': fib_levels,
            'gaps': gap_levels,
            'kmeans': kmeans_levels,
            'volatility': vol_levels
        }
        
        return jsonify({
            'success': True,
            'priceData': price_data,
            'levels': levels,
            'currentPrice': float(current_price),
            'volRegime': vol_regime,
            'hmmRegime': hmm_regime,
            'hurstData': hurst_data,
            'forecasts': forecasts,
            'macroIndicators': macro_indicators
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DEGEN DISCOVERY - BACKEND SERVER")
    print("="*60)
    
    init_db()
    
    print("\n✅ Server starting on http://localhost:5001")
    print("✅ Login: admin / admin123")
    print("✅ FRED API configured")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
