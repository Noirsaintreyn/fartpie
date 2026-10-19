"""Website entry point (gunicorn site_app:app): the existing backend + the new L'Argent Parle site.

  /              the new site (screener, mid/small book, charts, history, backtest) - login required
  /forecast      the old LSTM forecast page
  /api/site/...  data for the new site (see below); /api/site/push is called by the screener worker

Kept out of backend.py on purpose: backend.py is imported by the research pipeline, and this file only adds routes.
"""
import json, os, sys, threading, time
import numpy as np
import pandas as pd
from flask import jsonify, request, session, render_template, redirect, url_for

from backend import app, require_auth

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'discord_bot'))
SNAP_FILE = os.getenv('SITE_SNAPSHOT_FILE', '/tmp/site_snapshot.json')
_snap = {'data': None}
_lock = threading.Lock()


def _need_login():
    err = require_auth()
    return (jsonify({'success': False, 'error': err['error']}), err['code']) if err else None


# ---------------------------------------------------------------- pages
def site_index():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('site.html', username=session.get('username', ''))


app.view_functions['index'] = site_index


@app.route('/forecast')
def forecast_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')


# ---------------------------------------------------------------- screener snapshot (pushed by the worker)
@app.route('/api/site/push', methods=['POST'])
def site_push():
    tok = os.getenv('SITE_TOKEN', '')
    if not tok or request.headers.get('X-Site-Token', '') != tok:
        return jsonify({'success': False, 'error': 'forbidden'}), 403
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'bad payload'}), 400
    with _lock:
        _snap['data'] = data
        try:
            json.dump(data, open(SNAP_FILE, 'w'))
        except Exception:
            pass
    return jsonify({'success': True})


@app.route('/api/site/snapshot')
def site_snapshot():
    e = _need_login()
    if e: return e
    with _lock:
        d = _snap['data']
        if d is None and os.path.exists(SNAP_FILE):
            try:
                d = _snap['data'] = json.load(open(SNAP_FILE))
            except Exception:
                d = None
    return jsonify({'success': True, 'snapshot': d})


@app.route('/api/site/static')
def site_static():
    e = _need_login()
    if e: return e
    out = {}
    try:
        out['series'] = json.load(open(os.path.join(ROOT, 'live_data', 'site_series.json')))
    except Exception:
        out['series'] = None
    try:
        dec = json.load(open(os.path.join(ROOT, 'live_data', 'mid_book.json')))['decisions']
        k = sorted(dec)[-1]
        out['mid'] = {'decided': k, 'holdings': dec[k], 'first': sorted(dec)[0]}
    except Exception:
        out['mid'] = None
    return jsonify({'success': True, **out})


# ---------------------------------------------------------------- chart data: bars + vector (any timeframe pair)
FUT = {'NQ', 'ES', 'YM', 'RTY', 'GC', 'SI', 'CL', 'NG', 'ZB', 'ZN'}
TF = {'5m': ('5m', '59d', None, 5), '15m': ('15m', '59d', None, 15), '1h': ('1h', '729d', None, 60),
      '4h': ('1h', '729d', '4h', 240), 'D': ('1d', '10y', None, 1440)}
_bar_cache = {}


def _sym(tk):
    tk = tk.strip().upper()
    return tk + '=F' if tk in FUT else tk


def _bars(tk, tf):
    key = (tk, tf)
    hit = _bar_cache.get(key)
    if hit and time.time() - hit[0] < 120:
        return hit[1]
    import yfinance as yf
    interval, period, resample, _ = TF[tf]
    h = yf.Ticker(_sym(tk)).history(period=period, interval=interval, prepost=False, auto_adjust=False)
    if h is None or not len(h):
        raise RuntimeError(f'no {tf} data for {tk}')
    h = h.reset_index()
    t = pd.to_datetime(h.iloc[:, 0])
    t = t.dt.tz_convert('America/New_York').dt.tz_localize(None) if t.dt.tz is not None else t
    df = pd.DataFrame({'dt': t, 'open': h.Open.values, 'high': h.High.values, 'low': h.Low.values, 'close': h.Close.values}).dropna()
    if resample:
        x = df.set_index('dt')
        g = x.groupby([x.index.normalize(), pd.Grouper(freq='4h', origin='start_day', offset='9h30min')])
        r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna()
        r.index = r.index.get_level_values(1)
        df = r.reset_index().rename(columns={'index': 'dt'})
        df.columns = ['dt', 'open', 'high', 'low', 'close']
    df = df.sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    _bar_cache[key] = (time.time(), df)
    return df


def _vector(df):
    from pivot_ratchet_core import run_state_machine, BREAK_ATR
    if len(df) < 80:
        z = np.zeros(len(df))
        return z.astype(int), np.full(len(df), np.nan)
    reg, lv, atr, _ = run_state_machine(df[['open', 'high', 'low', 'close']].assign(datetime=df.dt))
    reg = np.asarray(reg).astype(int); lv = np.asarray(lv, float); atr = np.asarray(atr, float)
    line = np.where(reg == 1, lv - BREAK_ATR * atr, np.where(reg == -1, lv + BREAK_ATR * atr, np.nan))
    return reg, line


@app.route('/api/site/chart')
def site_chart():
    e = _need_login()
    if e: return e
    tk = request.args.get('ticker', 'SPY').strip().upper()
    tf = request.args.get('tf', '15m'); vtf = request.args.get('vtf', tf)
    if tf not in TF or vtf not in TF:
        return jsonify({'success': False, 'error': 'bad timeframe'}), 400
    try:
        C = _bars(tk, tf)
        V = C if vtf == tf else _bars(tk, vtf)
        reg, line = _vector(V)
        # vector state known at the close of each vector bar -> carried onto every chart bar that closes after it
        known = V.dt + pd.Timedelta(minutes=TF[vtf][3])
        S = pd.DataFrame({'known': known, 'reg': reg, 'line': line}).sort_values('known')
        close_t = C.dt + pd.Timedelta(minutes=TF[tf][3])
        m = pd.merge_asof(pd.DataFrame({'t': close_t}), S, left_on='t', right_on='known', direction='backward')
        C = C.iloc[-800:]; m = m.iloc[-800:]
        bars = [{'t': int(pd.Timestamp(d).timestamp()), 'o': round(float(o), 4), 'h': round(float(h), 4), 'l': round(float(l), 4), 'c': round(float(c), 4),
                 'r': int(r) if r == r else 0, 'v': (round(float(v), 4) if v == v else None)}
                for d, o, h, l, c, r, v in zip(C.dt, C.open, C.high, C.low, C.close, m.reg.values, m.line.values)]
        return jsonify({'success': True, 'ticker': tk, 'tf': tf, 'vtf': vtf, 'bars': bars, 'last': bars[-1]['c'] if bars else None})
    except Exception as ex:
        return jsonify({'success': False, 'error': str(ex)[:200]}), 400


# ---------------------------------------------------------------- quantile-regression THOD/TLOD (1h / 4h opening windows)
_qr = {'day': None, 'data': None, 'busy': False}


@app.route('/api/site/qr')
def site_qr():
    e = _need_login()
    if e: return e
    tk = request.args.get('ticker', 'QQQ').strip().upper()
    try:
        import qr_zones as QZ
    except Exception as ex:
        return jsonify({'success': False, 'error': f'QR model unavailable ({ex})'}), 400
    if tk not in QZ.TK:
        return jsonify({'success': False, 'error': f"QR bands are fitted for {', '.join(QZ.TK)} only"}), 400
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    day = now.normalize()
    if now.dayofweek >= 5 or now.hour * 60 + now.minute < 575:
        return jsonify({'success': False, 'error': 'QR bands are built from the 09:30 open - available on trading days after 09:35 ET'}), 400
    if _qr['day'] != day or _qr['data'] is None:
        try:
            QZ.update_bars()
            _qr['data'] = QZ.predict(day, refresh=False); _qr['day'] = day
        except Exception as ex:
            return jsonify({'success': False, 'error': f'QR model failed: {str(ex)[:160]}'}), 400
    d = (_qr['data'] or {}).get(tk)
    if not d:
        return jsonify({'success': False, 'error': 'no QR bands for today yet'}), 400
    return jsonify({'success': True, 'ticker': tk, 'qr': d, 'taus': QZ.TAUS})
