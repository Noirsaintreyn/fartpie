"""Thin client for the fartpie Flask backend (session-cookie auth for /api/data, public for the rest)."""
import os
import threading

import requests

BASE = os.environ.get('BACKEND_URL', 'http://localhost:5000').rstrip('/')
USER = os.environ.get('BACKEND_USER', '')
PASSWORD = os.environ.get('BACKEND_PASSWORD', '')
TIMEOUT = 110

ALIASES = {'NQ': 'NQ=F', 'MNQ': 'NQ=F', 'ES': 'ES=F', 'MES': 'ES=F', 'GC': 'GC=F', 'MGC': 'GC=F', 'CL': 'CL=F',
           'SI': 'SI=F', 'YM': 'YM=F', 'RTY': 'RTY=F', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD'}
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']


def norm_ticker(t):
    t = t.strip().upper().lstrip('$')
    return ALIASES.get(t, t)


def norm_tf(tf):
    tf = tf.strip().lower().replace('d', 'd')
    tf = {'60m': '1h', '240m': '4h', 'daily': '1d', 'd': '1d', 'h': '1h', '1hr': '1h', '4hr': '4h'}.get(tf, tf)
    if tf not in TIMEFRAMES:
        raise ValueError(f'timeframe must be one of {", ".join(TIMEFRAMES)}')
    return tf


class BackendDown(RuntimeError):
    pass


def _json(r):
    """parse a backend response; a non-JSON body means Render/gunicorn returned an error page (502 = worker crashed/restarting, often out of memory)"""
    try:
        return r.json()
    except ValueError:
        raise BackendDown(f'the fartpie backend is down or restarting (HTTP {r.status_code}). Try again in a minute or two; if it keeps happening the Render service needs a restart.')


class Backend:
    def __init__(self):
        self.s = requests.Session()
        self.lock = threading.Lock()
        self.logged_in = False

    def _login(self):
        r = self.s.post(f'{BASE}/api/login', json={'username': USER, 'password': PASSWORD}, timeout=30)
        j = _json(r)
        if r.status_code != 200 or not j.get('success'):
            raise RuntimeError(f'backend login failed ({r.status_code}): {str(j)[:120]}')
        self.logged_in = True

    def data(self, ticker, timeframe):
        """Full analysis (levels, price bars, vol regime). Auth-gated; re-logs in once on 401."""
        with self.lock:
            if not self.logged_in:
                self._login()
            for attempt in (0, 1):
                r = self.s.get(f'{BASE}/api/data', params={'ticker': ticker, 'timeframe': timeframe}, timeout=TIMEOUT)
                if r.status_code in (401, 403) and attempt == 0:
                    self._login()
                    continue
                break
        j = _json(r)
        if not j.get('success'):
            raise RuntimeError(j.get('error', f'backend error {r.status_code}'))
        return j

    def hod_lod(self, ticker, timeframe):
        """Level-constrained theoretical HOD/LOD (predicted + 1/2/3 sigma). Auth-gated."""
        with self.lock:
            if not self.logged_in:
                self._login()
            for attempt in (0, 1):
                r = self.s.get(f'{BASE}/api/level-constrained-hod-lod', params={'ticker': ticker, 'timeframe': timeframe}, timeout=TIMEOUT)
                if r.status_code in (401, 403) and attempt == 0:
                    self._login()
                    continue
                break
        j = _json(r)
        if not j.get('success'):
            raise RuntimeError(j.get('error', f'backend error {r.status_code}'))
        return j

    def expected_range(self, ticker):
        """Skew-t GJR (IV-blend for NQ/AAPL/AMD) range for TODAY; None if the instrument isn't supported."""
        try:
            r = self.s.get(f'{BASE}/api/macro-regime', params={'ticker': ticker}, timeout=TIMEOUT)
            j = _json(r)
            return j.get('expected_range') if j.get('success') else None
        except Exception:
            return None

    def target_probability(self, ticker, timeframe, direction, entry, stop, target, horizon_hours, paths=False):
        r = self.s.get(f'{BASE}/api/target-probability', timeout=TIMEOUT, params={
            'ticker': ticker, 'timeframe': timeframe if timeframe in ('1h', '4h', '1d') else '1h',
            'direction': direction, 'entry': entry, 'stop': stop, 'target': target, 'horizon_hours': horizon_hours,
            **({'include_sample_paths': 'true'} if paths else {})})
        j = _json(r)
        if not j.get('success'):
            raise RuntimeError(j.get('error', f'backend error {r.status_code}'))
        return j
