"""Quantile-regression THOD/TLOD for the 1h (09:30-10:30) and 4h (09:30-13:30) windows + zone builder for the 09:30 charts.
Same model as xs/qr_window.py (validated Oct 2024 -> Oct 2026): pooled QuantReg over QQQ AAPL AMD MSFT NVDA on everything before today,
features known at 09:30 (HAR ranges, the window's own past ranges, gap, premarket range/position, prior return, day of week).
predict(day) -> {tk: {'1h': {'THOD': [q50, q80, q95], 'TLOD': [...], 'open': O}, '4h': {...}}}   (cached in xs/ext/qr_win_<day>.json)
zones(qr_tf, hl, issue_price, tf, color) -> (bands, xlines) for vchart.render: your k-sigma line + QR line merged into a band when they are
within 0.35 x your 1s->2s spacing, otherwise the QR line is drawn on its own (labelled QR1 / QR2)."""
import os, json, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS = os.getenv('QR_BARS') or (f'{ROOT}/xs/ext/qr_bars' if os.path.isdir(f'{ROOT}/xs/ext/qr_bars') else f'{ROOT}/live_data/qr_bars')   # website: git copy
QR_CACHE = os.getenv('QR_CACHE') or (f'{ROOT}/xs/ext' if os.path.isdir(f'{ROOT}/xs/ext') else '/tmp')
TK = ['QQQ', 'AAPL', 'AMD', 'MSFT', 'NVDA']; TAUS = [0.50, 0.80, 0.95]; WIN = {'1h': 630, '4h': 810}
def update_bars():
    """append the last few days of 15m bars (yfinance, incl. premarket) so today's open + premarket are in the history files"""
    import yfinance as yf
    for tk in TK:
        f = f'{BARS}/{tk}_15m.parquet'; H = pd.read_parquet(f)
        try:
            gap = (pd.Timestamp.utcnow().tz_localize(None) - pd.to_datetime(H.datetime).max()).days
            y = yf.Ticker(tk).history(period='5d' if gap < 4 else '59d', interval='15m', prepost=True)
            if len(y):
                y = y.reset_index(); t = pd.to_datetime(y.iloc[:, 0], utc=True).dt.tz_localize(None)
                N = pd.DataFrame({'datetime': t, 'open': y.Open, 'high': y.High, 'low': y.Low, 'close': y.Close, 'volume': y.Volume.clip(lower=1)})
                H['datetime'] = pd.to_datetime(H.datetime); H = pd.concat([H, N]).drop_duplicates('datetime', keep='last').sort_values('datetime')
                H.to_parquet(f, index=False)
        except Exception as e: print('qr bars update failed', tk, e)
def _bars(tk):
    H = pd.read_parquet(f'{BARS}/{tk}_15m.parquet'); H['dt'] = pd.to_datetime(H.datetime).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
    H = H[(H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt'); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H['d'] = H.dt.dt.normalize()
    if tk == 'QQQ': H = H[(H.m >= 570) & (H.m < 960)]                              # QQQ history is RTH-only: keep today's features on the same footing
    return H
def _daily(H, day):
    R = H[(H.m >= 570) & (H.m < 960)].groupby('d').agg(O=('open', 'first'), H=('high', 'max'), L=('low', 'min'), C=('close', 'last'), n=('open', 'size'))
    R = R[(R.n >= 20) | (R.index == day)]
    P = H[(H.m >= 240) & (H.m < 570)].groupby('d').agg(PH=('high', 'max'), PL=('low', 'min'))
    D = R.join(P); D['rng'] = np.log(D.H / D.L); F = pd.DataFrame(index=D.index)
    F['har1'] = D.rng.shift(1); F['har22'] = D.rng.shift(1).rolling(22).mean(); F['gap'] = np.log(D.O / D.C.shift(1)); F['agap'] = F.gap.abs()
    F['ret1'] = np.log(D.C.shift(1) / D.C.shift(2)); F['pm_rng'] = np.log(D.PH / D.PL).fillna(0); F['pm_hi'] = np.log(D.PH / D.O).fillna(0); F['pm_lo'] = np.log(D.O / D.PL).fillna(0)
    F['dow'] = D.index.dayofweek; return D[['O']].join(F).dropna(subset=['har22', 'gap'])
def predict(day=None, refresh=True):
    import statsmodels.api as sm
    day = pd.Timestamp(day or pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()).normalize()
    cache = f'{QR_CACHE}/qr_win_{day:%Y%m%d}.json'
    if os.path.exists(cache) and not refresh: return json.load(open(cache))
    BB = {tk: _bars(tk) for tk in TK}; OUT = {}
    for wn, end in WIN.items():
        rows = []
        for tk in TK:
            H = BB[tk]; D = _daily(H, day)
            W = H[(H.m >= 570) & (H.m < end)].groupby('d').agg(WH=('high', 'max'), WL=('low', 'min'))
            D = D.join(W, how='left'); D['wu'] = np.log(D.WH / D.O); D['wd'] = np.log(D.O / D.WL); D['wr'] = np.log(D.WH / D.WL)
            D['w1'] = D.wr.shift(1); D['w5'] = D.wr.shift(1).rolling(5).mean(); D['w22'] = D.wr.shift(1).rolling(22).mean(); rows.append(D.assign(tk=tk))
        X = pd.concat(rows).dropna(subset=['w22', 'har22']); X.index.name = 'd'; X = X.reset_index(); s = X.w22
        N = pd.DataFrame({'w1': X.w1 / s, 'w5': X.w5 / s, 'dh': X.har22 / s, 'h1': X.har1 / X.har22, 'ag': X.agap / s, 'g': X.gap / s, 'pr': X.pm_rng / s,
                          'ph': X.pm_hi / s, 'pl': X.pm_lo / s, 'r1': X.ret1 / s, 'ls': np.log(s), 'dw0': (X.dow == 0) * 1.0, 'dw4': (X.dow == 4) * 1.0}).fillna(0)
        A = sm.add_constant(N.values, has_constant='add'); tr = ((X.d < day) & X.wu.notna()).values; te = (X.d == day).values
        if not te.any(): continue
        for side, col in (('THOD', 'wu'), ('TLOD', 'wd')):
            z = (X[col] / s).values
            for t in TAUS:
                b = sm.QuantReg(z[tr], A[tr]).fit(q=t, max_iter=5000).params; q = np.maximum(A[te] @ b, 0) * s.values[te]
                for tk, o, v in zip(X.tk[te], X.O[te], q):
                    d_ = OUT.setdefault(tk, {}).setdefault(wn, {'THOD': [], 'TLOD': [], 'open': float(o)}); d_[side].append(float(o * np.exp(v if side == 'THOD' else -v)))
    json.dump(OUT, open(cache, 'w'), indent=1); return OUT
def zones(qr_tf, hl, issue_price, tf, color):
    """qr_tf: predict()[tk][tf]; hl: backend hod_lod response for that tf (centred at issue_price)"""
    bands, xlines = [], []
    if not qr_tf or not hl: return bands, xlines
    sc = issue_price / qr_tf['open']
    for side, key in (('THOD', 'hod'), ('TLOD', 'lod')):
        sp = abs(hl[key]['2std'] - hl[key]['1std'])
        for k in (1, 2):
            a = hl[key][f'{k}std']; q = qr_tf[side][k - 1] * sc
            if abs(a - q) <= 0.35 * sp: bands.append((min(a, q), max(a, q), color, f'{tf} {side} Z{k}'))
            else: xlines.append((q, f'{tf} QR{k}', color))
    return bands, xlines
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[2] == 'update': update_bars()
    print(json.dumps(predict(sys.argv[1] if len(sys.argv) > 1 else None), indent=1)[:1500])
