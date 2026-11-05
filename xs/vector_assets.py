"""Nov 5 2026: ROUGH SCAN - does the vector flip carry over to other futures? DECLARED BEFORE RESULTS.
Data: yfinance continuous front-month futures, 1h bars, last ~730 days (the most yfinance gives at 1h; 15m only goes back 60 days, so this is the 1h vector, not the 15m one).
Vector = signal_lab.vec on each asset's own 1h bars (break buffer scales with the asset's own ATR, so no tuning per asset).
Trade every flip: enter next bar open, target 1.5 x ATR14(1h), stop 1.0 x ATR14(1h) (the 75 / 50 shape), stop checked first inside a bar, exit after 8 bars if neither,
cost 0.05 R per trade. Report R per trade, first / second half, coin-flip direction (median of 20), and a t-stat. NQ and ES included as calibration."""
import numpy as np, pandas as pd, sys, os, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
import yfinance as yf
AS = {'NQ=F': 'Nasdaq 100', 'ES=F': 'S&P 500', 'YM=F': 'Dow', 'RTY=F': 'Russell 2000', 'GC=F': 'Gold', 'SI=F': 'Silver', 'HG=F': 'Copper', 'CL=F': 'Crude oil', 'NG=F': 'Nat gas',
      '6E=F': 'Euro FX', '6J=F': 'Yen', '6B=F': 'Pound', '6A=F': 'Aussie', 'ZN=F': '10y note', 'ZB=F': '30y bond', 'LE=F': 'Live cattle', 'GF=F': 'Feeder cattle', 'HE=F': 'Lean hogs',
      'ZC=F': 'Corn', 'ZS=F': 'Soybeans', 'ZW=F': 'Wheat', 'BTC=F': 'Bitcoin'}
def trades(df, dirs=None, T=1.5, S=1.0, hold=8):
    o, h, l, c = (df[x].values for x in ('open', 'high', 'low', 'close')); r = np.asarray(vec(df[['open', 'high', 'low', 'close']]))
    tr = np.maximum.reduce([h - l, np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])]); atr = pd.Series(tr).rolling(14).mean().values
    out = []; busy = -1; n = 0
    for k in range(20, len(df) - 1):
        if r[k] == r[k - 1] or r[k - 1] == 0 or k + 1 <= busy or not np.isfinite(atr[k]) or atr[k] <= 0: continue
        d = int(r[k]) if dirs is None else dirs[n % len(dirs)]; n += 1; e = o[k + 1]; R = atr[k]; res = None
        for q in range(k + 1, min(k + 1 + hold, len(df))):
            adv = ((l[q] if d > 0 else h[q]) - e) * d / R; fav = ((h[q] if d > 0 else l[q]) - e) * d / R
            if adv <= -S: res = (-S, q); break
            if fav >= T: res = (T, q); break
        if res is None: q = min(k + hold, len(df) - 1); res = ((c[q] - e) * d / R, q)
        out.append((df.index[k], res[0] - .05)); busy = res[1]
    return pd.DataFrame(out, columns=['t', 'R'])
rng = np.random.default_rng(5); rows = []
for tk, nm in AS.items():
    try:
        df = yf.download(tk, period='730d', interval='1h', progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[['open', 'high', 'low', 'close']].dropna(); df = df[(df.high > df.low)]
    except Exception as ex: print(f'{tk} failed: {ex}', flush=True); continue
    if len(df) < 1000: print(f'{tk}: only {len(df)} bars, skipped', flush=True); continue
    R = trades(df); half = R.t < R.t.iloc[len(R) // 2]; t = R.R.mean() / (R.R.std() / np.sqrt(len(R)))
    cf = [trades(df, rng.choice([-1, 1], 5000)).R.mean() for _ in range(20)]
    rows.append((nm, tk, len(df), len(R), R.R.gt(0).mean(), R.R.mean(), R[half].R.mean(), R[~half].R.mean(), np.median(cf), t))
    print(f'  {nm:<14} {tk:<6} bars {len(df):>5} flips {len(R):>4} win {R.R.gt(0).mean():4.0%} avg {R.R.mean():+.3f} R (1st half {R[half].R.mean():+.3f} / 2nd {R[~half].R.mean():+.3f}) | coin-flip {np.median(cf):+.3f} | t {t:+.1f}', flush=True)
T = pd.DataFrame(rows, columns=['name', 'tk', 'bars', 'n', 'win', 'avg', 'h1', 'h2', 'coin', 't']).sort_values('t', ascending=False)
print('\nRANKED by t-stat (|t| < 2 is indistinguishable from noise; both halves positive is the minimum to look further):', flush=True)
for _, r in T.iterrows(): print(f'  {r["name"]:<14} avg {r.avg:+.3f} R  t {r.t:+.1f}  halves {r.h1:+.3f} / {r.h2:+.3f}  {"BOTH HALVES +" if r.h1 > 0 and r.h2 > 0 else ""}', flush=True)
