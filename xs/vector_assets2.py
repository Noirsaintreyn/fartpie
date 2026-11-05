"""Nov 5 2026: ROUGH SCAN, take 2 - SESSION-ONLY 1h vector per asset (the 24h version failed everywhere, NQ included). DECLARED BEFORE RESULTS.
Each asset's main session in ET (hour of bar start): equity index 09-15 | metals 08-12 | energy 09-13 | FX 03-11 | rates 08-14 | livestock / grains 09-13 | bitcoin 09-15.
Vector built on session bars only; trade flips at any session bar close except the last; enter next bar open; target 1.5 x ATR14 / stop 1.0 x ATR14 (session 1h bars);
flat at the session's last bar close; cost 0.05 R. Also the 'always in' version: hold until the opposite flip (R = move / ATR at entry). NQ / ES are the calibration:
if the 1h session vector does not work on NQ either, this scan cannot speak for the 15m system and other assets need 15m history."""
import numpy as np, pandas as pd, sys, os, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
import yfinance as yf
CACHE = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/yf1h'; os.makedirs(CACHE, exist_ok=True)
EQ, MT, EN, FX, RT, AG_ = (9, 15), (8, 12), (9, 13), (3, 11), (8, 14), (9, 13)
AS = {'NQ=F': ('Nasdaq 100', EQ), 'ES=F': ('S&P 500', EQ), 'YM=F': ('Dow', EQ), 'RTY=F': ('Russell 2000', EQ), 'GC=F': ('Gold', MT), 'SI=F': ('Silver', MT), 'HG=F': ('Copper', MT),
      'CL=F': ('Crude oil', EN), 'NG=F': ('Nat gas', EN), '6E=F': ('Euro FX', FX), '6J=F': ('Yen', FX), '6B=F': ('Pound', FX), '6A=F': ('Aussie', FX), 'ZN=F': ('10y note', RT),
      'ZB=F': ('30y bond', RT), 'LE=F': ('Live cattle', AG_), 'GF=F': ('Feeder cattle', AG_), 'HE=F': ('Lean hogs', AG_), 'ZC=F': ('Corn', AG_), 'ZS=F': ('Soybeans', AG_),
      'ZW=F': ('Wheat', AG_), 'BTC=F': ('Bitcoin', EQ)}
def get(tk):
    p = f'{CACHE}/{tk.replace("=", "_")}.parquet'
    if os.path.exists(p): return pd.read_parquet(p)
    df = yf.download(tk, period='730d', interval='1h', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[['open', 'high', 'low', 'close']].dropna(); df.index = df.index.tz_convert('America/New_York').tz_localize(None); df.to_parquet(p); return df
def trades(df, h0, h1, dirs=None, always=False):
    x = df[(df.index.hour >= h0) & (df.index.hour <= h1) & (df.index.dayofweek < 5) & (df.high > df.low)].copy()
    o, h, l, c = (x[k].values for k in ('open', 'high', 'low', 'close')); r = np.asarray(vec(x[['open', 'high', 'low', 'close']])); day = x.index.normalize().values
    pc = np.r_[c[0], c[:-1]]; atr = pd.Series(np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])).rolling(14).mean().values
    out = []; busy = -1; n = 0
    for k in range(20, len(x) - 1):
        if r[k] == r[k - 1] or r[k - 1] == 0 or k + 1 <= busy or day[k + 1] != day[k] or not atr[k] > 0: continue
        d = int(r[k]) if dirs is None else dirs[n % len(dirs)]; n += 1; e = o[k + 1]; R = atr[k]; res = None
        for q in range(k + 1, len(x)):
            if always:
                if r[q] == -d or q == len(x) - 1: res = ((c[q] - e) * d / R, q); break
                continue
            if day[q] != day[k]: res = ((c[q - 1] - e) * d / R, q - 1); break
            adv = ((l[q] if d > 0 else h[q]) - e) * d / R; fav = ((h[q] if d > 0 else l[q]) - e) * d / R
            if adv <= -1.0: res = (-1.0, q); break
            if fav >= 1.5: res = (1.5, q); break
        if res is None: res = ((c[-1] - e) * d / R, len(x) - 1)
        out.append((x.index[k], res[0] - .05)); busy = res[1] if not always else k
    return pd.DataFrame(out, columns=['t', 'R'])
rng = np.random.default_rng(7); rows = []
for tk, (nm, (h0, h1)) in AS.items():
    try: df = get(tk)
    except Exception as ex: print(f'{tk} failed: {ex}', flush=True); continue
    A = trades(df, h0, h1); B = trades(df, h0, h1, always=True)
    if len(A) < 30: print(f'  {nm}: {len(A)} trades, skipped', flush=True); continue
    half = A.t < A.t.iloc[len(A) // 2]; t = A.R.mean() / (A.R.std() / np.sqrt(len(A))); cf = np.median([trades(df, h0, h1, rng.choice([-1, 1], 5000)).R.mean() for _ in range(20)])
    tb = B.R.mean() / (B.R.std() / np.sqrt(len(B)))
    rows.append((nm, A.R.mean(), A[half].R.mean(), A[~half].R.mean(), t, cf, B.R.mean(), tb))
    print(f'  {nm:<14} session {h0:02d}-{h1 + 1:02d} ET | 1.5R/1R: trades {len(A):>4} win {A.R.gt(0).mean():4.0%} avg {A.R.mean():+.3f} R (halves {A[half].R.mean():+.3f} / {A[~half].R.mean():+.3f}) coin-flip {cf:+.3f} t {t:+.1f} | always-in: {len(B)} flips avg {B.R.mean():+.3f} R t {tb:+.1f}', flush=True)
T = pd.DataFrame(rows, columns=['name', 'avg', 'h1', 'h2', 't', 'coin', 'avg_ai', 't_ai']).sort_values('t', ascending=False)
print('\nRANKED (1.5R / 1R exits) by t-stat:', flush=True)
for _, r in T.iterrows(): print(f'  {r["name"]:<14} avg {r.avg:+.3f} R  t {r.t:+.1f}  halves {r.h1:+.3f} / {r.h2:+.3f}  {"BOTH HALVES +" if r.h1 > 0 and r.h2 > 0 else ""} | always-in t {r.t_ai:+.1f}', flush=True)
