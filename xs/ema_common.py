"""Common prelude for the EMA tests (Nov 4 2026): per-ticker 15m bars (premarket+RTH as the backtest) with aligned indicator states known at each bar's open (no look-ahead):
 e21, e55 (15m EMA 21/55 = the vector's own), c1/h20/h50 (last COMPLETED 1h bar's close, EMA20, EMA50), d_c/d10/d20/dhi... (last completed RTH day: close, EMA10, EMA20, Donchian highs/lows, ATR14)"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
def ema_(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
BAR = {}
def prep_tk(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: return None
    H = load(f[0]).reset_index(drop=True); t = H.dt.values; o, h, l, c = H.open.values, H.high.values, H.low.values, H.close.values; m = H.m.values
    B = dict(t=t, o=o, h=h, l=l, c=c, m=m); B['e21'] = ema_(c, 21); B['e55'] = ema_(c, 55)
    # 1h bars (aligned :30), known at bar start + 1h
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]; r = x.resample('1h', offset='30min').agg(dict(open='first', high='max', low='min', close='last')).dropna()
    r['e20'] = ema_(r.close.values, 20); r['e50'] = ema_(r.close.values, 50); r['known'] = r.index + pd.Timedelta('1h')
    # supertrend(10,3) on 1h
    hi, lo, cl = r.high.values, r.low.values, r.close.values; pc = np.r_[cl[0], cl[:-1]]; tr = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc))); atr = pd.Series(tr).ewm(alpha=1 / 10, adjust=False).mean().values
    mid = (hi + lo) / 2; ub, lb = mid + 3 * atr, mid - 3 * atr; fu, fl = ub.copy(), lb.copy(); st = np.ones(len(r), int)
    for i in range(1, len(r)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]; fl[i] = lb[i] if (lb[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
        st[i] = 1 if cl[i] > fu[i - 1] else (-1 if cl[i] < fl[i - 1] else st[i - 1])
    r['st'] = st; r['cross'] = (r.e20 > r.e50).astype(int)
    K = pd.merge_asof(pd.DataFrame({'q': t + np.timedelta64(15, 'm')}), r[['known', 'close', 'e20', 'e50', 'st', 'cross']].sort_values('known'), left_on='q', right_on='known', direction='backward')
    B['c1'], B['h20'], B['h50'], B['st1'], B['x1'] = K.close.values, K.e20.values, K.e50.values, K.st.values, K.cross.values
    # daily (RTH) known at 16:00
    R = H[(H.m >= 570) & (H.m < 960)]; dd = R.groupby(R.dt.dt.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
    dd['e10'] = ema_(dd.c.values, 10); dd['e20'] = ema_(dd.c.values, 20); dd['e50'] = ema_(dd.c.values, 50); pcl = dd.c.shift(1); tr = np.maximum(dd.h - dd.l, np.maximum((dd.h - pcl).abs(), (dd.l - pcl).abs())); dd['atr'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    for n in (10, 20, 40, 80): dd[f'hi{n}'] = dd.h.shift(1).rolling(n).max(); dd[f'lo{n}'] = dd.l.shift(1).rolling(n).min()
    dd['known'] = dd.index + pd.Timedelta(hours=16)
    D = pd.merge_asof(pd.DataFrame({'q': t + np.timedelta64(15, 'm')}), dd.reset_index(drop=True).sort_values('known'), left_on='q', right_on='known', direction='backward')
    for k in ('c', 'e10', 'e20', 'e50', 'atr', 'hi10', 'hi20', 'hi40', 'hi80', 'lo10', 'lo20', 'lo40', 'lo80'): B['d_' + k] = D[k].values
    B['pos'] = pd.Series(np.arange(len(H)), index=t); return B
for tk in sorted(XL.tk.unique()):
    b = prep_tk(tk)
    if b is not None: BAR[tk] = b
print('bars prepared for', len(BAR), 'tickers', flush=True)
