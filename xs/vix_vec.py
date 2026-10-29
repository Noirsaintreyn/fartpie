"""Oct 29 2026 sniff test (Yahoo keeps ~60 days of intraday bars): does the VIX's own 5m / 15m vector at entry separate good and bad flips?
Universe = the live screener's names. Signal = 15m vector bull flip (RTH bars), entry next open, exit next open after the 15m bear flip.
No slots (every flip counted) - signal-level only."""
import sys, os, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie'); os.chdir('/home/claude/fartpie'); import live_screener as LS
U = 'TSLA META AAPL MSFT COST AMD NVDA AMZN GS PLTR CVNA MU GOOGL LLY AVGO UNH BA COIN CRWD ORCL NFLX APP HOOD CAT BRK-B ADBE JPM'.split()
def get(tk, iv):
    d = yf.Ticker(tk).history(period='60d', interval=iv, prepost=False).reset_index()
    d['dt'] = pd.to_datetime(d.iloc[:, 0]).dt.tz_convert('America/New_York').dt.tz_localize(None)
    d = d.rename(columns=str.lower); d['volume'] = d.get('volume', 0)
    m = d.dt.dt.hour * 60 + d.dt.dt.minute; return d[(m >= 570) & (m < 960)][['dt', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
VX = {}
for iv in ('5m', '15m'):
    V = get('^VIX', iv); V['volume'] = 1.0; V['reg'] = LS.vec(V).reg.values
    ln = 5 if iv == '5m' else 15; VX[iv] = pd.Series(V.reg.values, index=V.dt + pd.Timedelta(minutes=ln)).sort_index()   # known at bar close
    VX[iv + '_px'] = pd.Series(V.close.values, index=V.dt + pd.Timedelta(minutes=ln)).sort_index()
rows = []
for tk in U:
    try: H = get(tk, '15m')
    except Exception as e: print(tk, e); continue
    if len(H) < 200: continue
    r = LS.vec(H).reg.values; o = H.open.values; t = H.dt.values
    k = 1
    while k < len(H) - 1:
        if r[k] == 1 and r[k - 1] != 1:
            e = k + 1; x = next((j + 1 for j in range(e, len(H) - 1) if r[j] == -1), None)
            if x is None: break
            te = pd.Timestamp(t[e]); rec = dict(tk=tk, t=te, ret=o[x] / o[e] - 1, bars=x - e)
            for iv in ('5m', '15m'):
                s = VX[iv][VX[iv].index <= te]; rec['vix' + iv] = s.iloc[-1] if len(s) else np.nan
            p = VX['15m_px'][VX['15m_px'].index <= te]; rec['vix_chg'] = p.iloc[-1] / p.iloc[-5] - 1 if len(p) > 5 else np.nan
            rows.append(rec); k = x
        k += 1
T = pd.DataFrame(rows).dropna(); T.to_parquet('/tmp/claude-0/vix/flips.parquet')
print(f'{len(T)} flips, {T.t.min():%b %d}..{T.t.max():%b %d}, all flips avg {T.ret.mean():+.2%}  win {(T.ret > 0).mean():.0%}')
lab = {-1: 'VIX vector bear (falling)', 1: 'VIX vector bull (rising)', 0: 'VIX vector flat'}
for iv in ('5m', '15m'):
    print(f'\nby VIX {iv} vector at entry')
    g = T.groupby('vix' + iv).ret.agg(n='size', avg='mean', med='median', win=lambda v: (v > 0).mean())
    g.index = [lab.get(int(i), i) for i in g.index]; print(g.assign(avg=lambda x: (x.avg * 100).round(2), med=lambda x: (x.med * 100).round(2), win=lambda x: (x.win * 100).round(0)).to_string())
print('\nby VIX change over the last hour (15m closes)')
q = pd.qcut(T.vix_chg, 3, labels=['VIX falling', 'flat', 'VIX rising'])
print(T.groupby(q).ret.agg(n='size', avg=lambda v: v.mean() * 100, win=lambda v: (v > 0).mean() * 100).round(2).to_string())
both = T[(T.vix5m == -1) & (T['vix15m'] == -1)]; print(f'\nboth 5m and 15m VIX bear: n={len(both)} avg {both.ret.mean():+.2%}   rest: n={len(T) - len(both)} avg {T.drop(both.index).ret.mean():+.2%}')
rng = np.random.default_rng(0); d = [T.ret.sample(len(both), random_state=int(s)).mean() for s in rng.integers(0, 1e9, 2000)]
print(f'random subsets of the same size: 5th..95th pct avg {np.percentile(d, 5):+.2%}..{np.percentile(d, 95):+.2%}')
