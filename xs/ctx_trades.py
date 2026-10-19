"""Two-vector context test. Live model (15m + 1h tracks, exit C, LOCK5/0.5, premarket+RTH vector, trades at RTH opens) plus the
RTH-only 15m vector as context (state known at an RTH open = RTH vector after the last completed RTH bar).
modes: live | A = enter only if RTH vector also bull | B = also exit when the RTH vector flips bear | C = half size if RTH vector not bull at entry
Writes xs/ext/ctx_trades/<TK>.parquet.  Usage: python3 xs/ctx_trades.py TK [TK ...]"""
import sys, glob, os, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
def sm(x): return np.asarray(run_state_machine(x[['open', 'high', 'low', 'close']].assign(datetime=x.dt))[0])
def bars(f):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H['reg'] = sm(H)
    R = H[H.m >= 570].reset_index(); H['rr'] = np.nan; H.loc[R['index'].values, 'rr'] = sm(R); H['rr'] = H.rr.ffill().fillna(0)
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]; g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']
    s = pd.DataFrame({'known': r.dt + pd.Timedelta('1h'), 'r': sm(r)}).sort_values('known')
    H['s1'] = pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), s, left_on='t', right_on='known', direction='backward').r.fillna(0).values
    return H
def run(H, h1track, mode):
    reg, s1, rr, o, c, hi, d, m = H.reg.values, H.s1.values, H.rr.values, H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if m[j] >= 570]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; KRR = [rr[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig and (mode != 'A' or KRR[p] == 1):
                pos, e, px, b1, peak = True, j, o[j], False, o[j]; w = 0.5 if (mode == 'C' and KRR[p] != 1) else 1.0
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; h1 = c[j - 1] / px - 1 >= .02 and b1
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1))
        if mode == 'B' and KRR[p] == -1 and KRR[p - 1] != -1: ex = True
        pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - 2e-4, w, int(KRR[g.index(e)] if False else 0))); pos = False
    return T
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sys.argv[1:]:
    out = f'xs/ext/ctx_trades/{tk}.parquet'
    if os.path.exists(out) or tk not in fs: continue
    t0 = time.time(); H = bars(fs[tk])
    rows = [(tk, trk, md) + t[:4] for md in ('live', 'A', 'B', 'C') for trk in (False, True) for t in run(H, trk, md)]
    pd.DataFrame(rows, columns=['tk', 'h1', 'mode', 'entry', 'exit', 'ret', 'w']).to_parquet(out)
    print(tk, len(rows), f'{time.time() - t0:.0f}s', flush=True)
