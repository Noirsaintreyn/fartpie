"""Options-flow context for the vector, ratio >= 2.5 names.  Flow = daily call share of option contracts (calls / all), known at the prior close:
level, and its z-score vs its own 60-day history (who's piling in: calls or puts).  Plus market regime = QQQ daily vector, and relative strength.
Longs: live model (sticky) trades.  Shorts: mirror of the 15m track (short at the RTH open after a bear flip, cover after the bull flip)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                       # RAT, last
# ---- call share
old = pd.read_csv('xs/ext/optvol_ratio.csv', usecols=['tk', 'date', 'call_c', 'opt_contracts'], parse_dates=['date'])
new = pd.concat(pd.read_csv(f, usecols=['tk', 'date', 'call_c', 'opt_contracts']) for f in glob.glob('data/theta_vol/*.csv')).dropna(subset=['date'])
new['date'] = pd.to_datetime(new.date); F = pd.concat([old, new]).drop_duplicates(['tk', 'date'], keep='last')
F = F[F.opt_contracts > 0]; F['cs'] = F.call_c / F.opt_contracts
CS = F.pivot_table(index='date', columns='tk', values='cs').sort_index()
CSZ = ((CS - CS.rolling(60, min_periods=30).mean()) / CS.rolling(60, min_periods=30).std()).shift(1); CS1 = CS.shift(1)
# ---- QQQ daily vector + 20d returns
def daily(tk):
    f = sorted(glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f)
    H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 570) & (m < 960)]; d = H.groupby(H.dt.dt.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
    return d
Q = daily('QQQ'); Q['reg'] = np.asarray(run_state_machine(Q.assign(datetime=Q.index).reset_index(drop=True))[0]); QREG = Q.reg.shift(1); Q20 = (Q.close / Q.close.shift(20) - 1).shift(1)
# ---- long trades
L = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); L = L[L['var'] == 'base+sticky'].copy()
# ---- short trades (mirror 15m track)
exec(open('xs/ah_sticky.py').read().split("if sys.argv[1] == '1':")[0].split('def run(')[0])     # bars()
def run_short(H):
    reg, o, c, lo, d, m = H.reg.values, H.open.values, H.close.values, H.low.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False; KR = [reg[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            if KR[p] == -1 and KR[p - 1] != -1: pos, e, px = True, j, o[j]
            continue
        if j == e: continue
        if KR[p] == 1: T.append((d[e], d[j], 1 - o[j] / px - 2e-4)); pos = False
    return T
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
SF = 'xs/ext/short_trades.parquet'
if not os.path.exists(SF):
    rows = []
    for tk in sorted(set(L.tk)):
        H = bars(fs[tk], 960); rows += [(tk,) + t for t in run_short(H)]
    pd.DataFrame(rows, columns=['tk', 'entry', 'exit', 'ret']).to_parquet(SF)
S = pd.read_parquet(SF)
def prep(T):
    T = T.copy(); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit); T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    d = T.entry.dt.normalize(); T['ratio'] = [RAT[t].asof(x) for t, x in zip(T.tk, d)]; T = T[T.ratio >= 2.5]
    d = T.entry.dt.normalize()
    T['cs'] = [CS1[t].asof(x) if t in CS1 else np.nan for t, x in zip(T.tk, d)]; T['csz'] = [CSZ[t].asof(x) if t in CSZ else np.nan for t, x in zip(T.tk, d)]
    T['qreg'] = QREG.reindex(d, method='ffill').values; T['q20'] = Q20.reindex(d, method='ffill').values
    T['per'] = np.where(T.entry < '2022-01-01', '19-21', '22-26'); return T
L, S = prep(L), prep(S)
def tab(T, col, bins, lab):
    b = pd.cut(T[col], bins); g = T.groupby([b, 'per']).ret.agg(['mean', 'count']).unstack('per')
    print(f'\n{lab}'); print((g['mean'] * 100).round(2).astype(str).add('%  n=').add(g['count'].astype(int).astype(str)).to_string())
print(f'LONGS  avg {L.ret.mean():+.2%} (19-21 {L[L.per=="19-21"].ret.mean():+.2%}, 22-26 {L[L.per=="22-26"].ret.mean():+.2%})   SHORTS avg {S.ret.mean():+.2%} (19-21 {S[S.per=="19-21"].ret.mean():+.2%}, 22-26 {S[S.per=="22-26"].ret.mean():+.2%})')
for nm, T in (('LONGS', L), ('SHORTS', S)):
    tab(T, 'cs', [0, .5, .55, .6, .65, 1], f'{nm} by call share of contracts (prior day)')
    tab(T, 'csz', [-9, -1.5, -0.5, 0.5, 1.5, 9], f'{nm} by call-share z vs own 60d (neg = puts piling in)')
    tab(T, 'qreg', [-2, -0.5, 0.5, 2], f'{nm} by QQQ daily vector (-1 bear / 1 bull)')
