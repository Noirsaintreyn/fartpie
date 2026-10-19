"""VWAP Bollinger bands (session VWAP anchored 04:00, +-k volume-weighted std) on the live model, ratio >= 2.5, 6 slots, ratio tie-break.
1) where entries sit vs the bands -> avg trade by z bucket;  2) entry filters on z;  3) take-profit exit at +2 / +3 sigma (exit next open)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
exec(open('xs/ctx_trades.py').read().split('fs = {')[0])                      # bars(), run(), sm()
exec(open('xs/ctx_sim.py').read().split('def sim(')[0])                         # T (ctx trades), RAT
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
T = T[(T['mode'] == 'live') & (T.ratio >= 2.5)].copy(); names = sorted(set(T.tk))
def vwap_z(f):
    H = bars(f); v = H.volume.fillna(0).clip(lower=1)
    tp = (H.high + H.low + H.close) / 3; d = H.dt.dt.normalize()
    cv = v.groupby(d).cumsum(); vw = (tp * v).groupby(d).cumsum() / cv
    var = (v * tp ** 2).groupby(d).cumsum() / cv - vw ** 2; sd = np.sqrt(var.clip(lower=0))
    H['z'] = ((H.close - vw) / sd.replace(0, np.nan)).clip(-6, 6); return H
def run_tp(H, h1track, K):
    reg, s1, o, c, hi, d, m, z = H.reg.values, H.s1.values, H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values, H.z.values
    g = [j for j in range(1, len(H)) if m[j] >= 570]; Tr = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig: pos, e, px, b1, peak = True, j, o[j], False, o[j]
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; h1 = c[j - 1] / px - 1 >= .02 and b1
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1))
        pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if K and z[j - 1] == z[j - 1] and z[j - 1] >= K and c[j - 1] > px: ex = True
        if ex: Tr.append((d[e], d[j], o[j] / px - 1 - 2e-4)); pos = False
    return Tr
Z, TP = [], []
for tk in names:
    H = vwap_z(fs[tk]); zz = pd.Series(H.z.values, index=H.dt.values)
    x = T[T.tk == tk].copy(); x['z'] = [zz.asof(pd.Timestamp(e) - pd.Timedelta('1min')) for e in x.entry]; Z.append(x)
    for K in (2.0, 3.0):
        for h1t in (False, True):
            TP += [(tk, h1t, K, a, b, r) for a, b, r in run_tp(H, h1t, K)]
T = pd.concat(Z); TPd = pd.DataFrame(TP, columns=['tk', 'h1', 'K', 'entry', 'exit', 'ret'])
print('avg trade by entry z (price vs session VWAP, in VWAP sigmas), ratio>=2.5:')
b = pd.cut(T.z, [-9, -2, -1, 0, 1, 2, 3, 9]); print(T.groupby(b).ret.agg(['mean', 'count', lambda s: (s > 0).mean()]).round(4).rename(columns={'<lambda_0>': 'win'}).to_string())
def sim(Z, n=6):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
    for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
        op = [o for o in op if o > e]
        if len(op) < n: op.append(x); ev.append((x, r / n))
    return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
def stt(s):
    e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'))
def show(nm, X):
    res = [stt(sim(X[(X.entry >= a) & (X.entry < b)])) for a, b in P]
    print(f'  {nm:<36}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
print(f'\n{"6 slots":<38}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}')
show('live', T)
for nm, q in (('skip entries above +2 sigma', T.z <= 2), ('skip entries above +1 sigma', T.z <= 1), ('only entries above VWAP', T.z > 0), ('only entries below VWAP', T.z <= 0), ('skip entries below -1 sigma', T.z >= -1)):
    show(nm, T[q.fillna(True)])
RT = T[['tk', 'entry', 'ratio']].drop_duplicates(['tk', 'entry'])
for K in (2.0, 3.0):
    X = TPd[TPd.K == K].copy(); X['entry'] = pd.to_datetime(X.entry); X['exit'] = pd.to_datetime(X.exit)
    X['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(X.tk, X.entry)]; X = X[X.ratio >= 2.5]
    show(f'take profit at +{K:.0f} sigma band', X)
