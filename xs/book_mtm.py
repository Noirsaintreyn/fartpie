"""book MTM daily returns for many variants (live engine trade files) -> vt/series.parquet"""
import numpy as np, pandas as pd, glob, os
from collections import deque
src = open('/tmp/claude-0/ml/stopbook.py').read(); exec(src[:src.index("S = pd.concat")])
def prep(fs):
    S = pd.concat(pd.read_parquet(f) for f in fs); S['d'] = S.entry.dt.normalize()
    S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].sort_values('d')
    S = pd.merge_asof(S, R, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)]; S['w'] = 1.0; return S
def taken(Z, n=6, brk=.08):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for tk, e, x, r_ in zip(Z.tk.values, Z.entry.values, Z.exit.values, Z.ret.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_)); out.append((tk, e, x, r_))
    return pd.DataFrame(out, columns=['tk', 'entry', 'exit', 'ret'])
BX = {}
def bars(tk):
    if tk not in BX:
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: BX[tk] = None; return None
        B = pd.read_parquet(f[0], columns=['datetime', 'open', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 240) & (m < 960)].drop_duplicates('dt').sort_values('dt')
        R_ = B[(B.dt.dt.hour * 60 + B.dt.dt.minute) >= 570]; dc = R_.groupby(R_.dt.dt.normalize()).close.last()
        BX[tk] = (B.set_index('dt').open, dc)
    return BX[tk]
def daily(T):
    P = {}
    for tk, e, x, r in T.itertuples(index=False):
        b = bars(tk)
        if b is None: continue
        op, dc = b; e, x = pd.Timestamp(e), pd.Timestamp(x)
        px = op.get(e); 
        if px is None or px != px: continue
        days = dc[(dc.index >= e.normalize()) & (dc.index < x.normalize())]
        v = [px] + list(days.values); end = px * (1 + r); idx = list(days.index) + [x.normalize()]
        vals = v[1:] + [end]
        for d_, a, b_ in zip(idx, v, vals): P[d_] = P.get(d_, 0) + (b_ / a - 1) / 6
    s = pd.Series(P).sort_index(); s = s.reindex(pd.bdate_range(s.index.min(), s.index.max()), fill_value=0.0); return s
if __name__ == '__main__':
    out = {}
    L = prep(glob.glob('/tmp/claude-0/ml/tm/*.parquet')); TM2 = prep(glob.glob('/tmp/claude-0/ml/tm2/*.parquet'))
    PB = prep(glob.glob('/tmp/claude-0/ml/pb/*.parquet')); RV = prep(glob.glob('/tmp/claude-0/ml/rv/*.parquet'))
    for src_, df in (('tm', L), ('tm2', TM2), ('pb', PB), ('rv', RV)):
        for v in df['var'].unique():
            out[f'{src_}:{v}'] = daily(taken(df[df['var'] == v])); print(v, flush=True)
    pd.DataFrame(out).fillna(0).to_parquet('/tmp/claude-0/ml/vt/series.parquet')
