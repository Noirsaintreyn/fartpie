"""V15a VECTOR-FLIP TRADES (full history): which entry-time conditions (volume profile, VWAP extension, z from open, YZ pct, HAR ratio; order flow in 2025 window) separate good flips from bad? Cuts on 2018-21 terciles; PASS = same sign & |t|>=2 in 2018-21 and 2022-26. Nov 10 2026."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, SIG, SI, OPEN
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
_s = open('/home/claude/fartpie/xs/dev_vp.py').read(); exec(_s[_s.index('C = .5; BIN = .25'):_s.index('rows = []')])
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); T = dict(zip(N, O['V2']))['V15a'][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day)
TG = pd.read_pickle(SC + 'tg_features.pkl'); TG.index = pd.to_datetime(TG.index).normalize()
rows = []
for r in T.itertuples():
    d = r.day; jb = int((int(r.m) - 570) // 5) - 1
    if d not in A or d not in SI or jb < 14: continue
    f = feats(d, jb, int(r.dir))
    if f is None: continue
    o_, h_, l_, c_, v_, vw_ = A[d][0]; s_ = OPEN[d] * SIG[SI[d]][-1]
    if not np.isfinite(s_) or np.isnan(c_[jb]): continue
    f.update(vwext=r.dir * (c_[jb] - vw_[jb]) / s_, zopen=r.dir * (c_[jb] - OPEN[d]) / s_)
    if d in TG.index: f.update(yz_pct=TG.loc[d, 'yz_pct'], har=TG.loc[d, 'har_ratio'], gap_rel=TG.loc[d, 'gap_rel'])
    rows.append(dict(day=d, m=int(r.m), pts=r.pts * r.sz if False else r.pts, **f))
R = pd.DataFrame(rows).dropna(subset=['pocd']); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B')
print(f'V15a trades with features: {len(R)} | 2018-21 {np.sum(R.era=="A")} | 2022-26 {np.sum(R.era=="B")} | mean pts {R.pts.mean():+.2f}')
def tstat(x): return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
FE = ['pocd', 'pocd1h', 'pocmig', 'vaw', 'outva', 'vwext', 'zopen', 'yz_pct', 'har', 'gap_rel']
print(f'{"feature":8} {"group":18} | {"n A":>4} {"A mean":>7} {"A t":>5} | {"n B":>4} {"B mean":>7} {"B t":>5} | verdict')
good = []
A_ = R[R.era == 'A']
for c in FE:
    col = R[c]
    if c == 'outva': G = {'beyond VA with': col == 1, 'inside VA': col == 0, 'beyond VA against': col == -1}
    else: q = A_[c].quantile([1 / 3, 2 / 3]).values; G = {'low third': col <= q[0], 'mid': (col > q[0]) & (col <= q[1]), 'high third': col > q[1]}
    for nm, sel in G.items():
        a = R[sel & (R.era == 'A')].pts; b = R[sel & (R.era == 'B')].pts
        ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
        print(f'{c:8} {nm:18} | {len(a):4d} {a.mean():+7.2f} {tstat(a):+5.1f} | {len(b):4d} {b.mean():+7.2f} {tstat(b):+5.1f} | {"PASS" if ok else ""}')
        if ok: good.append((c, nm))
print('\nPASS list:', good)
R.to_pickle(SC + 'vflip_feats.pkl')
