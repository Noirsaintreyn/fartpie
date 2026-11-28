"""Report 525 (Nov 27 2026): Rey's attachment #5 (Consensus): rank HEALTH flips on financing quality (hedge-fund PIPE neg, insider participation pos, discount neg, runway vs catalyst).
LIMIT (disclosed): PIPE investor identity / discount / catalyst dates need 8-K exhibit text; SEC full-text and exhibit fetch are not reachable here, so only proxies are testable:
 PIPE  = any 8-K Item 3.02 (unregistered equity sale) known in prior 365d (count too) -- dilution-financing proxy, investor identity unknown (declared NEGATIVE)
 INS   = open-market insider purchase (Form 4 code P) in prior 90d (declared POSITIVE; literature: insider participation certifies)
 RUN12 = latest-10-K cash runway >= 12 months at latest operating burn (declared POSITIVE; survives to catalyst)
 FQ    = INS + RUN12 - PIPE (composite, integers)
DECLARED BEFORE RESULTS. Universe: 360 HEALTH flips (flips517). Outcome: REACH (hit) first (Rey/lit: raise reach not just median), also mean/median of r. Train 2019-21 / test 2022+. Trials = 4 -> Bonferroni x4. Permutation null: 5000 same-size random subsets of health flips (reach and mean).
PASS: test reach lift >0 in direction declared with p*4<0.05 and same sign in train.
"""
import numpy as np, pandas as pd, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
F = pd.read_pickle(X + 'flips517.pkl'); H = F[F.sector == 'HEALTH'].copy().reset_index(drop=True); H['d'] = pd.to_datetime(H.d).astype('datetime64[ns]')
H = pd.merge_asof(H.sort_values('d'), FD[['tk', 'filed', 'RUNWAY', 'CASHTA']].rename(columns={'filed': 'ffd'}), left_on='d', right_on='ffd', by='tk', direction='backward', tolerance=pd.Timedelta(days=540)).reset_index(drop=True)
K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K['filed'] = pd.to_datetime(K.filed); K['acc'] = pd.to_datetime(K.accept, utc=True, errors='coerce')
K['known'] = pd.to_datetime(np.where(K.acc.dt.hour < 20, K.filed, K.filed + pd.Timedelta(days=1))).astype('datetime64[ns]'); K = K[K['items'].astype(str).str.split(',').apply(lambda l: '3.02' in l)]
KG = {t: np.sort(g.known.values) for t, g in K.groupby('ticker')}
I = pd.read_pickle(X + 'insider_P_hist.pkl'); I = I[(I.TRANS_CODE == 'P')]; I['fd'] = pd.to_datetime(I.fd).astype('datetime64[ns]'); IG = {t: np.sort(g.fd.values) for t, g in I.groupby('ISSUERTRADINGSYMBOL')}
def cnt(G, tk, d, days):
    a = G.get(tk)
    if a is None: return 0
    return int(np.searchsorted(a, d.to_datetime64(), side='right') - np.searchsorted(a, (d - pd.Timedelta(days=days)).to_datetime64(), side='left'))
H['PIPEn'] = [cnt(KG, t, d, 365) for t, d in zip(H.tk, H.d)]; H['PIPE'] = (H.PIPEn > 0).astype(int); H['INS'] = [int(cnt(IG, t, d, 90) > 0) for t, d in zip(H.tk, H.d)]
H['RUN12'] = np.where(H.RUNWAY.notna(), (H.RUNWAY >= 12).astype(float), np.nan); H['FQ'] = H.INS + H.RUN12.fillna(0) - H.PIPE
H['yr'] = H.d.dt.year; tr = H.yr.between(2019, 2021); te = ~tr
print('n', len(H), '| PIPE share', round(H.PIPE.mean(), 2), 'INS share', round(H.INS.mean(), 3), 'RUN12 cov', round(H.RUN12.notna().mean(), 2), 'RUN12=1', round(H.RUN12.mean(), 2), '| FQ dist', H.FQ.value_counts().sort_index().to_dict())
rng = np.random.default_rng(2); hit = H.hit.astype(float).values; rr = H.r.values; N = len(H)
def row(lab, m, sign):
    s = H[m.fillna(False)]; n = len(s)
    if n < 5: print(f'{lab:28} n={n} too few'); return
    o = f'{lab:28} n={n:3d} reach {s.hit.mean()*100:3.0f}% mean {s.r.mean()*100:5.1f} med {s.r.median()*100:5.1f}'
    for nm, mk in (('train', tr), ('test', te)):
        t = s[mk.loc[s.index]]; o += f' | {nm} n{len(t)} reach {t.hit.mean()*100 if len(t) else float("nan"):.0f}%'
    t = s[te.loc[s.index]]; pool = np.where(te)[0]; d = rng.choice(pool, (5000, len(t))); nul = hit[d].mean(1); obs = t.hit.mean()
    p = (nul >= obs).mean() if sign > 0 else (nul <= obs).mean(); print(o, f'| test reach perm p {p:.4f} x4 {min(1,p*4):.3f}')
print('base test: reach', round(H[te].hit.mean() * 100), 'mean', round(H[te].r.mean() * 100, 1), 'med', round(H[te].r.median() * 100, 1), 'n', te.sum())
row('PIPE=1 (3.02 in 365d)', H.PIPE == 1, -1); row('PIPE=0', H.PIPE == 0, +1)
row('INS=1 (open-mkt buy 90d)', H.INS == 1, +1); row('INS=0', H.INS == 0, -1)
row('RUN12=1 (>=12m)', H.RUN12 == 1, +1); row('RUN12=0 (<12m)', H.RUN12 == 0, -1)
row('FQ>=1', H.FQ >= 1, +1); row('FQ<=-1', H.FQ <= -1, -1); row('FQ==0', H.FQ == 0, +1)
