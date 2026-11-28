"""Report 526 (Nov 27 2026): Rey's attachment #6 (Consensus): flat 90d insider flag washes out signal; re-specify: role-weighted, cluster-aware, value-conditioned, joint with PIPE.
DECLARED BEFORE RESULTS (all positive direction). From Form 4 open-market purchases (code P), known on filing date fd, window 90d before flip date d:
 ROLE  >=1 purchase by an Officer or Director (RPTOWNER_RELATIONSHIP contains Officer/Director; excludes pure 10% owners)
 CLUS  >=2 distinct reporting owners (RPTOWNERCIK) bought in the 90d
 BIG   total purchase value in 90d >= $25k
 ROLEBIG ROLE & BIG
 JOINT ROLE & PIPE (8-K 3.02 in prior 365d)
Universes: HEALTH flips (360) and ALL-sector flips (1,504; insider signals are industry-dependent per Shi 2026). Outcome: reach first, also mean/median of r. Train 2019-21 / test 2022+. Trials = 5 variants x 2 universes = 10 -> Bonferroni x10. Permutation null 5000 same-size random draws from the same universe's test flips.
PASS: test reach lift in + direction, p*10<0.05, train sign same.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
F = pd.read_pickle(X + 'flips517.pkl').reset_index(drop=True); F['d'] = pd.to_datetime(F.d).astype('datetime64[ns]')
I = pd.read_pickle(X + 'insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P'].copy(); I['fd'] = pd.to_datetime(I.fd).astype('datetime64[ns]')
print('relationship values:', I.RPTOWNER_RELATIONSHIP.astype(str).value_counts().head(8).to_dict())
rel = I.RPTOWNER_RELATIONSHIP.astype(str).str.lower(); I['role'] = rel.str.contains('officer|director'); I['val'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce')
IG = {t: g.sort_values('fd') for t, g in I.groupby('ISSUERTRADINGSYMBOL')}
K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K['filed'] = pd.to_datetime(K.filed); K['acc'] = pd.to_datetime(K.accept, utc=True, errors='coerce')
K['known'] = pd.to_datetime(np.where(K.acc.dt.hour < 20, K.filed, K.filed + pd.Timedelta(days=1))).astype('datetime64[ns]'); K = K[K['items'].astype(str).str.split(',').apply(lambda l: '3.02' in l)]; KG = {t: np.sort(g.known.values) for t, g in K.groupby('ticker')}
o = []
for tk, d in zip(F.tk, F.d):
    g = IG.get(tk); r = dict(ROLE=0, CLUS=0, BIG=0)
    if g is not None:
        w = g[(g.fd <= d) & (g.fd > d - pd.Timedelta(days=90))]
        if len(w): r = dict(ROLE=int(w.role.any()), CLUS=int(w.RPTOWNERCIK.nunique() >= 2), BIG=int(w.val.sum() >= 25000))
    a = KG.get(tk); r['PIPE'] = int(a is not None and (np.searchsorted(a, d.to_datetime64(), 'right') - np.searchsorted(a, (d - pd.Timedelta(days=365)).to_datetime64(), 'left')) > 0); o.append(r)
F = F.join(pd.DataFrame(o)); F['ROLEBIG'] = F.ROLE & F.BIG; F['JOINT'] = F.ROLE & F.PIPE; F['yr'] = F.d.dt.year; tr = F.yr.between(2019, 2021)
rng = np.random.default_rng(3)
def run(U, name):
    print(f'\n== {name}: n={len(U)} base test reach {U[~tr.loc[U.index]].hit.mean()*100:.0f}% mean {U[~tr.loc[U.index]].r.mean()*100:.1f} med {U[~tr.loc[U.index]].r.median()*100:.1f}')
    te = U[~tr.loc[U.index]]; hit = te.hit.astype(float).values
    for v in ('ROLE', 'CLUS', 'BIG', 'ROLEBIG', 'JOINT'):
        s = U[U[v] == 1]; t = s[~tr.loc[s.index]]; a = s[tr.loc[s.index]]
        if len(t) < 5: print(f'{v:8} n={len(s)} test n={len(t)} too few'); continue
        d = rng.choice(len(te), (5000, len(t))); p = (hit[d].mean(1) >= t.hit.mean()).mean()
        print(f'{v:8} n={len(s):3d} reach {s.hit.mean()*100:3.0f}% mean {s.r.mean()*100:5.1f} med {s.r.median()*100:5.1f} | train n{len(a)} reach {a.hit.mean()*100 if len(a) else float("nan"):.0f}% | test n{len(t)} reach {t.hit.mean()*100:.0f}% mean {t.r.mean()*100:.1f} med {t.r.median()*100:.1f} | p {p:.4f} x10 {min(1,p*10):.3f}')
run(F[F.sector == 'HEALTH'], 'HEALTH'); run(F, 'ALL sectors')
