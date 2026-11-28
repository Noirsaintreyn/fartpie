"""Report 528 (Nov 27/28 2026): all-sector flip comparison + tags (Consensus #7: carry investor identity + clustered insider buying to the all-sector test; t hurdle 3.0).
DECLARED BEFORE RESULTS. Universe 1,504 flips (flips517, 2019-01..2026-02). Groups: ALL; TOPATR (ATRP > train(2019-21) 67th pct); HEALTH; HEALTH&TOPATR; CLUS (>=2 distinct insiders bought in 90d) all sectors; CLUS&TOPATR. Metrics by era (2019-21 / 2022 / 2023-26): n, trades/yr, reach, mean r, median r.
Test of a tag vs rest-of-universe in 2022+: reach difference z (two-prop), mean-r Welch t. Trials = 4 tag/group contrasts (TOPATR, HEALTH, CLUS, CLUS&TOPATR) vs rest; hurdle |t|>=3.0 AND same sign in train 2019-21.
HFPIPE (investor identity) NOT run all-sector: needs SEC filing pull via the Mac browser, bridge disconnected (disclosed); health-only n=8.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
F = pd.read_pickle(X + 'flips517.pkl').reset_index(drop=True); F['d'] = pd.to_datetime(F.d).astype('datetime64[ns]'); F['yr'] = F.d.dt.year
I = pd.read_pickle(X + 'insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P'].copy(); I['fd'] = pd.to_datetime(I.fd).astype('datetime64[ns]'); IG = {t: g for t, g in I.groupby('ISSUERTRADINGSYMBOL')}
F['CLUS'] = 0
for i, (tk, d) in enumerate(zip(F.tk, F.d)):
    g = IG.get(tk)
    if g is not None:
        w = g[(g.fd <= d) & (g.fd > d - pd.Timedelta(days=90))]; F.loc[i, 'CLUS'] = int(w.RPTOWNERCIK.nunique() >= 2)
tr = F.yr.between(2019, 2021); q = F.loc[tr, 'ATRP'].quantile(.67); F['TOPATR'] = (F.ATRP > q).astype(int); F['HEALTH'] = (F.sector == 'HEALTH').astype(int); F['HxT'] = F.HEALTH & F.TOPATR; F['CxT'] = F.CLUS & F.TOPATR
F['era'] = np.select([F.yr <= 2021, F.yr == 2022], ['2019-21', '2022'], '2023-26'); yrs = {'2019-21': 3, '2022': 1, '2023-26': 3.15}
def line(lab, m):
    s = F[m]; o = f'{lab:16}'
    for e in ('2019-21', '2022', '2023-26'):
        t = s[s.era == e]; o += f' | {e}: n{len(t):3d} ({len(t)/yrs[e]:4.1f}/yr) reach {t.hit.mean()*100 if len(t) else float("nan"):3.0f}% mean {t.r.mean()*100 if len(t) else float("nan"):5.1f} med {t.r.median()*100 if len(t) else float("nan"):5.1f}'
    print(o)
print('TOPATR cut ATRP >', round(q, 3))
for lab, m in (('ALL', F.r.notna()), ('TOPATR', F.TOPATR == 1), ('HEALTH', F.HEALTH == 1), ('HEALTH&TOPATR', F.HxT == 1), ('CLUS', F.CLUS == 1), ('CLUS&TOPATR', F.CxT == 1)): line(lab, m)
def contrast(lab, col):
    out = f'{lab:14}'
    for nm, m in (('train', F.yr.between(2019, 2021)), ('test2022+', F.yr >= 2022)):
        a, b = F[m & (F[col] == 1)], F[m & (F[col] == 0)]
        if len(a) < 5: out += f' | {nm}: n{len(a)} few'; continue
        p1, p2 = a.hit.mean(), b.hit.mean(); pp = (a.hit.sum() + b.hit.sum()) / (len(a) + len(b)); z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / len(a) + 1 / len(b)))
        t = (a.r.mean() - b.r.mean()) / np.sqrt(a.r.var() / len(a) + b.r.var() / len(b)); out += f' | {nm}: n{len(a)} reach {p1*100:.0f} vs {p2*100:.0f} z {z:.2f}; mean {a.r.mean()*100:.1f} vs {b.r.mean()*100:.1f} t {t:.2f}'
    print(out)
print('\ncontrasts vs rest (hurdle |t|>=3 & same sign in train):')
for lab, c in (('TOPATR', 'TOPATR'), ('HEALTH', 'HEALTH'), ('CLUS', 'CLUS'), ('CLUS&TOPATR', 'CxT')): contrast(lab, c)
