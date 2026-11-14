"""Nov 14 2026 TDA REGIME TEST (declared before results). Point cloud = last W=50 daily return vectors (9 LSE names, one 9-dim point per day, ending the PRIOR day); persistent homology H1 via ripser; landscape L1 norm = sum over bars of (death-birth)^2/4 (Gidea-Katz style). State = expanding (PIT) tercile of the norm, min 120 days. Question: does the norm separate good/bad regimes for the 15m vector flip (hold-to-bear, bp), same sign in 2018-21 and 2022+, AND does it add anything beyond realized vol and average pairwise correlation (norm residual after regressing on both, fit on 2018-21 only)? Flips = report_38 events. PASS = top-vs-bottom tercile same sign both eras, |t|>=2 in at least the pooled sample, and the residual version also holds."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from ripser import ripser
s38 = open('/home/claude/fartpie/xs/report_38_mtf_flip_baskets.py').read(); exec(s38[:s38.index("S={'A 15m")])
import glob
def daily(f): x = load(f); return x.close.groupby(x.index.normalize()).last()
DC = pd.DataFrame({f.split('/')[-1].split('.')[0]: daily(f) for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet'))}).dropna(); R = np.log(DC).diff().dropna(); W = 50
rows = []
for i in range(W, len(R) + 1):
    X = R.iloc[i - W:i].values; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    dg = ripser(X, maxdim=1)['dgms'][1]; l1 = float(np.sum((dg[:, 1] - dg[:, 0]) ** 2) / 4) if len(dg) else 0.0
    C = np.corrcoef(R.iloc[i - W:i].values.T); ac = (C.sum() - len(C)) / (len(C) * (len(C) - 1)); rv = float(R.iloc[i - 20:i].std().mean()) if i >= 20 else np.nan
    rows.append((R.index[i - 1], l1, ac, rv))
F = pd.DataFrame(rows, columns=['d', 'tda', 'corr', 'rv']).set_index('d'); F = F.shift(1).dropna()   # known at the prior close
print(f'days {len(F)} | level correlation with norm: corr {F.tda.corr(F["corr"]):+.2f}, rv {F.tda.corr(F.rv):+.2f}')
tr = F[F.index < '2022-01-01']; X = np.c_[np.ones(len(tr)), tr['corr'].values, tr.rv.values]; beta = np.linalg.lstsq(X, tr.tda.values, rcond=None)[0]
F['res'] = F.tda.values - np.c_[np.ones(len(F)), F['corr'].values, F.rv.values] @ beta
def pit_terc(s, minp=120):
    out = pd.Series(np.nan, index=s.index)
    for i in range(minp, len(s)):
        h = s.iloc[:i]; v = s.iloc[i]; q1, q2 = h.quantile([1 / 3, 2 / 3]); out.iloc[i] = 0 if v <= q1 else 2 if v >= q2 else 1
    return out
for c in ('tda', 'corr', 'rv', 'res'): F['T_' + c] = pit_terc(F[c])
Pj = P.copy(); Pj['d'] = pd.to_datetime(Pj.d); Pj = Pj.join(F[[c for c in F.columns if c.startswith('T_')]], on='d').dropna(subset=['T_tda'])
print(f'flips with a state: {len(Pj)}')
def rep(col, lab):
    print(f' {lab}')
    for ex in ('hold', 'r16'):
        line = f'   exit {ex:4s}'
        for e, (a, b) in list(ERA.items()) + [('pooled', ('2018-01-01', '2027-01-01'))]:
            s = Pj[(Pj.t >= a) & (Pj.t < b)].dropna(subset=[ex]); lo = s[s[col] == 0][ex]; mid = s[s[col] == 1][ex]; hi = s[s[col] == 2][ex]
            line += f' | {e}: low {lo.mean():+6.1f} mid {mid.mean():+6.1f} high {hi.mean():+6.1f} (n {len(lo)}/{len(mid)}/{len(hi)}) high-low t {tt(hi, lo):+.1f}'
        print(line)
for c, lab in (('T_tda', 'TDA landscape norm tercile'), ('T_res', 'TDA norm residual (beyond corr and vol) tercile'), ('T_corr', 'avg pairwise correlation tercile (benchmark)'), ('T_rv', 'realized vol tercile (benchmark)')): rep(c, lab)
