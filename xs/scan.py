"""Nov 2 2026: broad pattern scan of everything known at entry (how the 'no buys after 2pm' rule was found), with a chance baseline.
Cells: every trait by within-year quintile (or category), every pair of traits by tercile x tercile.
A cell is a PATTERN if its mean trade differs from the rest with |t| >= 2.5 in BOTH 2019-21 and 2022-26, same sign, n >= 150 per period.
Chance baseline: the same scan on 20 copies with returns shuffled within each period (how many 'patterns' noise alone produces).
Plus a depth-3 decision tree (min leaf 400) fit on 2019-21, its leaves scored on 2022-26."""
import os, numpy as np, pandas as pd, warnings, itertools; warnings.filterwarnings('ignore')
src = open('/home/claude/fartpie/xs/opt_three.py').read(); exec(src.split("X['h'] = np.where")[0].replace("print(", "(lambda *a, **k: None)("))
V = pd.read_parquet('/tmp/claude-0/ml/vs_flips.parquet')[['appr', 'age', 'depth', 'hl', 'lh', 'reason', 'dC']]
N = pd.read_parquet('/tmp/claude-0/ml/nt_X.parquet').set_index(['tk', 'entry'])[['gskew', 'res_now', 'res1', 'res5', 'pjump', 'p_y_up']]
X = X.join(V.reindex(X.index)) if len(V) == len(X) else X
X = X.set_index(['tk', 'entry']).join(N).reset_index()
import json; SEC = json.load(open('/home/claude/fartpie/sp500_sectors.json')); X['sector'] = X.tk.map(SEC).fillna('other')
X['month'] = X.entry.dt.month; X['h1c'] = X.h1.astype(int); X['first_bar'] = (X.tod == 570).astype(int)
X['h'] = np.where(X.entry.dt.year <= 2021, 0, 1); X['yr'] = X.entry.dt.year
CONT = ['ratio', 'gap', 'runup', 'z20', 'prior', 'margin', 'tod', 'whip', 'tkrec', 'iv', 'term', 'opening', 'reach', 'stopd', 'appr', 'age', 'depth', 'dC',
        'gskew', 'res_now', 'res1', 'res5', 'p_y_up']
CAT = ['dow', 'qqq', 'h1c', 'first_bar', 'hl', 'lh', 'reason', 'pjump', 'sector', 'month']
B5 = {c: X.groupby('yr')[c].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) if s.notna().sum() > 50 else s * np.nan) for c in CONT}
B3 = {c: X.groupby('yr')[c].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if s.notna().sum() > 50 else s * np.nan) for c in CONT + ['ratio']}
for c in CAT: B5[c] = X[c].astype(str).where(X[c].notna())
for c in ['dow', 'qqq', 'h1c', 'first_bar', 'hl', 'reason']: B3[c] = X[c].astype(str).where(X[c].notna())
cells = []
for c, b in B5.items():
    for v in b.dropna().unique(): cells.append((f'{c}={v}' if c in CAT else f'{c} q{int(v) + 1}/5', (b == v).values))
pairs = list(itertools.combinations(B3.keys(), 2))
for a, c in pairs:
    for va in B3[a].dropna().unique():
        ma = (B3[a] == va).values
        for vc in B3[c].dropna().unique(): cells.append((f'{a} {"t" + str(int(va) + 1) if a in CONT else "=" + str(va)} & {c} {"t" + str(int(vc) + 1) if c in CONT else "=" + str(vc)}', ma & (B3[c] == vc).values))
H = X.h.values
def scan(R):
    out = []
    for nm, m in cells:
        ts = []
        for p in (0, 1):
            z = H == p; a = R[m & z]; b = R[~m & z]
            if len(a) < 150 or len(b) < 150: ts = None; break
            t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)); ts.append((t, a.mean(), len(a)))
        if ts and abs(ts[0][0]) >= 2.5 and abs(ts[1][0]) >= 2.5 and np.sign(ts[0][0]) == np.sign(ts[1][0]): out.append((nm, ts))
    return out
R = X.ret.values; real = scan(R)
print(f'{len(cells)} cells scanned ({len(B5)} traits single, {len(pairs)} pairs)', flush=True)
rng = np.random.default_rng(5); null = []
for k in range(20):
    Rs = R.copy()
    for p in (0, 1): idx = np.where(H == p)[0]; Rs[idx] = R[rng.permutation(idx)]
    null.append(len(scan(Rs)))
print(f'patterns found: {len(real)};  chance baseline (shuffled returns): median {np.median(null):.0f}, 95th pct {np.percentile(null, 95):.0f}, max {max(null)}', flush=True)
top = X.groupby('h').ret.transform(lambda s: s >= s.quantile(.95)).values
real.sort(key=lambda r: -min(abs(r[1][0][0]), abs(r[1][1][0])))
print('\npatterns (t-stat, avg trade, n per period 2019-21 / 2022-26; big-winner ratio):')
for nm, ts in real[:40]:
    m = dict(cells)[nm]; jr = [(top[m & (H == p)].sum() / top[H == p].sum()) / ((m & (H == p)).sum() / (H == p).sum()) for p in (0, 1)]
    print(f'  {nm:<46} t {ts[0][0]:+.1f} / {ts[1][0]:+.1f}   avg {ts[0][1]:+.2%} / {ts[1][1]:+.2%}   n {ts[0][2]} / {ts[1][2]}   big-winner ratio {jr[0]:.2f} / {jr[1]:.2f}')
from sklearn.tree import DecisionTreeRegressor, export_text
F = [c for c in CONT] + ['dow', 'qqq', 'h1c', 'first_bar', 'hl', 'lh', 'reason', 'pjump', 'month']
Xf = X[F].astype(float).fillna(-999); tr = H == 0
t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=400).fit(Xf[tr], X.ret.clip(-.1, .2)[tr])
lv = t.apply(Xf); print('\ndecision tree fit on 2019-21; leaves scored on unseen 2022-26:')
print(export_text(t, feature_names=F, decimals=2))
for l in np.unique(lv):
    a, b = X.ret[(lv == l) & tr], X.ret[(lv == l) & ~tr]
    print(f'  leaf {l}: 2019-21 avg {a.mean():+.2%} (n {len(a)})   2022-26 avg {b.mean():+.2%} (n {len(b)})')
print(f'  tree leaf-mean rank corr, 2019-21 fit vs 2022-26 outcome: {pd.Series([X.ret[(lv == l) & tr].mean() for l in np.unique(lv)]).corr(pd.Series([X.ret[(lv == l) & ~tr].mean() for l in np.unique(lv)]), method="spearman"):+.2f}')
