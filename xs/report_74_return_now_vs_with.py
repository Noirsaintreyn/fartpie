"""Nov 14 2026 RETURN NOW vs WITH THE NEW FILTERS (Rey: 'what's return now vs with it'). NOW = the screener's rule on the wide daily flips: flip-bar volume >=1.5x, all 31.8k flips 2019-2026Q1 (survivorship-biased, in-sample thresholds, 30bp round trip). WITH = (a) skip entries in TIGHTENING months (2y-yield 60d change in the upper expanding tercile, class known at month start; first 12 months allowed); (b) squeeze-regime BUYs only (DTC>=10 & high-short basket leading); (c) squeeze-hot ranked first when slots are contested (no filter); (d) insider buy + volume tilt; (e) insider+volume skipping TIGHTENING. Two views: PER-FLIP mean net by year (no slot noise, unlimited slots) and the K=10 book (10% slots, realized at exit, same-day ties by flip volume or hot-first). The K=10 book is unstable (report_66): read the per-flip view first."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_71_sector_and_si_change.py').read(); exec(src.split("def mc(S, m, base):")[0].replace("src = open('report_69_short_regime.py').read(); exec(src.split(\"print('PART A\")[0])", "src0 = open('report_69_short_regime.py').read(); exec(src0.split(\"print('PART A\")[0])"))
Y = pd.read_pickle(SC + 'regime_monthly.pkl'); x = Y.X5_2ychg60; cls = pd.Series('na', index=Y.index)
for i in range(len(Y)):
    h = x.iloc[:i].dropna()
    if len(h) >= 12: lo, hi_ = h.quantile([1 / 3, 2 / 3]); cls.iloc[i] = 'EASING' if x.iloc[i] <= lo else ('TIGHTENING' if x.iloc[i] >= hi_ else 'MID')
P['mo'] = P.t.dt.to_period('M'); P['cls'] = P.mo.map(cls).fillna('na'); P['yr'] = P.t.dt.year; P['HOT'] = P.SQ10 & P.LEAD
P['ent'] = P.t.dt.normalize(); P['ext'] = [e + pd.offsets.BDay(int(h)) for e, h in zip(P.ent, P.hold)]; P = P.sort_values('t').reset_index(drop=True)
def book(S, K=10, key='BV'):
    S = S.sort_values(['ent', key], ascending=[True, False]); opn = []; pnl = []
    for r in S.itertuples():
        keep = []
        for (xx, w, rr) in opn:
            if xx > r.ent: keep.append((xx, w, rr))
            else: pnl.append((xx, w * rr))
        opn = keep
        if len(opn) < K: opn.append((r.ext, 1 / K, r.rn))
    pnl += [(xx, w * rr) for (xx, w, rr) in opn]; return pnl
def annual(pnl):
    s = pd.DataFrame(pnl, columns=['d', 'c']).sort_values('d'); eq = 1.0; out = {}; yr = None; st = 1.0
    for d, c in zip(s.d, s.c):
        if d.year != yr:
            if yr is not None: out[yr] = eq / st - 1
            yr = d.year; st = eq
        eq *= (1 + c)
    out[yr] = eq / st - 1; return out
B = P[P.BV >= 1.5].copy(); B['hk'] = B.BV + 1000 * B.HOT
V = {'NOW: volume>=1.5x': (B, 'BV'), '+ skip TIGHTENING months': (B[B.cls != 'TIGHTENING'], 'BV'), '+ squeeze-hot only': (B[B.HOT], 'BV'), '+ squeeze-hot ranked first': (B, 'hk'), 'insider + volume': (B[B.NI >= 1], 'BV'), 'insider+vol, skip TIGHTENING': (B[(B.NI >= 1) & (B.cls != 'TIGHTENING')], 'BV')}
yrs = list(range(2019, 2027)); print('VIEW 1 PER-FLIP MEAN NET % BY ENTRY YEAR (n)'); print(f'{"variant":30s} ' + ' '.join(f'{y:>11d}' for y in yrs) + '   ALL      worst-yr')
for k, (S, _) in V.items():
    m = [S[S.yr == y].rn.mean() * 100 if (S.yr == y).sum() >= 10 else np.nan for y in yrs]; n = [(S.yr == y).sum() for y in yrs]
    print(f'{k:30s} ' + ' '.join(f'{(f"{a:+.1f}({b})" if a == a else f"-({b})"):>11s}' for a, b in zip(m, n)) + f'  {S.rn.mean()*100:+6.2f}% ({len(S)})  {np.nanmin(m):+.1f}')
print('\nVIEW 2 K=10 BOOK, compounded by year (2026 = Jan-Mar)'); print(f'{"variant":30s} ' + ' '.join(f'{y:>8d}' for y in yrs) + '  compounded  worst-yr  pos-yrs')
for k, (S, key) in V.items():
    a = annual(book(S, 10, key)); g = np.prod([1 + a.get(y, 0) for y in yrs]) - 1
    print(f'{k:30s} ' + ' '.join(f'{a.get(y, np.nan)*100:+7.1f}%' for y in yrs) + f'  {g*100:+8.0f}%  {min(a.values())*100:+7.1f}%  {sum(v>0 for v in a.values())}/{len(a)}')
print('\nstrict eras per-flip mean net % (n): ' ); 
for k, (S, _) in V.items(): print(f'  {k:30s} 2019-21 {S[S.yr<=2021].rn.mean()*100:+6.2f}% ({int((S.yr<=2021).sum())}) | 2022-26Q1 {S[S.yr>=2022].rn.mean()*100:+6.2f}% ({int((S.yr>=2022).sum())})')
