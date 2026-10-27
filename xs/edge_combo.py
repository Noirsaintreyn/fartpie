"""Oct 26 2026: combine the small edges into one score and use it for book size, ranking and sizing; plus 5m entries in good macro.
Components whose direction held in BOTH periods on admitted trades (xs/macro_lab.py, xs/book_lab.py mine), signs fixed, equal weights,
each as a percentile against the admitted 2019-21 trades (no look-ahead in the scaling):
  - asset managers' S&P net long (COT index; crowded = worse)      + VIX / VIX3M (stress = better)
  - SPY stretch above its 200-day (stretched = worse)               + trading days since last earnings (later in the quarter = better)
  + options ratio
macro score = mean of the first three; composite = mean of all five. 'good' = top third of the score (cut on 2019-21 admitted trades).
 A  size the book by the score: up to 8 or 10 positions at 1/6 when good (133% / 167%), or 8 x 1/8 (100%); controls: always 8 / 10 at 1/6
 B  rank by the composite (tie-break), and size each trade 1.5x / 1x / 0.5x by composite third
 C  5m entries (5m flip with the 1h bull, xs/five_min_lab.py e5_1h) only when macro is good, live entries otherwise - on the 25 names
    with 5m history, against live on the same names"""
import glob, numpy as np, pandas as pd
src = open('xs/macro_lab.py').read(); exec(src[:src.index("B = L[L.inbook]; rows = []")])
ss2 = ss.replace('def sim_cap(', 'def sim_cw(').replace("for e, x, r_, ww, cp in", "for e, x, r_, ww, cp, wn in").replace("Z.w.values, Z.cap.values)", "Z.w.values, Z.cap.values, Z.wn.values)").replace("pend.append((x, r_, ww / n))", "pend.append((x, r_, ww / wn))")
exec(ss2)
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
FX = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/fx/*.parquet'))[['tk', 'h1', 'entry', 'dse']]
L = L.merge(FX.drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left')
REF = L[L.inbook & (L.half == '19-21')]
def pct(c, sign):
    ref = np.sort(REF[c].dropna().values); v = L[c].values; p = np.searchsorted(ref, v) / len(ref); p = np.where(np.isnan(v), .5, p)
    return p if sign > 0 else 1 - p
comp = {'es_am': -1, 'vix_ts': 1, 'spy200': -1, 'dse': 1, 'ratio': 1}
P_ = {c: pct(c, s) for c, s in comp.items()}
L['macro'] = (P_['es_am'] + P_['vix_ts'] + P_['spy200']) / 3; L['comp'] = sum(P_.values()) / len(P_)
R0 = L[L.inbook & (L.half == '19-21')]
for c in ('macro', 'comp'):
    q = pd.qcut(L.loc[L.inbook, c].rank(method='first'), 3, labels=['low', 'mid', 'high'])
    g = L[L.inbook].groupby([q, 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100
    print(f'\nadmitted trades by {c} score third (avg trade %, n)'); print(g.round(3).to_string())
def show(nm, Z, f=None):
    f = f or sim_cw; R_ = [f(Z, a, b) for a, b in PER]; flag = '  <-- outside the luck band in both' if R_[0][0] > .60 and R_[1][0] > .30 else ''
    print(f'  {nm:<60}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + flag, flush=True)
print(f'\n{"":<62}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
z = L.copy(); z['cap'] = 6; z['wn'] = 6; show('live (6 x 1/6)', z)
print('A  book size by score')
for c in ('macro', 'comp'):
    good = L[c] > R0[c].quantile(2 / 3)
    for cap, wn, lab in ((8, 6, 'up to 8 x 1/6 (133%)'), (10, 6, 'up to 10 x 1/6 (167%)'), (8, 8, '8 x 1/8 (100%)')):
        z = L.copy(); z['cap'] = np.where(good, cap, 6); z['wn'] = np.where(good, wn, 6); show(f'{c} good: {lab}', z)
    z = L.copy(); z['cap'] = np.where(good, 8, np.where(L[c] < R0[c].quantile(1 / 3), 4, 6)); z['wn'] = 6; show(f'{c}: 8 slots good / 6 mid / 4 bad (at 1/6)', z)
for cap in (8, 10):
    z = L.copy(); z['cap'] = cap; z['wn'] = 6; show(f'control: always up to {cap} x 1/6 (no score)', z)
print('B  ranking and per-trade size by composite')
z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['ratio'] = z.comp; show('rank by composite', z)
z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['ratio'] = z.ratio + z.comp; show('ratio + composite tie-break', z)
t1, t2 = R0.comp.quantile([1 / 3, 2 / 3]); z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = np.where(z.comp > t2, 1.5, np.where(z.comp < t1, .5, 1.0)); show('size 1.5x / 1x / 0.5x by composite third', z)
print('C  5m entries only in good macro (25 names with 5m history)')
F5 = prep(glob.glob('xs/ext/f5/*.parquet')) if glob.glob('xs/ext/f5/*.parquet') else prep(glob.glob('/tmp/claude-0/ml/f5/*.parquet'))
mac = L.drop_duplicates('d').set_index('d').macro.sort_index()
F5['macro'] = mac.reindex(F5.d.values, method='ffill').values; g5 = F5.macro > R0.macro.quantile(2 / 3)
for nm, Z in (('live engine on the 25 names', F5[F5['var'] == 'vec']), ('5m entries always (e5_1h)', F5[F5['var'] == 'e5_1h']),
              ('5m entries only in good macro, live otherwise', pd.concat([F5[(F5['var'] == 'e5_1h') & g5], F5[(F5['var'] == 'vec') & ~g5]]))):
    z = Z.copy(); z['cap'] = 6; z['wn'] = 6; show(nm, z)
