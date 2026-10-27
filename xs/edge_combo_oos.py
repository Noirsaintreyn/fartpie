"""Oct 26 2026: honest version of xs/edge_combo.py - the composite's ingredients and signs chosen on 2019-21 admitted trades ONLY
(every input with |t| >= 1.4 there, sign from 2019-21), then 2022-26 is a true out-of-sample test. Also return / drawdown so leverage
is not mistaken for edge."""
import glob, numpy as np, pandas as pd
exec(open('xs/edge_combo.py').read().split("R0 = L[L.inbook")[0])
FX = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/fx/*.parquet'))[['tk', 'h1', 'entry', 'dage', 'mkt20']]
PX = pd.read_parquet('xs/ext/px/feat.parquet')[['tk', 'h1', 'entry', 'opt_vol_trend']]
L = L.merge(FX.drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left').merge(PX.drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left')
REF = L[L.inbook & (L.half == '19-21')]
cands = ['es_am', 'es_lev', 'nq_am', 'nq_lev', 'vix_pct', 'vix_ts', 'credit', 'usd', 'rates', 'cu_au', 'oil', 'breadth', 'spy200', 'dse', 'ratio', 'dage', 'mkt20', 'opt_vol_trend']
sel = {}
for c in cands:
    x = REF[c]; ok = x.notna(); q = pd.qcut(x[ok].rank(method='first'), 3, labels=False); g = REF[ok].groupby(q).ret
    t = (g.mean().iloc[-1] - g.mean().iloc[0]) / np.hypot(g.std().iloc[0] / np.sqrt(g.size().iloc[0]), g.std().iloc[-1] / np.sqrt(g.size().iloc[-1]))
    if abs(t) >= 1.4: sel[c] = np.sign(t)
print('picked on 2019-21 only:', {k: int(v) for k, v in sel.items()})
P_ = {c: pct(c, s) for c, s in sel.items()}; L['oos'] = sum(P_.values()) / len(P_)
q = pd.qcut(L.loc[L.inbook, 'oos'].rank(method='first'), 3, labels=['low', 'mid', 'high'])
g = L[L.inbook].groupby([q, 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100
print('admitted trades by out-of-sample composite third'); print(g.round(3).to_string())
R0 = L[L.inbook & (L.half == '19-21')]; t1, t2 = R0.oos.quantile([1 / 3, 2 / 3])
def show2(nm, Z):
    R_ = [sim_cw(Z, a, b) for a, b in PER]
    print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + '   return/DD ' + ' '.join(f'{c / -d:.1f}' for c, d in R_[:2]), flush=True)
print(f'\n{"":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
z = L.copy(); z['cap'] = 6; z['wn'] = 6; show2('live', z)
for lev in (7, 8): z = L.copy(); z['cap'] = lev; z['wn'] = 6; show2(f'control: always up to {lev} x 1/6', z)
z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = np.where(z.oos > t2, 1.5, np.where(z.oos < t1, .5, 1.0)); show2('size 1.5x / 1x / 0.5x by oos composite', z)
z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = np.where(z.oos > t2, 1.25, np.where(z.oos < t1, .75, 1.0)); show2('size 1.25x / 1x / 0.75x by oos composite', z)
z = L.copy(); z['cap'] = np.where(z.oos > t2, 8, 6); z['wn'] = 6; show2('up to 8 x 1/6 when oos composite good', z)
z = L.copy(); z['cap'] = np.where(z.oos > t2, 8, np.where(z.oos < t1, 4, 6)); z['wn'] = 6; show2('8 good / 6 mid / 4 bad', z)
