"""Oct 27 2026: the market signals from xs/market_signal.py applied to the screener book (prior close values): no new entries, or 4 slots
instead of 6 (rest cash), when the market signal is bearish. Luck band ~+51..+60% (2019-21), ~+24..+30% (2022-26)."""
import numpy as np, pandas as pd
ms = open('xs/market_signal.py').read(); exec(ms[:ms.index("PER = [('2016-06-01'")])
MK = OUT
src = open('xs/macro_lab.py').read(); exec(src[:src.index("B = L[L.inbook]; rows = []")])
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
Q, S = MK['QQQ'].copy(), MK['SPY'].copy()
for V in (Q, S): V['sum'] = V[[c for c in V.columns if c != 'ret']].sum(axis=1)
def at(V, c): return V[c].reindex(L.d.values, method='ffill').values
sig = {'QQQ daily vector bear': at(Q, 'vec_d') < 0, 'QQQ 1h vector bear (prior close)': at(Q, 'vec_1h') < 0, 'SPY daily vector bear': at(S, 'vec_d') < 0,
       'SPY under its 200-day': at(S, 'sma200') < 0, 'QQQ vote sum < 0': at(Q, 'sum') < 0, 'SPY vote sum < 0': at(S, 'sum') < 0,
       'SPY vector bear AND under 200-day': (at(S, 'vec_d') < 0) & (at(S, 'sma200') < 0), 'vol cross bearish (QQQ)': at(Q, 'volx') < 0}
def show(nm, Z, cap=False):
    f = sim_cap if cap else sim; R_ = [f(Z, a, b) for a, b in PER]; flag = '  <-- outside the luck band in both' if R_[0][0] > .60 and R_[1][0] > .30 else ''
    print(f'  {nm:<58}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + flag, flush=True)
print(f'{"":<60}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
for nm, bad in sig.items():
    b = L[L.inbook]; bb = bad[L.inbook.values]
    g = b.groupby([np.where(bb, 'bear', 'bull'), 'half']).ret.mean().unstack() * 100
    print(f'  {nm}: share of signals {bad.mean():.0%}; admitted avg trade bear {g.loc["bear", "19-21"]:+.2f}% / {g.loc["bear", "22-26"]:+.2f}%  vs bull {g.loc["bull", "19-21"]:+.2f}% / {g.loc["bull", "22-26"]:+.2f}%')
    show(f'   no new entries when {nm}', L[~bad]); z = L.copy(); z['cap'] = np.where(bad, 4, 6); show(f'   4 slots when {nm}', z, cap=True)
print('\nflip it: more exposure when the market is bearish (leaders flipping up against a weak tape)')
ss2 = ss.replace('def sim_cap(', 'def sim_cw(').replace("for e, x, r_, ww, cp in", "for e, x, r_, ww, cp, wn in").replace("Z.w.values, Z.cap.values)", "Z.w.values, Z.cap.values, Z.wn.values)").replace("pend.append((x, r_, ww / n))", "pend.append((x, r_, ww / wn))")
exec(ss2)
def show2(nm, Z):
    R_ = [sim_cw(Z, a, b) for a, b in PER]; flag = '  <-- outside the luck band in both' if R_[0][0] > .60 and R_[1][0] > .30 else ''
    print(f'  {nm:<58}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + '  ret/DD ' + ' '.join(f'{c / -d:.1f}' for c, d in R_[:2]) + flag, flush=True)
z = L.copy(); z['cap'] = 6; z['wn'] = 6; show2('live', z)
for nm in ('SPY daily vector bear', 'SPY under its 200-day', 'SPY vector bear AND under 200-day', 'QQQ daily vector bear'):
    bad = sig[nm]
    z = L.copy(); z['cap'] = np.where(bad, 8, 6); z['wn'] = 6; show2(f'up to 8 x 1/6 when {nm}', z)
    z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = np.where(bad, 1.5, 1.0); show2(f'1.5x size when {nm}', z)
    z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = np.where(bad, 1.5, .85); show2(f'1.5x when {nm}, 0.85x otherwise (~same exposure)', z)
z = L.copy(); z['cap'] = 7; z['wn'] = 6; show2('control: always up to 7 x 1/6', z)
print('\nrobustness: 1.5x-type sizing when SPY is in a downtrend (vector bear AND under 200-day)')
Y = [(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01')) for y in range(2019, 2027)]
z0 = L.copy(); z0['cap'] = 6; z0['wn'] = 6; base = [sim_cw(z0, a, b)[0] for a, b in Y]
bad = sig['SPY vector bear AND under 200-day']
for k in (1.25, 1.5, 2.0):
    z = z0.copy(); z['w'] = np.where(bad, k, 1.0); yr = [sim_cw(z, a, b)[0] for a, b in Y]; show2(f'{k}x when SPY downtrend', z)
    print('      per year vs live: ' + ' '.join(f'{y}:{x - b_:+.0%}' for y, x, b_ in zip(range(2019, 2027), yr, base)))
for nm, b2 in (('SPY downtrend, vector bear for 3+ days', bad & (pd.Series(at(S, 'vec_d')).rolling(1).min().values < 0)),
               ('QQQ vector bear AND QQQ under its 200-day', (at(Q, 'vec_d') < 0) & (Q['close'].reindex(L.d.values, method='ffill').values < 0) if 'close' in Q else bad)):
    pass
q200 = (lambda c: (c / c.rolling(200).mean() - 1).shift(1))(etf('QQQ').close); qb = (at(Q, 'vec_d') < 0) & (q200.reindex(L.d.values, method='ffill').values < 0)
z = z0.copy(); z['w'] = np.where(qb, 1.5, 1.0); show2('1.5x when QQQ vector bear AND QQQ under 200-day', z)
bb = L[L.inbook]; m = bad[L.inbook.values]
print('  admitted downtrend trades by year (n, avg %):', ' '.join(f"{y}:{int(((bb.d.dt.year == y) & m).sum())}/{bb[(bb.d.dt.year == y) & m].ret.mean() * 100:+.1f}" for y in range(2019, 2027)))
