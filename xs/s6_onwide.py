"""Nov 10 2026: OVERNIGHT-RANGE SIZING ON THE FULL CURRENT SYSTEM (S6 = Rule A + M1 + R1 + ON+ + ON- + RULES(C+TPO) + V15a, with Yang-Zhang sizing, IV-jump skip and HAR relative sizing = V2).
DECLARED BEFORE RESULTS. Trade tables from xs/flex50.py variant(True, relm) (pts per trade at 1 contract after costs; sz = the live relative size, 0 = skipped).
WIDE = Globex range (18:00 -> 09:30, real NQ 5m) in the top third of its last 20 sessions (xs/nq_edge_hunt.py). The flag is known at 09:30, so it can only size engines that enter after 09:30
(RA, M1, R1, RULES, V15a); the overnight legs (ON+, ON-) enter before the range exists and are left alone. Data: NQ file ends 2026-01-30, so everything is cut there.
POINTS PER DAY: NQ points at 1 contract per engine, summed over engines, per trading day (all RTH days, flat days count as 0) - and the same weighted by the live size (sz).
VARIANTS on the RTH engines: BASE | NARROW x0.5 | SKIP narrow | WIDE x1.5 / NARROW x0.5 | WIDE x1.5 only.
REPORT: per engine wide vs narrow pts / trade (2018-21 | 2022-Jan26, t); system pts / day by period and year, share of days >= +7.5, losing days, months >= 50 pts, worst month."""
import pandas as pd, numpy as np, pickle
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); Dn = pd.read_pickle(f'{SP}/nq_days.pkl')
onr = Dn.OH - Dn.OL; WIDE = (onr > onr.shift(1).rolling(20).quantile(2 / 3)); DAYS = Dn.index[(Dn.index >= '2018-02-01') & (Dn.index <= '2026-01-30')]
RTH = ['RA', 'M1', 'R1', 'RULES', 'V15a']; T = {}
for (t, u, y), n in zip(OUT['V2'], NAMES):
    t = t.copy(); t['day'] = pd.to_datetime(t.day); t = t[(t.day >= '2018-02-01') & (t.day <= '2026-01-30')]; t['wide'] = WIDE.reindex(t.day).fillna(False).values; T[n] = t
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
print(f'days {len(DAYS):,} ({DAYS.min():%Y-%m-%d} - {DAYS.max():%Y-%m-%d}); wide overnight on {WIDE.reindex(DAYS).mean():.0%} of days\n')
print('=== per engine: trades taken (sz > 0) | avg pts per trade, wide vs narrow | 2018-21 | 2022-Jan26 (t of the difference) ===')
for n in NAMES:
    t = T[n]; t = t[t.sz > 0]; A = t.day < '2022-01-01'; L = []
    for lab, mm in (('18-21', A), ('22-26', ~A)):
        a, b = t.pts[mm & t.wide], t.pts[mm & ~t.wide]; L.append(f'{lab}: wide {a.mean():+6.1f} (n {len(a):3}) vs narrow {b.mean():+6.1f} (n {len(b):3}) t {tt(a, b):+5.2f}')
    print(f'  {n:<6} n {len(t):4} | ' + ' | '.join(L) + ('   [enters before 09:30: flag not usable]' if n in ('ON+', 'ON-') else ''))
MUL = {'BASE': (1, 1), 'NARROW x0.5': (1, .5), 'SKIP narrow': (1, 0), 'WIDE x1.5 / NARROW x0.5': (1.5, .5), 'WIDE x1.5 only': (1.5, 1)}
def daily(mul, weighted):
    s = pd.Series(0.0, index=DAYS)
    for n in NAMES:
        t = T[n]; t = t[t.sz > 0]; w, nr = mul if n in RTH else (1, 1); f = np.where(t.wide, w, nr)
        v = t.pts * f * (t.sz if weighted else 1.0); s = s.add(v.groupby(t.day.values).sum().reindex(DAYS).fillna(0.0), fill_value=0)
    return s
for weighted in (False, True):
    print(f'\n=== system points per day, {"weighted by the live size (sz)" if weighted else "1 contract per engine"} | all | 2018-21 | 2022-Jan26 | days >= +7.5 | losing days | months >= 50 pts | worst month ===')
    for k, mul in MUL.items():
        s = daily(mul, weighted); A = s.index < '2022-01-01'; mo = s.groupby(s.index.to_period('M')).sum()
        print(f'  {k:<26} {s.mean():+6.2f} | {s[A].mean():+6.2f} | {s[~A].mean():+6.2f} | {np.mean(s >= 7.5):4.0%} | {np.mean(s < 0):4.0%} | {np.mean(mo >= 50):4.0%} | {mo.min():+6.0f}', flush=True)
    s0, s1 = daily(MUL['BASE'], weighted), daily(MUL['WIDE x1.5 / NARROW x0.5'], weighted)
    print('   by year base -> wide x1.5/narrow x0.5: ' + ' | '.join(f'{y}: {s0[s0.index.year == y].mean():+.1f} -> {s1[s1.index.year == y].mean():+.1f}' for y in range(2018, 2027)))
print('\n=== per engine pts/day (1 contract) base | with WIDE x1.5 / NARROW x0.5 ===')
for n in NAMES:
    t = T[n]; t = t[t.sz > 0]; f = np.where(t.wide, 1.5, .5) if n in RTH else 1.0
    print(f'  {n:<6} {t.pts.groupby(t.day.values).sum().reindex(DAYS).fillna(0).mean():+5.2f} | {(t.pts * f).groupby(t.day.values).sum().reindex(DAYS).fillna(0).mean():+5.2f}')
