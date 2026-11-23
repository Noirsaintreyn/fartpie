"""Nov 23 2026 report 394. Rey: rank flips that come AFTER accumulation/manipulation higher (use the phase as slot priority, not as a filter).  Live book (b302, 6 slots, 8% breaker, 2+2bp, ratio-first order) is the locked benchmark; ONLY the pick order among simultaneous candidates changes, so exposure/filters are identical.
Phase flags from report 393 (RTH flips; the 3,505 pre-market flips get score 0): P1 sweep of prior-day low, P2 opening-range-low sweep reclaimed, P5 weekly sweep + above week open, P6 textbook distribution (sweep and not in first 30 min).
Orders: PHASE-FIRST(P6), P1-first, P2-first, PHASE SCORE (P1+P2+P5+(1-P4)) first, MIXED (average of ratio rank and phase-score rank), LOW-RANGE-POSITION-first (P3 ascending), and the reverse of PHASE SCORE as a control.  Pass: Sharpe above live in BOTH halves."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; F = pd.read_parquet(SP + 'po3_393.parquet'); F['entry'] = pd.to_datetime(F.entry).astype('datetime64[ns]')
B0['entry'] = B0.entry.astype('datetime64[ns]'); B0 = B0.merge(F[['tk', 'entry', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']], on=['tk', 'entry'], how='left').reset_index(drop=True)
for c in ('P1', 'P2', 'P5', 'P6'): B0[c] = B0[c].fillna(0)
B0['P4'] = B0.P4.fillna(1); B0['P3'] = B0.P3.fillna(.5); B0['PS'] = B0.P1 + B0.P2 + B0.P5 + (1 - B0.P4); print('flips with phase info %d of %d' % (F.shape[0], len(B0)))
LIVE = order
def mk(fn):
    def f(Bs):
        k = fn(Bs); return Bs.assign(_k=k + 1e-6 * Bs.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable')
    return f
def run(lab, ordf):
    global order; order = ordf; TK, _ = sim_swap(B0, None); order = LIVE; z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print('%-30s trades %4d CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f' % (lab, len(TK), cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b) * 100, sharpe(b.values)), flush=True); return sharpe(a.values), sharpe(b.values)
base = run('LIVE (ratio first)', LIVE)
res = {'P6 first': run('P6 textbook first', mk(lambda B: -1000 * B.P6.values)), 'P1 first': run('P1 sweep PDL first', mk(lambda B: -1000 * B.P1.values)), 'P2 first': run('P2 OR sweep first', mk(lambda B: -1000 * B.P2.values)),
       'score first': run('phase score first', mk(lambda B: -1000 * B.PS.values)), 'mixed': run('mixed (avg of ranks)', lambda B: B.assign(_k=(B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values) * 0 + (B.ratio.rank(ascending=False, pct=True).values + B.PS.rank(ascending=False, pct=True).values) / 2).sort_values(['entry', '_k', 'tk'], kind='stable')),
       'low pos first': run('low range position first', mk(lambda B: 1000 * B.P3.values)), 'reverse score (control)': run('reverse phase score (control)', mk(lambda B: 1000 * B.PS.values))}
for k, v in res.items(): print('  %-26s beats live in both halves: %s' % (k, v[0] > base[0] and v[1] > base[1]))
