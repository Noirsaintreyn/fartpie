"""Nov 18 2026 report 324 (reuses 322 machinery). Combined screener: live vs ranker (3-signal, 4-signal incl. momentum leaders) with and without the sector cap. Original 322 header:  Idea from the sell-side reviews (GSAM Market Pulse via Rey's review: 'diversification should be judged by distinct earnings and cash-flow drivers'; names in different sectors/regions can share one capex cycle) and the L/S desk doc: do the 6 slots cluster on one driver, and does capping names per sector reduce drawdown without costing return? THEORY (declared): 56-67% of flips share a bar with other candidates (report 302), so slots fill with names that moved together (AI complex: tech+semi+comm); the book's drawdown is a common-driver event. Rule (set a priori, one parameter): at most C open positions from the same hand-map sector (ctx_rank SEC map, 54 names), C = 2 and 3, skipped entries not replaced out of order. Also an AI-cluster cap (tech+semi+comm together <= 3). Slot book, live ratio order, CAGR / DD / vol 2019-22 | 2023-26. Why it would fail: the sector map is coarse, and the winners are exactly the clustered AI names."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
_c = open('/home/claude/fartpie/xs/ctx_rank.py').read(); i = _c.index('SEC = dict('); SEC = eval(_c[i + 6:_c.index("\nX['sec']")].replace('\n', ' ').strip().replace('dict', 'dict'))  # parse the dict literal
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B = pd.read_parquet(SP + 'b302.parquet'); B = B[B.exit < END].copy(); B['h1'] = False
AI = {'tech', 'semi', 'comm'}
def taken_cap(N=6, brk=.08, cap=None, group=lambda tk: SEC.get(tk, tk)):
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for e, x, r, tk, ra in zip(T0.entry.values, T0.exit.values, T0.ret.values, T0.tk.values, T0.ratio.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o[0] > e]
        if len(op) >= N: continue
        if cap and sum(1 for o in op if o[1] == group(tk)) >= cap[group(tk)] if isinstance(cap, dict) else (cap and sum(1 for o in op if o[1] == group(tk)) >= cap): continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        op.append((x, group(tk))); pend.append((x, r)); out.append((tk, pd.Timestamp(e), pd.Timestamp(x), r, ra))
    return out
B['R'] = B.ret / B.vol20; B['MOM'] = B.momp >= 2 / 3; B['SIZE'] = B.ldvp >= 2 / 3; B['TREND'] = B.trend.astype(bool); B['HI52'] = B.h52b.astype(bool)
brank = B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values; ns3 = B[['TREND', 'HI52', 'SIZE']].sum(axis=1).values; ns4 = B[['TREND', 'MOM', 'HI52', 'SIZE']].sum(axis=1).values
def run(key, cap=None, group=lambda tk: SEC.get(tk, tk), N=6):
    global T0
    T0 = B.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable')
    tk_ = taken_cap(N, cap=cap, group=group); R = np.zeros(len(D)); Wt = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; Wt[a:b + 1] += w
    s = pd.Series(R * np.where(Wt > 1, 1 / Wt, 1.0), index=D); o = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; c = e.iloc[-1] ** (1 / y) - 1; o.append(f'{c:6.1%} / DD {(e / e.cummax() - 1).min():6.1%} / vol {z.std() * np.sqrt(252):4.1%} / Calmar {c / -(e / e.cummax() - 1).min():4.2f}')
    return ' | '.join(o)
G = {'tech', 'semi', 'comm'}; allg = sorted(set(SEC.values()))
def capd(g3, o3): return {k: (g3 if k in G else o3) for k in allg}
def yr(key, cap):
    global T0
    T0 = B.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_ = taken_cap(6, cap=cap); R = np.zeros(len(D)); Wt = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; R[a:b + 1] += (1 / 6) * ((1 + r) * (1 - 2e-4)) ** (1 / n) - 1 / 6; Wt[a:b + 1] += 1 / 6
    return pd.Series(R * np.where(Wt > 1, 1 / Wt, 1.0), index=D)
print('[2019-22 | 2023-26]  CAGR / maxDD / vol / Calmar. Growth = tech+semi+comm (declared before running); cap = max open names per sector')
V = [('live, no cap', brank, None), ('live, cap 3 all sectors', brank, 3), ('live, cap 3 NON-growth only (growth uncapped)', brank, capd(6, 3)), ('live, cap 3 on growth only', brank, capd(3, 6)), ('live, cap 2 non-growth only', brank, capd(6, 2)),
     ('ranker, no cap', brank - 2 * ns3, None), ('ranker, cap 3 non-growth only', brank - 2 * ns3, capd(6, 3))]
for nm, key, cap in V: print(f'{nm:48s}', run(key, cap))
# does concentration in growth help? mean flip return and sd when a same-sector name is already open at entry
B['sec'] = B.tk.map(SEC); B = B.sort_values('entry'); rows = []
for tk, g in B.groupby('tk'): pass
Bs = B.assign(e=pd.to_datetime(B.entry), x=pd.to_datetime(B.exit)); op = []
nsame = np.zeros(len(Bs)); 
for sec, g in Bs.groupby('sec'):
    ee = g.e.values; xx = g.x.values; 
    for j, (idx, e) in enumerate(zip(g.index, ee)): nsame[Bs.index.get_loc(idx)] = int(((ee < e) & (xx > e)).sum())
Bs['nsame'] = nsame; Bs['grow'] = Bs.sec.isin(G)
print('\nflip ret by number of same-sector flips already open at entry (all candidate flips, not slot-limited): mean % (n), train | test')
for gl, m in (('growth', Bs.grow), ('non-growth', ~Bs.grow)):
    for lab, c in (('0 open', Bs.nsame == 0), ('1-2 open', (Bs.nsame >= 1) & (Bs.nsame <= 2)), ('3+ open', Bs.nsame >= 3)):
        o = []
        for nm, t in (('tr', Bs.tr), ('te', ~Bs.tr)): r = Bs[t & m & c].ret; o.append(f'{100 * r.mean():+.2f} (n{len(r)})')
        print(f'  {gl:10s} {lab:9s}', ' | '.join(o))
