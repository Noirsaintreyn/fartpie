"""Oct 28 2026 (GOOGL -7% after hours): how often did the live book hold a name through an earnings gap, what did it cost, and does
any pre-report exit rule beat holding? Live-engine trades (tm, var=vec), live book (6 slots, ratio tie-break, 8% breaker).
Gap day G = report day if before noon, else the next trading day. A trade 'holds through' if entry < G open and exit >= G open.
Variants (the trade is cut at the close before G; the slot frees then):  ALL  every report   GUARD  live guard (up >= 3% but < typical move)
 LOSE  only if the trade is red at that close   UP3  only if up >= 3%"""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
TDs = {}
def gapdays(tk):
    f = f'earn/{tk}.csv'
    if not os.path.exists(f): return None
    e = pd.read_csv(f)
    if not len(e): return None
    t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    b = bars(tk)
    if b is None: return None
    dc = b[1]; D = dc.index; G = []
    for tt in t:
        d = tt.normalize(); g = d if tt.hour < 12 else (D[D > d][0] if (D > d).any() else None)
        if g is not None and g in D: G.append(g)
    return sorted(set(G))
rows = []; mods = {k: [] for k in ('ALL', 'GUARD', 'LOSE', 'UP3')}
for x in L.itertuples():
    G = gapdays(x.tk); b = bars(x.tk)
    rec = x._asdict(); cut = None
    if G and b is not None:
        op, dc = b; e, xt = pd.Timestamp(x.entry), pd.Timestamp(x.exit); px = op.get(e)
        hit = [g for g in G if e < g + pd.Timedelta('9h30min') <= xt]
        if hit and px == px and px:
            g = hit[0]; D = dc.index; pv = D[D < g][-1]; c0 = dc[pv]; o1 = op.get(g + pd.Timedelta('9h30min'), np.nan)
            mv = []
            for gg in [z for z in G if z < g][-8:]:
                pp = D[D < gg]
                if len(pp) and (gg + pd.Timedelta('9h30min')) in op.index: mv.append(abs(op[gg + pd.Timedelta('9h30min')] / dc[pp[-1]] - 1))
            up = c0 / px - 1; typ = np.mean(mv) if len(mv) >= 4 else np.nan
            rows.append(dict(tk=x.tk, entry=e, g=g, up=up, gap=o1 / c0 - 1, ret=x.ret, typ=typ))
            cut = (pv + pd.Timedelta('15h59min'), up)
    for k in mods:
        r = dict(rec)
        if cut is not None:
            up = cut[1]; typ = rows[-1]['typ']
            go = k == 'ALL' or (k == 'LOSE' and up < 0) or (k == 'UP3' and up >= .03) or (k == 'GUARD' and up >= .03 and typ == typ and up < typ)
            if go: r['exit'], r['ret'] = cut[0], up
        mods[k].append(r)
E = pd.DataFrame(rows)
TK = taken(L); key = set(zip(TK.tk, TK.entry.astype('datetime64[ns]')))
E['taken'] = [(a, np.datetime64(b, 'ns')) in key for a, b in zip(E.tk, E.entry)]
T = E[E.taken]; T['yr'] = T.g.dt.year
print(f'names with earnings data: {L.tk.isin([f[5:-4] for f in glob.glob("earn/*.csv")]).mean():.0%} of trades')
print(f'book trades held through a report: {len(T)} of {len(TK)} ({len(T) / len(TK):.1%}), ~{len(T) / 7.8:.0f}/yr')
print(f'gap: mean {T.gap.mean():+.2%}  median {T.gap.median():+.2%}  <=-5%: {(T.gap <= -.05).sum()}  >=+5%: {(T.gap >= .05).sum()}  worst {T.gap.min():+.1%}  best {T.gap.max():+.1%}')
print(f'book P&L from the gaps (1/6 each): {T.gap.sum() / 6:+.1%} total over 2019-26')
print(T.groupby(pd.cut(T.up, [-1, 0, .03, .1, 9])).agg(n=('gap', 'size'), gap=('gap', 'mean'), worse5=('gap', lambda v: (v <= -.05).mean()), full=('ret', 'mean')).round(3))
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
print('\nbook  2019-21 / 2022-26  (return, max DD)')
def show(nm, Z):
    Z = pd.DataFrame(Z) if not isinstance(Z, pd.DataFrame) else Z
    print(f'  {nm:<8}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b_) for a, b_ in P)))
show('hold', L)
for k, v in mods.items(): show(k, v)
