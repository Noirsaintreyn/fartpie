"""Nov 1 2026: one score per flip from every trait that showed excess edge per trade in BOTH periods, fixed weights (+1 each, no fitting):
 dn     SPY downtrend at entry (daily vector bear AND under its 200-day; the A+ grade)
 pre    an earnings report 2-3 trading days ahead
 hi4    the 4h vector still bear at entry (early reversal)
 hiD    the daily vector still bear at entry
Score 0-4. Uses: (a) score first, ratio breaks ties   (b) ratio + score   (c) gate: when 4+ slots are full, a flip needs score >= 1."""
import os, glob, numpy as np, pandas as pd
from collections import deque
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
L = pd.read_parquet('/tmp/claude-0/ml/mtf_states.parquet').sort_values('entry').reset_index(drop=True); L['d'] = L.entry.dt.normalize()
S = pd.read_parquet('/mnt/user-data/uploads/fartpie/etf_daily/SPY.parquet'); S['dt'] = pd.to_datetime(S.datetime).dt.normalize(); S['volume'] = 1.0
V = LS.vec(S[['dt', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)); dn = ((V.reg.values == -1) & (S.close.values < S.close.rolling(200).mean().values))
DN = pd.Series(dn, index=S.dt.values).shift(1)                                   # known at the prior close
L['dn'] = DN.reindex(L.d.values, method='ffill').fillna(False).astype(bool).values
src2 = open('/home/claude/fartpie/xs/pre_earn_flip.py').read(); exec(src2[src2.index('k = np.full'):src2.index("L['k'] = k")]); L['k'] = k
L['pre'] = L.k.between(2, 3); L['hi4'] = L.v4h == -1; L['hiD'] = L.vD == -1
L['score'] = L[['dn', 'pre', 'hi4', 'hiD']].sum(axis=1)
TK = taken(L).merge(L[['tk', 'entry', 'dn', 'pre', 'hi4', 'hiD', 'score']], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('avg trade % (n) by score, trades the live book takes'); print((TK.groupby(['score', 'h']).ret.agg(['size', 'mean']).unstack('h') * [1, 1, 100, 100]).round(2).to_string())
for c in ('dn', 'pre', 'hi4', 'hiD'): print(f'  {c}: ' + '  '.join(f'{h} {"on" if o else "off"} {v * 100:+.2f}%' for (o, h), v in TK.groupby([c, 'h']).ret.mean().items()))
def simg(Z, a, b_, need=1, full=4, n=6, brk=.08):
    Z = Z[(Z.entry >= a) & (Z.entry < b_)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_, sc in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.score.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n or (len(op) >= full and sc < need): continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z, f=sim, **kw): print(f'  {nm:<44}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (f(Z, a, b_, **kw) for a, b_ in P)), flush=True)
print('\nbook (yearly return / max DD)'); sh('live', L); pr = L.ratio.groupby(L.entry.dt.year).rank(pct=True)
Z = L.copy(); Z['ratio'] = (L.score * 10 + pr).values; sh('score first, ratio breaks ties', Z)
Z = L.copy(); Z['ratio'] = (L.score / 4 + pr).values; sh('ratio + score (blend)', Z)
for full in (3, 4, 5):
    for need in (1, 2): sh(f'gate: {full}+ slots full -> need score >= {need}', L, simg, need=need, full=full)
