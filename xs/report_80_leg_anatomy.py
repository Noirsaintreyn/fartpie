"""Nov 14 2026 THE VECTOR AS A CLASSIFIER: HOW MUCH OF EACH MOVE IS LOST TO LATE ENTRY AND LATE EXIT? (Rey: returns are dictated by when the state flips relative to the move, not by trend; a perfect peak-catching classifier still loses if it flips after the move has happened or fades.) Declared before results. For every long flip (report_64 set: wide daily, 31.8k) define from hindsight: TROUGH = lowest low of the preceding bearish vector leg (previous flip through the flip bar), PEAK = highest high from the entry bar through the bar of the exit flip. Identity (log terms): ln(1+M) = ln(1+L) + ln(1+C) + ln(1+F), where M = peak/trough-1 is the full move available to a perfect classifier, L = entry/trough-1 is lost to late entry (entry = next open after the flip bar), C = exit/entry-1 is captured, F = peak/exit-1 is given back by late exit. Report per era and per move size bucket (M <20%, 20-50%, 50-100%, >=100%): medians of M, L, C, F, the share of ln(1+M) lost to entry, captured, and lost to exit (aggregate sums), days trough->entry and peak->exit, and the share of legs where C is negative. Then ask what separates legs with small L (early) from large L: flip-bar volume, gap-open above trigger (report_68). Costs ignored here (30bp would be subtracted from C)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['d0'] = pd.to_datetime(P.d0); D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); rows = []
for tk, g in P.groupby('tk'):
    B = D[tk]; B = B[~B.index.duplicated()]; n = len(B); o = B.open.values; h = B.high.values; l = B.low.values; ix = B.index; reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); ch = np.array([i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0])
    for pi, r in zip(g.index, g.itertuples()):
        i = ix.get_indexer([r.d0])[0]
        if i < 0: continue
        prev = ch[ch < i]; j = i + int(r.hold); e = i + 1; x = j + 1
        if len(prev) == 0 or x >= n: continue
        p = prev[-1]; tr = l[p:i + 1].min(); pk = h[e:j + 1].max(); ent = o[e]; ex = o[x]
        tr = min(tr, ent); pk = max(pk, ent, ex); rows.append(dict(i=pi, tk=tk, t=r.t, M=pk / tr - 1, L=ent / tr - 1, C=ex / ent - 1, F=pk / ex - 1, dte=e - (p + int(np.argmin(l[p:i + 1]))), dxp=x - (e + int(np.argmax(h[e:j + 1]))), hold=r.hold, BV=r.BV, HUGE=r.HUGE))
A = pd.DataFrame(rows); A['era'] = np.where(A.t < '2022-01-01', '2019-21', '2022-26Q1'); A['bucket'] = pd.cut(A.M, [0, .2, .5, 1.0, 100], labels=['M<20%', '20-50%', '50-100%', '>=100%']); lg = lambda x: np.log1p(x)
def rep(lab, z):
    if len(z) < 30: return
    tot = lg(z.M).sum(); print(f'  {lab:22s} n {len(z):5d} | median M {z.M.median()*100:5.1f}% L {z.L.median()*100:5.1f}% C {z.C.median()*100:+6.1f}% F {z.F.median()*100:5.1f}% | share of the move: late entry {lg(z.L).sum()/tot*100:4.0f}% captured {lg(z.C).sum()/tot*100:+5.0f}% late exit {lg(z.F).sum()/tot*100:4.0f}% | days trough->entry {z.dte.median():.0f} peak->exit {z.dxp.median():.0f} | C<0 in {(z.C<0).mean()*100:.0f}%')
print('ALL LONG LEGS BY ERA'); [rep(en, A[A.era == en]) for en in ('2019-21', '2022-26Q1')]
print('\nBY SIZE OF THE AVAILABLE MOVE (what a perfect classifier could have had)')
for en in ('2019-21', '2022-26Q1'):
    print(f' {en}'); [rep(str(b), A[(A.era == en) & (A.bucket == b)]) for b in A.bucket.cat.categories]
print('\nSCREENER RULE (BV>=1.5) vs HUGE legs (>=+100% captured)'); 
for en in ('2019-21', '2022-26Q1'):
    print(f' {en}'); rep('BV>=1.5', A[(A.era == en) & (A.BV >= 1.5)]); rep('HUGE legs', A[(A.era == en) & (A.HUGE == 1)]); rep('BV<1.5', A[(A.era == en) & (A.BV < 1.5)])
print('\nWHAT SEPARATES EARLY FROM LATE ENTRY? (L = share of the move already gone at entry; early = L<median L within era)')
for en in ('2019-21', '2022-26Q1'):
    z = A[A.era == en]; e = z[z.L / z.M < (z.L / z.M).median()]; lt = z[z.L / z.M >= (z.L / z.M).median()]
    print(f' {en}: early-half legs: median BV {e.BV.median():.2f}, mean C {e.C.mean()*100:+.2f}%, HUGE {e.HUGE.mean()*100:.2f}/100 | late-half: median BV {lt.BV.median():.2f}, mean C {lt.C.mean()*100:+.2f}%, HUGE {lt.HUGE.mean()*100:.2f}/100 | corr(L/M, C) {np.corrcoef(z.L / z.M, z.C)[0, 1]:+.2f}')
