"""Nov 4 2026: ICT context on the live VECTOR trades (AMT context was already tested Oct: value area / POC / VWAP / IB, ranker 18% / 8%, exits failed). DECLARED BEFORE RESULTS.
54 names, 17,447 live-rule trades, 15m bars premarket+RTH (ET). Everything known before the entry bar's open (entry = RTH open of bar j; state uses bars < j).
  DISC   discount: entry price below the 50% line of the PRIOR day's RTH range (ICT 'buy in discount')            DISC20 same for the 20-day dealing range
  BELOWO entry price below today's 09:30 open ('buy below the open' on a bullish day)                             KZ     entry 09:30-11:00 (NY AM kill zone)
  SWEEP  today's RTH low already took out the prior day's low before entry, and price is back above it (sell-side liquidity swept)
  PMSWP  same for the premarket low (the stock analog of an Asia / London low sweep)
  FVG    an unfilled bullish fair value gap (low[k] > high[k-2]) on 15m within the last 8 bars                    BIAS   prior day closed in the top half of its range
PART A per-trade avg / win / runner rate for each state, both halves. PART B book (40 matched runs): each as a filter (keep only the ICT-favourable state) and as a
rank bonus (ratio x 1.25), plus a count of favourable states x .1. PASS = >= 70% of runs beat live in both halves."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h.py').read(); exec(src[:src.index("if __name__ == '__main__':")])
e = open('xs/stop_1h_ev.py').read(); exec(e[:e.index("B0 = ev(XL)")])
feat = []
for tk, g in XL.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]); o, h, l, c, m = H.open.values, H.high.values, H.low.values, H.close.values, H.m.values; dts = H.dt.values; day = H.dt.dt.normalize().values
    rth = (m >= 570) & (m < 960)
    D = H[rth].groupby(H.dt.dt.normalize()[rth]).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
    D['h20'] = D.h.rolling(20).max(); D['l20'] = D.l.rolling(20).min()
    P = D.shift(1)                                                     # prior completed day, indexed by today
    pm = H[m < 570].groupby(H.dt.dt.normalize()[m < 570]).agg(pl=('low', 'min'))
    pos = pd.Series(np.arange(len(H)), index=dts)
    Lr = pd.Series(np.where(rth, l, np.inf)); cm = Lr.groupby(day).cummin().values; same = np.r_[False, day[1:] == day[:-1]]
    pre_lo = np.where(same, np.r_[np.inf, cm[:-1]], np.inf)                      # RTH low of today strictly before bar j
    op_day = D.o
    for i, r in g.iterrows():
        j = pos.get(r.entry)
        if j is None or np.ndim(j) or j < 20: continue
        j = int(j); d = pd.Timestamp(day[j])
        if d not in P.index or P.loc[d].isna().any(): continue
        p = P.loc[d]; px = o[j]
        rlo = pre_lo[j] if np.isfinite(pre_lo[j]) else np.nan; op = op_day.get(d, np.nan)
        pml = pm.pl.get(d, np.nan)
        fvg = 0.0
        for k in range(max(2, j - 8), j):
            if l[k] > h[k - 2] and l[k:j].min() > h[k - 2]: fvg = 1.0; break
        feat.append(dict(i=i, DISC=float(px < (p.h + p.l) / 2), DISC20=float(px < (p.h20 + p.l20) / 2), BELOWO=float(px < op) if op == op else np.nan,
                         KZ=float(570 <= m[j] < 660), SWEEP=float(rlo == rlo and rlo < p.l and px > p.l), PMSWP=float(pml == pml and rlo == rlo and rlo < pml and px > pml),
                         FVG=fvg, BIAS=float((p.c - p.l) / max(p.h - p.l, 1e-9) > .5)))
    print(tk, end=' ', flush=True)
F = pd.DataFrame(feat).set_index('i'); Z = XL.join(F); Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26'); Z['run'] = Z.ret >= .15
ks = ['DISC', 'DISC20', 'BELOWO', 'KZ', 'SWEEP', 'PMSWP', 'FVG', 'BIAS']
print(f'\ncoverage {Z.DISC.notna().mean():.0%}', flush=True)
print('\nPART A: per trade, ICT-favourable state (1) vs not (0): avg / win / runner rate / n', flush=True)
for k in ks:
    s = f'  {k:<7}'
    for p in ('19-21', '22-26'):
        for v in (1.0, 0.0):
            z = Z[(Z.per == p) & (Z[k] == v)]; s += f' | {p} {int(v)} {z.ret.mean()*100:+.2f}% w{z.ret.gt(0).mean():.0%} r{z.run.mean()*100:.1f}% n{len(z)}'
    print(s, flush=True)
B0 = ev(XL); print('\nPART B: 40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
for k in ks:
    keep = (Z[k] == 1) | Z[k].isna(); Y = XL[keep.values].reset_index(drop=True); line(f'filter {k} (drops {(~keep).mean():.0%})', ev(Y), B0)
    Y = XL.copy(); Y['ratio'] = XL.ratio * (1 + .25 * (Z[k] == 1).astype(float).values); line(f'rank {k} x1.25', ev(Y), B0)
n = sum((Z[k] == 1).astype(int) for k in ks); Y = XL.copy(); Y['ratio'] = XL.ratio * (1 + .1 * n.values); line('rank ratio x (1 + .1 n_ICT)', ev(Y), B0)
