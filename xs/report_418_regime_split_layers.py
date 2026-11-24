"""Nov 24 2026 report 418. Rey: some models/analysis work in normal markets but not risk-off (2018 Q4, 2022); pooled tests can hide that.
DECLARED: RISK-ON = SPY close > its 200d SMA at the prior close (else RISK-OFF).  Trades = live book's 12,320 gated 15m flips with the live VECTOR exit (no time holds), net 4bp.  Layers (all known at the prior close): F40 (fundamentals pct >= .6, report 408), R (name Parkinson relief), M (VIX relief), HOT (theme: sector ETF top-3 20d RS and rising dollar volume, report 413), HOT&TOP (HOT and within-sector fundamentals pct >= .6).
For each regime separately: layer vs BASE (same regime) win rate, median, mean, month-clustered t of mean difference; halves train 2019-21 | test 2018+2022-26.
PASS (a layer 'works in risk-on'): beats BASE on win AND median in BOTH halves within risk-on, pooled risk-on |t| >= 2.  Risk-off reported, not required (cannot be judged if n < 150).  Layers = 5, so a lone pass at t ~2-2.5 is a lead, not a result."""
import numpy as np, pandas as pd
_t = open('xs/report_408_analysis_then_flip.py').read(); exec(_t[:_t.index("# 15m: trades + book")])
F15 = feats(B0); F15['entry'] = pd.to_datetime(F15.entry)
TH = pd.read_parquet(SP + 'theme413.parquet')[['tk', 'entry', 'hot', 'top']]; TH['entry'] = pd.to_datetime(TH.entry); F15 = F15.merge(TH, on=['tk', 'entry'], how='left')
M_ = pd.read_parquet(SP + 'macro_daily.parquet'); sp = M_[M_.symbol == 'SPY'].assign(d=lambda x: pd.to_datetime(x.ts).dt.normalize()).drop_duplicates('d').set_index('d').close.sort_index()
on = (sp > sp.rolling(200).mean()).shift(1); F15['on'] = on.reindex(F15.d0, method='ffill').values; F15 = F15[F15.on.notna()].copy(); F15['on'] = F15.on.astype(bool)
F15['n'] = F15.ret - 4e-4; e_ = F15.entry; F15['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test'); F15['mo'] = e_.dt.to_period('M'); F15['hot'] = F15.hot.fillna(False).astype(bool); F15['top'] = F15.top.fillna(False).astype(bool)
print('flips', len(F15), '| risk-on share', F15.on.mean().round(3), '| by half risk-on share', F15.groupby('half').on.mean().round(2).to_dict())
LAY = {'F40': lambda d: d.F40, 'R': lambda d: d.R, 'M': lambda d: d.M, 'HOT': lambda d: d.hot, 'HOT&TOP': lambda d: d.hot & d.top}
for reg, rn in ((True, 'RISK-ON'), (False, 'RISK-OFF')):
    B = F15[(F15.on == reg) & F15.has]; bm = B.groupby('mo').n.mean(); print(f'\n=== {rn}: n {len(B)} | BASE win {(B.n>0).mean()*100:.1f}% med {B.n.median()*100:.2f}% mean {B.n.mean()*100:.2f}% | train n {int((B.half=="train").sum())} win {(B[B.half=="train"].n>0).mean()*100:.1f} med {B[B.half=="train"].n.median()*100:.2f} | test n {int((B.half=="test").sum())} win {(B[B.half=="test"].n>0).mean()*100:.1f} med {B[B.half=="test"].n.median()*100:.2f}')
    print(f"{'layer':9s}{'n':>6s} | win% med% mean% | mean-BASE pp (t) | train win/med (vs base) | test win/med (vs base)")
    for nm, f in LAY.items():
        d = B[f(B).astype(bool)]
        if len(d) < 60: print(nm, len(d)); continue
        mm = (d.groupby('mo').n.mean() - bm.reindex(d.groupby('mo').n.mean().index)).dropna(); t = mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 3 else np.nan; s = f"{nm:9s}{len(d):6d} | {(d.n>0).mean()*100:5.1f} {d.n.median()*100:5.2f} {d.n.mean()*100:5.2f} | {(d.n.mean()-B.n.mean())*100:+.2f} ({t:+.1f}) |"
        for h in ('train', 'test'): x = d[d.half == h]; b = B[B.half == h]; s += f" {(x.n>0).mean()*100:4.1f}/{x.n.median()*100:5.2f} ({(x.n>0).mean()*100-(b.n>0).mean()*100:+.1f}/{(x.n.median()-b.n.median())*100:+.2f}) |" if len(x) >= 30 else ' n<30 |'
        print(s)
