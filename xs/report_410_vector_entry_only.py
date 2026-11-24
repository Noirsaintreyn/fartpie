"""Nov 23 2026 report 410. Rey: 'full stack then enter off the vector; the vector is ONLY the entry trigger (fresh flip), not the exit'.
DECLARED: entries = the live book's 12,320 gated 15m flips (54 names, PIT S&P, ratio >= 1.5) at the CLOSE of the flip day (no intraday look-ahead); NO vector exit.  Exit = fixed hold H in {5, 21, 63} trading days at the close, with a wide -8% close-based hard stop; cost 4bp.
Arms (analysis layers at the prior close, report 408 definitions): BASE | F40 | R | F40&R | F40&R&M.
H_A: stack arms show a higher win rate AND median than BASE in both halves.  H_B (vector value as an entry trigger): the same trades entered on a random day 1-20 trading days away (same name, same flags carried, 20 random draws) do worse than the flip entry: flip-minus-control mean > 0 in both halves, month-clustered |t| >= 2.
train 2019-21 | test 2018+2022-26."""
import numpy as np
_t = open('xs/report_408_analysis_then_flip.py').read(); exec(_t[:_t.index("# 15m: trades + book")])
CL = {tk: g.sort_values('day').set_index('day').close for tk, g in D1.groupby('symbol')}; CL = {k: v[~v.index.duplicated()] for k, v in CL.items()}
F15 = feats(B0); F15['de'] = pd.to_datetime(F15.entry).dt.normalize(); F15 = F15[F15.has].reset_index(drop=True)
def path_ret(tk, pos, H, stop=-.08, cost=4e-4):
    c = CL[tk].values
    if pos < 0 or pos + H >= len(c): return np.nan
    p = c[pos + 1:pos + H + 1] / c[pos] - 1; hit = np.where(p <= stop)[0]; return (p[hit[0]] if len(hit) else p[-1]) - cost
rng = np.random.default_rng(410); POS = np.array([CL[tk].index.searchsorted(d) for tk, d in zip(F15.tk, F15.de)])
for H in (5, 21, 63):
    F15[f'r{H}'] = [path_ret(tk, p, H) for tk, p in zip(F15.tk, POS)]
    F15[f'c{H}'] = np.nanmean([[path_ret(tk, p + int(rng.choice([k for k in range(-20, 21) if k != 0])), H) for tk, p in zip(F15.tk, POS)] for _ in range(20)], axis=0)
F15.to_parquet(SP + 'entry410.parquet'); F15['mo'] = pd.to_datetime(F15.entry).dt.to_period('M'); e_ = pd.to_datetime(F15.entry); F15['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', np.where((e_ < '2019-01-01') | (e_ >= '2022-01-01'), 'test', 'x'))
for H in (5, 21, 63):
    print(f'\nHOLD {H} TRADING DAYS, -8% close stop, entry at flip-day close'); print(f"{'arm':9s}{'n':>6s} | FLIP entry: win% med% mean% | RANDOM-day control (mean of 20 draws): mean% | flip-ctrl mean pp (t) train | test | win train/test | med train/test")
    base = F15[F15[f'r{H}'].notna() & (F15.half != 'x')]
    for nm, f in ARM.items():
        d = base[f(base)]
        if len(d) < 60: continue
        diff = (d[f'r{H}'] - d[f'c{H}']); m = diff.groupby(d.mo).mean(); tt = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))); hs = {h: d[d.half == h] for h in ('train', 'test')}
        dd = {h: (x[f'r{H}'] - x[f'c{H}']) for h, x in hs.items()}
        print(f"{nm:9s}{len(d):6d} | {(d[f'r{H}']>0).mean()*100:5.1f} {d[f'r{H}'].median()*100:6.2f} {d[f'r{H}'].mean()*100:6.2f} | {d[f'c{H}'].mean()*100:6.2f} | {diff.mean()*100:+.2f} (t {tt:+.1f}) train {dd['train'].mean()*100:+.2f} | test {dd['test'].mean()*100:+.2f} | win {(hs['train'][f'r{H}']>0).mean()*100:4.1f}/{(hs['test'][f'r{H}']>0).mean()*100:4.1f} | med {hs['train'][f'r{H}'].median()*100:5.2f}/{hs['test'][f'r{H}'].median()*100:5.2f}")
