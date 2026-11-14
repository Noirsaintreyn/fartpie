"""Nov 14 2026 ACTIVIST / 5%-HOLDER FILINGS AS FLIP FILTER (declared before results; from the catalyst list Rey pasted). Hypothesis: a Schedule 13D (a >=5% holder who intends to influence the company) filed in the 30 days before a long flip -> BETTER flips (outside money accumulating); 13G (passive >=5%) is the control (expected no effect). Wide daily universe (report_64 flips, 2019-2026Q1, survivorship-biased), 30bp. 13D = SC 13D / SCHEDULE 13D incl. amendments; NEW 13D = initial filing only (no '/A'). Strict eras 2019-21 / 2022-26Q1, month-clustered t of group vs no-filing flips; tested on all flips and on BV>=1.5 flips; plus combined with the insider tilt."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; F = pd.read_csv(SC + '13d_wide.csv'); F['fd'] = pd.to_datetime(F.filed)
print('filings', len(F), 'tickers', F.ticker.nunique(), F.form.value_counts().to_dict())
def flag(mask_forms, days=30):
    x = F[mask_forms(F.form)]; per = {tk: g.fd.sort_values().values for tk, g in x.groupby('ticker')}; out = np.zeros(len(P), bool)
    for tk, ix in P.groupby('tk').groups.items():
        a = per.get(tk)
        if a is None: continue
        t = P.loc[ix, 'd0'].values; pos = np.searchsorted(a, t, side='right') - 1; ok = pos >= 0; out[P.index.get_indexer(ix)] = ok & (a[np.clip(pos, 0, None)] > t - np.timedelta64(days, 'D'))
    return out
P['D13'] = flag(lambda s: s.str.contains('13D')); P['D13NEW'] = flag(lambda s: s.str.contains('13D') & ~s.str.contains('/A')); P['G13'] = flag(lambda s: s.str.contains('13G')); P['INS'] = P.NI >= 1
def mc(S, m, base):
    s = S[m & base]; r = S[~m & base]; a = s.groupby(s.t.dt.to_period('M')).rn.mean(); b = r.groupby(r.t.dt.to_period('M')).rn.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
for en, a, b in (('2019-21', '2019', '2022'), ('2022-26Q1', '2022', '2026-04'), ('ALL', '2019', '2026-04')):
    S = P[(P.t >= a) & (P.t < b)].reset_index(drop=True); print(f'\n=== {en}: {len(S)} flips')
    for bn, base in (('all flips', pd.Series(True, index=S.index)), ('BV>=1.5', S.BV >= 1.5)):
        z = S[base]; print(f'  -- base {bn}: mean {z.rn.mean()*100:+.2f}% n {len(z)} huge {z.HUGE.mean()*100:.2f}%')
        for k, m in (('13D (any, 30d)', S.D13), ('NEW 13D only', S.D13NEW), ('13G (control)', S.G13), ('13D or insider', S.D13 | S.INS), ('13D and insider', S.D13 & S.INS)):
            s = S[m & base]
            if len(s) < 5: print(f'     {k:18s} n {len(s)}'); continue
            print(f'     {k:18s} n {len(s):5d} | mean {s.rn.mean()*100:+7.2f}% | med {s.rn.median()*100:+5.2f} | huge {s.HUGE.mean()*100:4.1f}% | t vs rest {mc(S, m, base):+.1f}')
