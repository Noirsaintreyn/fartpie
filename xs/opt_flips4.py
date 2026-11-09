"""Nov 9 2026: A PUT-SPREAD INCOME RULE CHOSEN ON 2019-21 ONLY, JUDGED ON 2022-26. DECLARED BEFORE RESULTS.
xs/opt_flips2.py, 2019-21 half only: put credit spreads paid more when the stock's own net GEX was NEGATIVE at the prior close (put-heavy: put buyers pay up) and when
IV rank was NOT in its top third. Rule: sell the put credit spread only when own gamma < 0 and IV rank < 2/3. Every 5th trading day per name (the random-day
set: the flip adds nothing to put selling), held to expiry, the long put is the stop. MONTHLY (~30 days) and WEEKLY (~7 days), short strike at spot (k 0) or 0.5 sigma below.
Report 2019-21 (where it was chosen) and 2022-26 (out of sample): per trade at the TOUCH and MID, and a monthly book at 1 unit of risk per spread."""
import numpy as np, pandas as pd, glob, os
U = '/mnt/user-data/uploads/fartpie'; R = pd.read_parquet('/tmp/claude-0/opt_flips_R.parquet'); R = R[(R.set == 'base') & (R.ex == 'EXP')]
CTX = []
for f in glob.glob(f'{U}/xs/ext/gex/*.csv'):
    tk = os.path.basename(f)[:-4]; g = pd.read_csv(f, parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
    CTX.append(pd.DataFrame({'tk': tk, 'd0': g.index, 'ivr': g.atm_iv.rolling(252, min_periods=120).rank(pct=True).values, 'gam': g.net_gex.shift(1).values}))
R = R.merge(pd.concat(CTX), on=['tk', 'd0'], how='inner').dropna(subset=['ivr', 'gam']); R['rule'] = (R.gam < 0) & (R.ivr < 2 / 3); R['mo'] = R.d0.dt.to_period('M')
print(f'{R.tk.nunique()} names; rule on {R[R.st == "PCS 0"].drop_duplicates(["tk", "d0"]).rule.mean():.0%} of days', flush=True)
for ten, st in (('M', 'PCS 0'), ('M', 'PCS 0.5'), ('W', 'PCS 0'), ('W', 'PCS 0.5')):
    for fill in ('TOUCH', 'MID'):
        g = R[(R.ten == ten) & (R.st == st) & (R.fill == fill)]; line = f'  {ten} {st:<7} {fill:<5}'
        for per in ('19-21', '22-26'):
            for lab, msk in (('rule', g.rule), ('every day', np.ones(len(g), bool))):
                t = g[(g.per == per) & msk]; m_ = t.groupby('mo').ret.sum(); nm = t.groupby('mo').size().mean()
                line += f' | {per} {lab}: {t.ret.mean() * 100:+5.2f}% win {t.win.mean():3.0%} n{len(t):4d}, {nm:4.1f}/mo, month avg {m_.mean():+5.2f}u, months+ {np.mean(m_ > 0):3.0%}, worst {m_.min():+5.2f}u'
        print(line, flush=True)
