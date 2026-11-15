"""Nov 15 2026 CHANDELIER ON THE NQ PROP ALGO'S RULE A LEG (Rey: 'my prop algorithm I currently have, plug it in'). Declared before results. SCOPE: only Rule A is replayable here (live_data/nq_rule_a_trades.csv: 357 trades 2018-01..2026-09, entry time, direction, exit type stop/target/noon, net pts after 1 pt cost); the other legs (M1, R1, ON+/-, RULES, V15a, add-ons) and the eval/payout sim are NOT replayed because their per-trade paths are not saved (only pts/mae in the pickles). Replay on NQ 5m (nq5_drive_et.parquet, ET): entry at the open of the 5m bar at the logged entry time, 1.0 pt cost, hard stop -50 (fills at the stop price on touch), target +75 (fills at the target on touch), noon exit at the open of the 12:00 bar (validated: replay vs logged net_pts corr 0.998 on the noon trades). Variants: BASE (stop/target/noon as logged); NOTGT (stop -50 + noon, no target); CH-M (stop -50 kept + chandelier from entry, extreme since entry -/+ M x ATR22 on 5m closes, exit at that close, noon backstop, no target, M = 3 primary, 4/5 sensitivity); CHONLY-M (chandelier replaces the stop too, noon backstop, no target). Metrics: n, mean/median pts per trade, win rate, total pts, max drawdown pts (closed-trade equity), worst trade, 2018-21 vs 2022+. Rey's Nov 11 rule noted: for vector flips the vector flip is the stop; Rule A is not a vector strategy and keeps its stop. Lead bar: higher mean pts AND lower or equal max drawdown in BOTH eras versus BASE. Only 357 trades, so any difference under about 5 pts per trade is noise."""
import numpy as np, pandas as pd
T = pd.read_csv('/mnt/attach/outputs/acct/nq_rule_a_trades.csv', parse_dates=['day', 'entry']); d = pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').set_index('dt').sort_index(); COST = 1.0; STOP = 50.; TGT = 75.
def rma(x, n):
    o = np.full(len(x), np.nan); o[n - 1] = np.nanmean(x[:n]); a = 1 / n
    for k in range(n, len(x)): o[k] = o[k - 1] * (1 - a) + x[k] * a
    return o
h = d.high.values; l = d.low.values; c = d.close.values; o_ = d.open.values; pc = np.r_[c[0], c[:-1]]; A = rma(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), 22); ix = {t: k for k, t in enumerate(d.index)}
def replay(r, variant, M=3):
    i = ix.get(r.entry)
    if i is None: return np.nan
    day = r.entry.normalize(); noon = day + pd.Timedelta(hours=12); s = r.dir; e = o_[i]; ext = e; k = i
    use_stop = variant != 'CHONLY'; use_tgt = variant == 'BASE'; use_ch = variant in ('CH', 'CHONLY')
    while True:
        t = d.index[k]
        if t >= noon: return ((o_[k] - e) * s - COST)
        if k > i or True:
            if use_stop and ((l[k] <= e - STOP) if s == 1 else (h[k] >= e + STOP)): return -STOP - COST
            if use_tgt and ((h[k] >= e + TGT) if s == 1 else (l[k] <= e - TGT)): return TGT - COST
        ext = max(ext, h[k]) if s == 1 else min(ext, l[k])
        if use_ch and k > i and not np.isnan(A[k]) and ((s == 1 and c[k] < ext - M * A[k]) or (s == -1 and c[k] > ext + M * A[k])): return (c[k] - e) * s - COST
        k += 1
        if k >= len(d) or d.index[k].normalize() != day: return (c[k - 1] - e) * s - COST
rows = []
for _, r in T.iterrows():
    row = dict(day=r.day, rec=r.net_pts, BASE=replay(r, 'BASE'), NOTGT=replay(r, 'NOTGT'))
    for M in (3, 4, 5): row[f'CH{M}'] = replay(r, 'CH', M); row[f'CHONLY{M}'] = replay(r, 'CHONLY', M)
    rows.append(row)
R = pd.DataFrame(rows).dropna(); print('trades replayed', len(R), 'of', len(T), '| BASE replay vs logged: mean', R.BASE.mean().round(2), R.rec.mean().round(2), 'corr', R[['BASE', 'rec']].corr().iloc[0, 1].round(3))
def st(x): eq = x.cumsum(); return f'n {len(x):4d} mean {x.mean():+6.2f} med {x.median():+6.1f} win {(x>0).mean()*100:4.1f}% total {x.sum():+7.0f} maxDD {(eq.cummax()-eq).max():5.0f} worst {x.min():+6.1f}'
for era, z in (('2018-21', R[R.day < '2022-01-01']), ('2022+', R[R.day >= '2022-01-01'])):
    print(f'\n {era}')
    for k in ('rec', 'BASE', 'NOTGT', 'CH3', 'CH4', 'CH5', 'CHONLY3', 'CHONLY4', 'CHONLY5'): print(f'  {k:8s}', st(z[k]))
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/rulea_chand.pkl')
