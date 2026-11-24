"""Nov 24 2026 report 424 (rerunnable: uses every name in scratchpad lse_5m that also has a gex file).  Rey: 5m flips; 'positive gamma exits or vector close exits'.
DECLARED: 5m RTH bars, long flips from xs/vector_pine.run_state_machine (enter next 5m bar open), cost 4bp round trip, no time holds.  NEG = net_gex < 0 at the prior close (gex files, OI-based).  Entry arms: ALL flips | NEG flips | POS flips.
EXITS (all signal-based): E1 = vector regime flip (confirmed -1, exit next bar open); E2 = first 5m close below the vector level in force (close < lv[j-1]); E3 = gamma turns/is positive: first session open after entry with prior-close net_gex >= 0, OR vector flip, whichever first; E4 = E2 or E3 whichever first; E5 (added at Rey's request) = PURE positive-gamma exit: first session open after entry with prior-close net_gex >= 0, no vector exit (entries never reaching it within ~64 days are dropped and counted).
PRIMARY: NEG entries beat ALL entries under each exit on mean and median in train (2019-21) and test (2018+2022-26) and in >= 3 of 4 eras (n>=30), month-clustered |t| >= 2.  EXIT comparison (same NEG entries): E2/E3/E4 vs E1 on mean and median, same pass rule.  Pooled across all names present; per-name table shown so one name cannot carry it."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; GX = 'xs/ext/gex/'
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and os.path.exists(GX + f[:-8] + '.csv')); print('names', names)
rows = []
for tk in names:
    H = pd.read_parquet(SP + f'lse_5m/{tk}.parquet'); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    g = pd.read_csv(GX + tk + '.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); gprev = g.shift(1).reindex(days, method='ffill').values
    regs, lv, atr, fl = run_state_machine(H[['high', 'low', 'close']]); o = H.open.values; c = H.close.values; n = len(H); newday = np.r_[True, day[1:] != day[:-1]]
    for i, d in fl:
        if d != 1 or i + 2 >= n: continue
        e = i + 1; ep = o[e]; neg = bool(gprev[e] < 0) if np.isfinite(gprev[e]) else None
        if neg is None: continue
        ex = {}
        for j in range(e + 1, n):
            if 'E1' not in ex and regs[j - 1] == -1: ex['E1'] = j
            if 'E2' not in ex and np.isfinite(lv[j - 1]) and c[j - 1] < lv[j - 1]: ex['E2'] = j
            if 'G' not in ex and newday[j] and np.isfinite(gprev[j]) and gprev[j] >= 0: ex['G'] = j
            if 'E1' in ex and 'E2' in ex and 'G' in ex: break
            if j > e + 5000: break
        ex['E3'] = min([v for k, v in ex.items() if k in ('E1', 'G')], default=None); ex['E4'] = min([v for k, v in ex.items() if k in ('E2', 'G')], default=None); ex['E5'] = ex.get('G')
        for k in ('E1', 'E2', 'E3', 'E4', 'E5'):
            j = ex.get(k)
            if j is None: continue
            rows.append(dict(tk=tk, exit=k, neg=neg, entry=H.dt[e], xt=H.dt[j], ret=o[j] / ep - 1 - 4e-4))
T = pd.DataFrame(rows); T = T[T.entry >= '2018-07-01']; e_ = T.entry; T['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test'); T['mo'] = e_.dt.to_period('M'); T['hold_d'] = (T.xt - T.entry).dt.total_seconds() / 86400
print('flips (E1)', int((T.exit == 'E1').sum()), '| NEG share', T[T.exit == 'E1'].neg.mean().round(2))
def line(d, lab): return f"{lab:14s} n{len(d):6d} win {(d.ret>0).mean()*100:5.1f} med {d.ret.median()*100:6.2f} mean {d.ret.mean()*100:5.2f} hold {d.hold_d.mean():5.2f}d | " + ' '.join(f"{h} {d[d.half==h].ret.mean()*100:5.2f}/{d[d.half==h].ret.median()*100:5.2f}" for h in ('train', 'test')) + ' | ' + ' '.join(f"{era} {(x.ret>0).mean()*100:4.1f}/{x.ret.median()*100:5.2f}" if len(x := d[d.era == era]) >= 30 else f'{era} n<30' for era in ('18-19', '20-21', '22', '23-26'))
def tt(a, b):
    m = (a.groupby('mo').ret.mean() - b.groupby('mo').ret.mean()).dropna(); return m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 else np.nan
for k in ('E1', 'E2', 'E3', 'E4', 'E5'):
    D = T[T.exit == k]; print(f'\n--- exit {k} ---'); print(line(D, 'ALL entries')); print(line(D[D.neg], 'NEG entries') + f" | NEG-ALL t {tt(D[D.neg], D):+.1f}"); print(line(D[~D.neg], 'POS entries'))
print('\nExit comparison on NEG entries (vs E1): mean diff pp (t)'); N1 = T[(T.exit == 'E1') & T.neg]
for k in ('E2', 'E3', 'E4', 'E5'): Nk = T[(T.exit == k) & T.neg]; print(f"{k}: {(Nk.ret.mean()-N1.ret.mean())*100:+.2f}pp (t {tt(Nk, N1):+.1f}) | win {(Nk.ret>0).mean()*100:.1f} vs {(N1.ret>0).mean()*100:.1f} | med {Nk.ret.median()*100:.2f} vs {N1.ret.median()*100:.2f} | hold {Nk.hold_d.mean():.2f}d vs {N1.hold_d.mean():.2f}d")
print('\nper name (E1, NEG vs ALL mean %):', {tk: (round(T[(T.exit == 'E1') & (T.tk == tk) & T.neg].ret.mean() * 100, 3), round(T[(T.exit == 'E1') & (T.tk == tk)].ret.mean() * 100, 3), int(((T.exit == 'E1') & (T.tk == tk)).sum())) for tk in names})
T.to_parquet(SP + 'g5m424.parquet')
