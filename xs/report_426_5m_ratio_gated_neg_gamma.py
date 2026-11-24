"""Nov 24 2026 report 426. Rey: intraday is noisy, so only take GOOD flips: into negative gamma, on stocks where options flow matters (options-ratio gate).  Vector = classifier.
DECLARED: 5m RTH long flips (xs/vector_pine), enter next bar open, 4bp, signal exits only: E1 = vector flip exit; E3 = first of positive gamma (first session open with prior-close net_gex >= 0) or vector flip.  Names = every name in scratchpad lse_5m with a gex file (rerunnable; pull continues).
Ratio gates (all known at the prior close, per name, r = 100 x option contracts / shares volume): G252 = rolling-252d median >= 1.5 (live gate), G20 = rolling-20d median >= 2 (shorter 'intraday-era' window), G1 = prior-day r >= 3.  NEG = net_gex < 0 at the prior close.
Arms: ALL | NEG | GATED (per gate) | GATED&NEG (per gate).  PRIMARY (3 gates -> |t| >= 2.4): GATED&NEG beats GATED-not-NEG AND beats ALL on mean and median in train (2019-21) and test (2018+2022-26) and in >= 3 of 4 eras (n>=30), month-clustered.  Overlap check: flips per name-day, distinct names, and results with ONE flip per name-day (first)."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; GX = 'xs/ext/gex/'; UP = '/mnt/user-data/uploads/fartpie/'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and os.path.exists(GX + f[:-8] + '.csv') and f[:-8] in Rr.columns); print('names', len(names), names)
GATE = {'G252': Rr.rolling(252, min_periods=126).median().shift(1) >= 1.5, 'G20': Rr.rolling(20, min_periods=10).median().shift(1) >= 2, 'G1': Rr.shift(1) >= 3}
rows = []
for tk in names:
    H = pd.read_parquet(SP + f'lse_5m/{tk}.parquet'); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    g = pd.read_csv(GX + tk + '.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); gprev = g.shift(1).reindex(days, method='ffill').values
    gt = {k: v[tk].reindex(days, method='ffill').values for k, v in GATE.items()}
    regs, lv, atr, fl = run_state_machine(H[['high', 'low', 'close']]); o = H.open.values; n = len(H); newday = np.r_[True, day[1:] != day[:-1]]
    for i, d in fl:
        if d != 1 or i + 2 >= n: continue
        e = i + 1; ep = o[e]
        if not np.isfinite(gprev[e]): continue
        j1 = j3 = None
        for j in range(e + 1, n):
            if j1 is None and regs[j - 1] == -1: j1 = j
            if newday[j] and np.isfinite(gprev[j]) and gprev[j] >= 0 and j3 is None: jg = j; j3 = j
            if j1 is not None: break
            if j > e + 3000: break
        if j1 is None: continue
        j3 = min(j1, j3) if j3 is not None else j1
        rows.append(dict(tk=tk, day=pd.Timestamp(day[e]), entry=H.dt[e], neg=bool(gprev[e] < 0), E1=o[j1] / ep - 1 - 4e-4, E3=o[j3] / ep - 1 - 4e-4, h1=(H.dt[j1] - H.dt[e]).total_seconds() / 86400, **{k: bool(v[e]) if np.isfinite(v[e].astype(float)) else False for k, v in gt.items()}))
T = pd.DataFrame(rows); T = T[T.entry >= '2018-07-01']; e_ = T.entry; T['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test'); T['mo'] = e_.dt.to_period('M')
print('flips', len(T), '| NEG', T.neg.mean().round(2), '| gate shares', {k: round(float(T[k].mean()), 2) for k in GATE})
def line(d, c, lab): return f"{lab:18s} n{len(d):6d} names {d.tk.nunique():3d} win {(d[c]>0).mean()*100:5.1f} med {d[c].median()*100:6.2f} mean {d[c].mean()*100:5.2f} | " + ' '.join(f"{h} {d[d.half==h][c].mean()*100:5.2f}/{d[d.half==h][c].median()*100:5.2f}" for h in ('train', 'test')) + ' | ' + ' '.join(f"{era} {(x[c]>0).mean()*100:4.1f}/{x[c].median()*100:5.2f}" if len(x := d[d.era == era]) >= 30 else f'{era} n<30' for era in ('18-19', '20-21', '22', '23-26'))
def tt(a, b, c):
    m = (a.groupby('mo')[c].mean() - b.groupby('mo')[c].mean()).dropna(); return m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 else np.nan
for c in ('E1', 'E3'):
    print(f'\n===== exit {c} ====='); print(line(T, c, 'ALL')); print(line(T[T.neg], c, 'NEG'))
    for gk in GATE:
        G_ = T[T[gk]]; GN = G_[G_.neg]; GP = G_[~G_.neg]; print(line(G_, c, f'{gk} gated')); print(line(GN, c, f'{gk} gated&NEG') + f" | vs gated-notNEG t {tt(GN, GP, c):+.1f}, vs ALL t {tt(GN, T, c):+.1f}"); print(line(GP, c, f'{gk} gated&POS'))
print('\nOne flip per name-day (first), exit E1, G252/G20 gated&NEG:')
F1 = T.sort_values('entry').groupby(['tk', 'day']).head(1)
for gk in ('G252', 'G20', 'G1'): d = F1[F1[gk] & F1.neg]; print(line(d, 'E1', gk + ' first/day'), f"| flips per name-day (all) {len(T[T[gk] & T.neg]) / max(1, len(d)):.1f}")
T.to_parquet(SP + 'g5m426.parquet')
