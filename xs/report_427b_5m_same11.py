"""Nov 24 2026 report 427. Rey: the ratio-gated negative-gamma 5m model with (a) the positive-gamma exit, (b) the other options flow.
DECLARED: names = lse_5m names with gex + greeks files (rerunnable).  Episode = consecutive sessions with prior-close net_gex < 0 (per name).  ONE trade per episode: the first 5m vector flip (long) in the episode whose gate is on at entry; cost 4bp; no time holds.  Gates (prior close): G252 (252d median r >= 1.5), G20 (20d median >= 2), G1 (prior-day r >= 3).  FLOW = put-tilt composite (mean 252d PIT pct of gamma_net, vanna_net, vega_net from greeks_daily) <= .33 at the prior close.
EXITS: X1 vector flip | X2 PURE positive gamma (first session open with prior-close net_gex >= 0) | X3 first of X1/X2 | X4 flow-turn: first session open with composite > .67 (call-heavy) or vector flip, whichever first.
ARMS: gate alone, gate&FLOW (entry filter).  CONTROL: OPEN = buy the first bar of the episode when the gate is on (no vector flip), same exits.  PRIMARY (3 gates x 4 exits -> |t| >= 3): flip-entry minus OPEN same episodes, episode-clustered; and mean/median positive in train and test.  Also report absolute win/median/mean/hold, eras.  Episodes with no gate-on flip are skipped."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; GX = 'xs/ext/gex/'; UP = '/mnt/user-data/uploads/fartpie/'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
GATE = {'G252': Rr.rolling(252, min_periods=126).median().shift(1) >= 1.5, 'G20': Rr.rolling(20, min_periods=10).median().shift(1) >= 2, 'G1': Rr.shift(1) >= 3}
GD = pd.read_parquet(SP + 'greeks_daily.parquet'); GD['d'] = pd.to_datetime(GD.d); FL = {}
for tk, g in GD.groupby('tk'):
    g = g.set_index('d').sort_index(); pc = [g[f].rolling(252, min_periods=126).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).shift(1) for f in ('gamma_net', 'vanna_net', 'vega_net')]; FL[tk] = sum(pc) / 3
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f[:-8] in ['AAPL','ADBE','AMD','CAT','COST','CRM','DIS','HD','IBM','MCD','NVDA'] and f.endswith('.parquet') and os.path.exists(GX + f[:-8] + '.csv') and f[:-8] in Rr.columns and f[:-8] in FL); print('names', len(names), names)
rows = []
for tk in names:
    H = pd.read_parquet(SP + f'lse_5m/{tk}.parquet'); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    g = pd.read_csv(GX + tk + '.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); gprev = g.shift(1).reindex(days, method='ffill').values; fl = FL[tk].reindex(days, method='ffill').values
    gt = {k: v[tk].reindex(days, method='ffill').values.astype(float) == 1 for k, v in GATE.items()}; n = len(H); o = H.open.values; newday = np.r_[True, day[1:] != day[:-1]]; neg = np.where(np.isfinite(gprev), gprev < 0, False)
    regs, lv, atr, fls = run_state_machine(H[['high', 'low', 'close']]); flips = [i + 1 for i, d in fls if d == 1 and i + 2 < n]; ep = np.zeros(n, int); k = 0
    for i in range(n):
        if neg[i] and (i == 0 or not neg[i - 1]): k += 1
        ep[i] = k if neg[i] else 0
    def exits(e):
        j1 = j2 = j4 = None
        for j in range(e + 1, min(n, e + 6000)):
            if j1 is None and regs[j - 1] == -1: j1 = j
            if newday[j]:
                if j2 is None and np.isfinite(gprev[j]) and gprev[j] >= 0: j2 = j
                if j4 is None and np.isfinite(fl[j]) and fl[j] > .67: j4 = j
            if j1 is not None and j2 is not None and j4 is not None: break
        c = lambda a, b: min([x for x in (a, b) if x is not None], default=None); return dict(X1=j1, X2=j2, X3=c(j1, j2), X4=c(j1, j4))
    for eid in range(1, k + 1):
        idx = np.where(ep == eid)[0]; s = idx[0]
        if H.dt[s] < pd.Timestamp('2018-07-01'): continue
        for gk in GATE:
            fe = [b for b in flips if ep[b] == eid and gt[gk][b]]
            for arm, e in (('FLIP', fe[0] if fe else None), ('OPEN', s if gt[gk][s] else None)):
                if e is None: continue
                ex = exits(e); flow = bool(np.isfinite(fl[e]) and fl[e] <= .33)
                for xk, j in ex.items():
                    if j is None: continue
                    rows.append(dict(tk=tk, ep=f'{tk}{eid}', gate=gk, arm=arm, exit=xk, flow=flow, entry=H.dt[e], ret=o[j] / o[e] - 1 - 4e-4, hold=(H.dt[j] - H.dt[e]).total_seconds() / 86400))
T = pd.DataFrame(rows); e_ = T.entry; T['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test')
def line(d, lab): return f"{lab:20s} n{len(d):5d} win {(d.ret>0).mean()*100:5.1f} med {d.ret.median()*100:6.2f} mean {d.ret.mean()*100:5.2f} hold {d.hold.mean():5.1f}d | " + ' '.join(f"{h} {d[d.half==h].ret.mean()*100:5.2f}/{d[d.half==h].ret.median()*100:5.2f}" for h in ('train', 'test')) + ' | ' + ' '.join(f"{era} {x.ret.mean()*100:5.2f}" if len(x := d[d.era == era]) >= 15 else f'{era} n<15' for era in ('18-19', '20-21', '22', '23-26'))
for gk in GATE:
    for xk in ('X1', 'X2', 'X3', 'X4'):
        F = T[(T.gate == gk) & (T.exit == xk) & (T.arm == 'FLIP')]; O = T[(T.gate == gk) & (T.exit == xk) & (T.arm == 'OPEN')]; common = sorted(set(F.ep) & set(O.ep)); fi = F.set_index('ep').loc[common]; oi = O.set_index('ep').loc[common]; d = (fi.ret - oi.ret); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 3 else np.nan
        print(f"\n[{gk} {xk}]"); print(line(F, 'FLIP (gated,NEG)')); print(line(F[F.flow], 'FLIP & FLOW')); print(line(O, 'OPEN control') + f" | FLIP-OPEN {d.mean()*100:+.2f}pp (episode t {t:+.1f}, {len(d)} ep)")
T.to_parquet(SP + 'g5m427_same11.parquet')
