"""Nov 24 2026 report 425: controls for the E5 result (enter on a negative-gamma 5m vector flip, exit at the first session open with prior-close net_gex >= 0; report 424: win 73%, mean +1.59%, hold 11.6d).
Question: is that the vector flip, or just 'be long during a negative-gamma stretch until it ends'?  Episode = consecutive sessions with prior-close net_gex < 0 (per name).
Arms (all exit at the episode end = first session open with prior-close net_gex >= 0, cost 4bp): FLIP-ALL (every 5m vector flip inside an episode), FLIP-FIRST (first flip per episode), OPEN (episode start: first bar of the first NEG session), RAND (random bar inside the episode, 20 draws per episode, same count as flips-first), plus per-name and per-era tables.  Cluster = episode (not month) for t.
H: the flip adds value if FLIP-FIRST minus RAND (same episodes) mean > 0 in both halves with episode-clustered |t| >= 2.  Also report max adverse excursion and the share of episodes with a loss > 10%."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; GX = 'xs/ext/gex/'; rng = np.random.default_rng(425)
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and os.path.exists(GX + f[:-8] + '.csv')); print('names', names)
rows = []
for tk in names:
    H = pd.read_parquet(SP + f'lse_5m/{tk}.parquet'); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    g = pd.read_csv(GX + tk + '.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; day = H.dt.dt.normalize().values; gprev = g.shift(1).reindex(pd.DatetimeIndex(day), method='ffill').values; n = len(H); o = H.open.values; lo = H.low.values
    neg = np.where(np.isfinite(gprev), gprev < 0, False); newday = np.r_[True, day[1:] != day[:-1]]
    ep = np.zeros(n, int); k = 0
    for i in range(n):
        if neg[i] and (i == 0 or not neg[i - 1]): k += 1
        ep[i] = k if neg[i] else 0
    # episode end bar = first new-day bar with gprev >= 0 after the episode
    end = {}
    for e in range(1, k + 1):
        idx = np.where(ep == e)[0]; last = idx[-1]; nxt = last + 1
        while nxt < n and not (newday[nxt] and np.isfinite(gprev[nxt]) and gprev[nxt] >= 0): nxt += 1
        if nxt < n: end[e] = (idx[0], last, nxt)
    regs, lv, atr, fl = run_state_machine(H[['high', 'low', 'close']]); flips = [i + 1 for i, d in fl if d == 1 and i + 2 < n]
    def trade(e_bar, eid): s, l, x = end[eid]; r = o[x] / o[e_bar] - 1 - 4e-4; mae = lo[e_bar:x].min() / o[e_bar] - 1; return r, mae, (H.dt[x] - H.dt[e_bar]).total_seconds() / 86400
    for eid, (s, l, x) in end.items():
        if H.dt[s] < pd.Timestamp('2018-07-01'): continue
        fe = [b for b in flips if ep[b] == eid]; rec = lambda arm, b: rows.append(dict(tk=tk, ep=f'{tk}{eid}', arm=arm, entry=H.dt[b], **dict(zip(('ret', 'mae', 'hold'), trade(b, eid)))))
        for b in fe: rec('FLIP-ALL', b)
        if fe: rec('FLIP-FIRST', fe[0])
        rec('OPEN', s)
        for _ in range(20): rec('RAND', int(rng.integers(s, l + 1)))
T = pd.DataFrame(rows); e_ = T.entry; T['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test')
print('episodes', T[T.arm == 'OPEN'].ep.nunique(), '| flips-first episodes', T[T.arm == 'FLIP-FIRST'].ep.nunique())
def line(d, lab): return f"{lab:11s} n{len(d):6d} win {(d.ret>0).mean()*100:5.1f} med {d.ret.median()*100:6.2f} mean {d.ret.mean()*100:5.2f} hold {d.hold.mean():5.1f}d | MAE median {d.mae.median()*100:5.1f}% | loss>10% {(d.ret<-.1).mean()*100:4.1f}% | " + ' '.join(f"{h} {d[d.half==h].ret.mean()*100:5.2f}" for h in ('train', 'test')) + ' | ' + ' '.join(f"{era} {x.ret.mean()*100:5.2f}(n{len(x)})" if len(x := d[d.era == era]) >= 20 else f'{era} n<20' for era in ('18-19', '20-21', '22', '23-26'))
for a in ('FLIP-ALL', 'FLIP-FIRST', 'OPEN', 'RAND'): print(line(T[T.arm == a], a))
ff = T[T.arm == 'FLIP-FIRST'].set_index('ep'); rd = T[T.arm == 'RAND'].groupby('ep').ret.mean(); op = T[T.arm == 'OPEN'].set_index('ep').ret
for lab, ctl in (('RAND', rd), ('OPEN', op)):
    d = (ff.ret - ctl.reindex(ff.index)).dropna(); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))); h = {k: (d[ff.loc[d.index].half == k].mean() * 100) for k in ('train', 'test')}; print(f"FLIP-FIRST minus {lab}: {d.mean()*100:+.2f}pp (episode t {t:+.1f}, n={len(d)}) | train {h['train']:+.2f} test {h['test']:+.2f}")
print('\nper name FLIP-FIRST vs RAND mean %:', {tk: (round(ff[ff.tk == tk].ret.mean() * 100, 2), round(rd[rd.index.str.startswith(tk)].mean() * 100, 2), int((ff.tk == tk).sum())) for tk in names})
print('episode length (days) median', round(T[T.arm == 'OPEN'].hold.median(), 1), '| by name episodes:', T[T.arm == 'OPEN'].groupby('tk').size().to_dict())
T.to_parquet(SP + 'ep425.parquet')
