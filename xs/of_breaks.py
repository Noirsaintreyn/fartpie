"""Nov 10 2026: ORDER FLOW (Databento MNQ TBBO aggregated to 1 min, 2024-12-18..2025-11-18) AT NOISE-BAND BREAKS. DECLARED BEFORE RESULTS.
The NQ 5m noise engine's trades on the days covered (about 11 months, ~250 trades) are tagged at the signal bar with: d5 (signed volume of the signal bar, aligned to the trade direction, / its total volume), d30 (last 30 min, aligned), dsess (session cumulative signed volume since 09:30, aligned, / session volume), div (1 if price made a new session extreme in the trade direction while the 30-min aligned delta is negative = absorption / divergence), imb (n buy trades share proxy: buy_vol / (buy+sell) of the signal bar), spr (mean spread in ticks), size (bid+ask size mean).
Judged by top-vs-bottom third points per trade with a 5000-permutation p-value (one period only, so low power: only large effects can show). Also the same on the ORB breaks. Nothing here can PASS the two-period rule; report effect sizes and permutation p, and say so."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, DAYS, GRID, NB
O = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); O['dt'] = pd.to_datetime(O.datetime).dt.tz_convert('America/New_York').dt.tz_localize(None); O = O.set_index('dt').sort_index()
O['tot'] = O.buy_vol + O.sell_vol; O['day'] = O.index.normalize(); O['m'] = O.index.hour * 60 + O.index.minute
cov = sorted(set(O.day)); cov = [d for d in cov if d in A]; print(len(cov), 'covered days', cov[0], cov[-1])
rows = []
for d in cov:
    g = O[(O.day == d) & (O.m >= 570) & (O.m < 960)]
    if len(g) < 300: continue
    cs = g.signed_vol.cumsum(); ctot = g.tot.cumsum()
    for (dd, em, pos, pts, mae) in Z.noise(d):
        t1 = 570 + 5 * em; t0 = t1 - 5                                          # signal bar = em-1: minutes [570+5(em-1), 570+5em)
        gb = g[(g.m >= t0) & (g.m < t1)]; g30 = g[(g.m >= t1 - 30) & (g.m < t1)]; sess = g[g.m < t1]
        if len(gb) < 3 or len(g30) < 20 or len(sess) < 20: continue
        o_, h_, l_, c_, v_, vw_ = A[d][0]; jb = em - 1; newhi = (pos > 0 and h_[jb] >= np.nanmax(h_[:jb + 1])) or (pos < 0 and l_[jb] <= np.nanmin(l_[:jb + 1]))
        d30 = pos * g30.signed_vol.sum() / max(g30.tot.sum(), 1)
        rows.append(dict(day=d, pos=pos, pts=pts, d5=pos * gb.signed_vol.sum() / max(gb.tot.sum(), 1), d30=d30, dsess=pos * sess.signed_vol.sum() / max(sess.tot.sum(), 1), div=float(newhi and d30 < 0), imb=gb.buy_vol.sum() / max(gb.tot.sum(), 1) * (1 if pos > 0 else -1), spr=gb.mean_spread_ticks.mean(), size=(gb.mean_bid_sz + gb.mean_ask_sz).mean()))
R = pd.DataFrame(rows); print(len(R), 'NZ trades with order flow;', f'mean {R.pts.mean():+.1f} pts, long {R[R.pos > 0].pts.mean():+.1f} (n{int((R.pos > 0).sum())}), short {R[R.pos < 0].pts.mean():+.1f} (n{int((R.pos < 0).sum())})', flush=True)
rng = np.random.default_rng(1)
def perm(x, y, hi, lo):
    a, b = y[hi], y[lo]; real = a.mean() - b.mean(); n1, n2 = len(a), len(b); allv = np.concatenate([a, b]); cnt = 0
    for _ in range(5000):
        p = rng.permutation(allv); cnt += abs(p[:n1].mean() - p[n1:].mean()) >= abs(real)
    return real, cnt / 5000, n1, n2
for lab, sub in (('all', R), ('long only', R[R.pos > 0])):
    print(f'\n{lab}: {len(sub)} trades; points per trade, top third minus bottom third, permutation p')
    y = sub.pts.values
    for c in ('d5', 'd30', 'dsess', 'imb', 'spr', 'size'):
        x = sub[c].values; q = np.quantile(x, [1 / 3, 2 / 3]); r, p, n1, n2 = perm(x, y, x > q[1], x <= q[0]); print(f'   {c:<6} top - bottom {r:+7.1f} pts (p {p:.3f}, n {n1}/{n2}); top third avg {y[x > q[1]].mean():+.1f}, bottom {y[x <= q[0]].mean():+.1f}')
    a, b = sub[sub['div'] == 1].pts, sub[sub['div'] == 0].pts; print(f'   divergence (new extreme with opposing 30-min delta): n {len(a)} avg {a.mean():+.1f} vs {b.mean():+.1f} without')
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/of_breaks.pkl')
