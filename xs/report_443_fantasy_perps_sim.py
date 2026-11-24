"""Nov 24 2026 report 443. Rey's friend's perps prop (usefantasy.xyz rules page). Does the flat ratio-gated 15m flip book (report 442 F arm) pass and get paid under THEIR rules?
DECLARED BEFORE RESULTS.
RULES modeled: equity-based (balance + unrealized, bar LOWS for open longs, conservative) STATIC max drawdown 3/6/8% (Low/Med/High) of start; daily loss 3/3/4% of start vs the previous day's last equity (day = calendar date, ~00:00 UTC); eval target +10% (pass when equity >= start*1.10 -> everything closed at market); no time limit; stock perps; costs: 0.06% taker per fill (0.12% round trip) + financing 10.95%/yr on notional accrued per calendar hour held (long, always a cost). Breach = account dead.
BOOK: 6 slots, each slot notional = G/6 of start balance (G = gross leverage with all 6 slots full; per-position leverage G/6 <= 5x stock cap), 15m long flips with G20 ratio gate, 25 gex names (19 with ratio), exit vector flip, first-come first-served; trades must START at/after the account start.
FUNDED: same account, repeated cycles: when equity >= start*1.05, close all and pay out 90% of 5% (4.5% of start), balance back to start; until breach or data end. Payout policy is MY choice, declared here.
STARTS: every 5th trading day 2018-09 .. 2026-03. OUTPUT by tier x G: eval pass %, breach %, still running %, median days to pass, funded cycles before breach, expected payouts per eval attempt in % of account and per $ of fee. By start era 18-19|20-21|22|23-26.
PASS BAR (my read of Rey's bar: >=1 payout/month, cost in a dry stretch <= ~$250): a tier x G cell is viable if pass >= 50% in BOTH 2018-21 and 2022+ starts AND expected net per attempt > 0 in both."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_440_neggamma_short_and_2025.py').read(); src = src[:src.index("L = pd.DataFrame(L)")].replace("pd.Timestamp('2025-12-30')", "pd.Timestamp('2026-09-30')")
exec(src)
L = pd.DataFrame(L).sort_values('entry').reset_index(drop=True); L['x_t'] = L.entry + pd.to_timedelta(L.hold, unit='D')
BARS = {}
for tk in L.tk.unique():
    H = rth(pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}), 'ts'); BARS[tk] = H.set_index('dt')[['open', 'high', 'low', 'close']]
grid = np.array(sorted(set().union(*[set(b.index.values) for b in BARS.values()])), dtype='datetime64[ns]'); gi = {t: i for i, t in enumerate(grid)}; NG = len(grid)
print('trades', len(L), 'grid', NG, flush=True)
RC, RT, FIN, FEE = 0.0006, 0.0006, 0.1095 / 8760, 0.0006
# per-trade paths on grid slice: close-return and low-return (net of entry fee + financing)
TP = []
for r in L.itertuples():
    B = BARS[r.tk]; seg = B.loc[r.entry:r.x_t]
    if len(seg) < 2: TP.append(None); continue
    o0 = seg.open.iloc[0]; idx = np.searchsorted(grid, seg.index.values); hrs = (seg.index.values - seg.index.values[0]).astype('timedelta64[m]').astype(float) / 60
    c = seg.close.values / o0 - 1 - RC - FIN * hrs; lo = seg.low.values / o0 - 1 - RC - FIN * hrs
    # last bar = exit bar: value at its OPEN (exit at open), minus exit fee
    xr = seg.open.values[-1] / o0 - 1 - RC - FIN * hrs[-1] - FEE; c[-1] = xr; lo[-1] = xr
    TP.append((idx, c, lo, xr))
ok = np.array([t is not None for t in TP]); L = L[ok].reset_index(drop=True); TP = [t for t in TP if t is not None]
ent_i = np.array([t[0][0] for t in TP]); ex_i = np.array([t[0][-1] for t in TP]); days = pd.DatetimeIndex(grid).normalize(); dayid = pd.factorize(days)[0]
def cycle(si, G, dd, dl, tgt, slots=6):
    """one eval/funded cycle starting at grid index si; returns (outcome, end_index). outcome 'pass'|'breach'|'open'."""
    w = G / slots; nt = np.searchsorted(ent_i, si); P = np.zeros(NG); PL = np.zeros(NG); S = np.zeros(NG + 1); open_ = []
    last = NG - 1
    for k in range(nt, len(TP)):
        e = ent_i[k]
        if e > last: break
        open_ = [x for x in open_ if x > e]
        if len(open_) >= slots: continue
        open_.append(ex_i[k]); idx, c, lo, xr = TP[k]; P[idx] += w * c; PL[idx] += w * lo; S[ex_i[k] + 1] += w * xr
    eqc = P + np.cumsum(S)[:NG]; eql = PL + np.cumsum(S)[:NG]
    # a trade's path adds only inside its slice; after exit the step S carries realized value. Within slice P adds c; no double count because S starts at ex+1.
    sl = slice(si, NG); e_c, e_l = eqc[sl], eql[sl]; di = dayid[sl]
    # daily floor: previous day's last equity - dl
    lastofday = np.r_[np.where(np.diff(di) != 0)[0], len(di) - 1]; prev_eq = np.full(len(di), np.nan); pe = 0.0
    start_of = np.r_[0, lastofday[:-1] + 1]
    for s_, l_ in zip(start_of, lastofday):
        prev_eq[s_:l_ + 1] = pe; pe = e_c[l_]
    brk = (e_l <= -dd) | (e_l <= prev_eq - dl); pas = e_c >= tgt
    bi = np.argmax(brk) if brk.any() else None; pi = np.argmax(pas) if pas.any() else None
    if bi is not None and (pi is None or bi <= pi): return 'breach', si + bi
    if pi is not None: return 'pass', si + pi
    return 'open', NG - 1
TIER = {'Low': (.03, .03), 'Med': (.06, .03), 'High': (.08, .04)}; FEES10 = {'Low': 75, 'Med': 115, 'High': 155}
tds = pd.DatetimeIndex(sorted(set(days))); starts = [tds[i] for i in range(0, len(tds), 5) if pd.Timestamp('2018-09-01') <= tds[i] <= pd.Timestamp('2026-03-31')]
rows = []
for tier, (dd, dl) in TIER.items():
    for G in (.5, 1, 1.5, 2, 3, 4):
        for sd in starts:
            si = int(np.searchsorted(grid, np.datetime64(sd)));
            o, ei = cycle(si, G, dd, dl, .10); days_ = (pd.Timestamp(grid[ei]) - pd.Timestamp(grid[si])).days; pays = 0; broke = False
            if o == 'pass':
                cur = ei + 1
                while cur < NG - 5:
                    o2, e2 = cycle(cur, G, dd, dl, .05)
                    if o2 == 'pass': pays += 1; cur = e2 + 1
                    else: broke = o2 == 'breach'; break
            rows.append(dict(tier=tier, G=G, start=sd, out=o, days=days_, pays=pays, fund_broke=broke))
        print(tier, G, 'done', flush=True)
R = pd.DataFrame(rows); R['era'] = pd.cut(R.start.dt.year, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26']); R['half'] = np.where(R.start.dt.year <= 2021, '18-21', '22+')
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/r443.pkl')
print('\ntier G | pass% breach% open% | median days to pass | funded payouts/eval (4.5% each) | net/eval on 10K acct ($) | by half: pass% 18-21 / 22+ | net 18-21 / 22+')
for (tier, G), g in R.groupby(['tier', 'G']):
    pc = lambda x: (x.out == 'pass').mean() * 100; net = lambda x: ((x.pays + (x.out == 'pass')).clip(upper=None) * 0 + x.pays * 450 * 1.0 - FEES10[tier]).mean()
    n18, n22 = net(g[g.half == '18-21']), net(g[g.half == '22+'])
    print(f"{tier:4s} {G:3.1f} | {pc(g):4.0f} {(g.out=='breach').mean()*100:4.0f} {(g.out=='open').mean()*100:4.0f} | {g[g.out=='pass'].days.median():5.0f} | {g.pays.mean():4.2f} | {net(g):+7.0f} | {pc(g[g.half=='18-21']):3.0f} / {pc(g[g.half=='22+']):3.0f} | {n18:+6.0f} / {n22:+6.0f}")
