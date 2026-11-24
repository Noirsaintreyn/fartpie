"""Nov 24 2026 report 445. Rey: if vector flips on equities outperform QQQ this should work on the perps prop; run the other stats.
DECLARED BEFORE RESULTS. Same Fantasy rules/costs as report 443-444 (High tier: static DD 8% equity-based using bar lows, daily loss 4%, target 10%, fee $155 on 10K; funded cycles +5% -> payout 4.5%; 0.06%/fill + 10.95%/yr financing). Starts every 10th trading day 2018-09..2026-03; eval horizon 200d, funded cycle horizon 120d, payouts counted within 180d of passing.
UNIVERSES (15m, vector flip entry next open, exit opposite flip, overnight allowed):
 U1 19 gex names, G20 ratio gate, long (report 442/443 baseline)
 U2 WIDE stocks: every name with 15m cache AND options-ratio history, G20 gate, long
 U3 INDEX: QQQ SPY IWM DIA ungated, long AND short (stand-ins for index perps, 10x cap)
 U4 U2 + U3
SIZING: flat G (gross, slot = G/6) vs DYN: slot weight x clip((equity+DD)/DD, 0.3, 1.5) at entry (shrinks as the cushion above the static floor shrinks, grows as profit builds).
Ge in {1,2,3}, Gf = 1. Metrics: passed by 30/60/90d, first payout by 60/90/120d, eval breach %, net$/attempt (450/payout - fee) all | 18-21 | 22+.
Objective (declared): first payout by 90d >= 25% of attempts AND net > 0 in BOTH halves; else the universe/sizing is not viable for this prop."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
src = open('xs/report_440_neggamma_short_and_2025.py').read(); exec(src[:src.index("names = sorted")].split("SP = ")[0] + "SP = '" + SP + "'; UP = '" + UP + "'; ET = 'America/New_York'\n" + "\n".join(l for l in src[:src.index("names = sorted")].split("\n") if l.startswith("R = ") or l.startswith("G20 =") or l.startswith("Rr")))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
G19 = [f[:-4] for f in os.listdir(UP + 'xs/ext/gex') if f.endswith('.csv') and f[:-4] != 'QQQ']
files = {os.path.basename(f).split('_')[0]: f for f in glob.glob(UP + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')}
INDEX = ['QQQ', 'SPY', 'IWM', 'DIA']; WIDE = sorted(t for t in files if t in G20.columns and t not in INDEX)
print('wide ratio names', len(WIDE), 'index', INDEX, flush=True)
BARS, TRADES = {}, []
def trades_for(tk, sides, gated):
    H = rth(pd.read_parquet(files[tk]).rename(columns={'datetime': 'ts'}), 'ts'); BARS[tk] = H.set_index('dt')[['open', 'high', 'low', 'close']]; n = len(H); o_ = H.open.values; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day)
    regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']]); gate = G20[tk].reindex(days, method='ffill').values.astype(float) == 1 if (gated and tk in G20.columns) else np.ones(n, bool)
    out = []
    for i, dd in fls:
        if dd not in sides: continue
        fb = i + 1; e = fb + 1
        if e + 1 >= n or not gate[fb - 1]: continue
        t0 = H.dt[e]
        if t0 < pd.Timestamp('2018-07-01') or t0 > pd.Timestamp('2026-09-30'): continue
        x = None
        for j in range(e + 1, n):
            if regs[j - 1] == -dd: x = j; break
        if x is None: continue
        out.append(dict(tk=tk, side=dd, entry=H.dt[e], x_t=H.dt[x]))
    return out
U = {'U1': [], 'U2': [], 'U3': []}
for tk in G19:
    if tk in files: U['U1'] += trades_for(tk, (1,), True)
for tk in WIDE: U['U2'] += trades_for(tk, (1,), True)
for tk in INDEX: U['U3'] += trades_for(tk, (1, -1), False)
print({k: len(v) for k, v in U.items()}, flush=True)
U['U4'] = U['U2'] + U['U3']
grid = np.array(sorted(set().union(*[set(b.index.values) for b in BARS.values()])), dtype='datetime64[ns]'); NG = len(grid); days_ = pd.DatetimeIndex(grid).normalize(); dayid = pd.factorize(days_)[0]
RC, FIN, FEE = 0.0006, 0.1095 / 8760, 0.0006
def prep(T):
    T = sorted(T, key=lambda r: r['entry']); TP = []
    for r in T:
        B = BARS[r['tk']]; seg = B.loc[r['entry']:r['x_t']]
        if len(seg) < 2: continue
        s = r['side']; o0 = seg.open.iloc[0]; idx = np.searchsorted(grid, seg.index.values); hrs = (seg.index.values - seg.index.values[0]).astype('timedelta64[m]').astype(float) / 60
        cl = s * (seg.close.values / o0 - 1); wr = (seg.low.values if s == 1 else seg.high.values) / o0 - 1; wr = s * wr
        c = cl - RC - FIN * hrs; lo = wr - RC - FIN * hrs; xr = s * (seg.open.values[-1] / o0 - 1) - RC - FIN * hrs[-1] - FEE; c[-1] = xr; lo[-1] = xr; TP.append((idx, c, lo, xr))
    return TP, np.array([t[0][0] for t in TP]), np.array([t[0][-1] for t in TP])
def cycle(D, si, G, dd, dl, tgt, dyn, HZ, slots=6):
    TP, ent, ex = D; hi = min(NG, si + HZ); m = hi - si
    if m < 30: return 'open', hi - 1
    P = np.zeros(m); PL = np.zeros(m); S = np.zeros(m + 1); open_ = []
    for k in range(np.searchsorted(ent, si), len(TP)):
        e = ent[k]
        if e >= hi: break
        open_ = [x for x in open_ if x > e]
        if len(open_) >= slots: continue
        w = G / slots
        if dyn: w *= min(1.5, max(.3, (P[e - si] + S[:e - si + 1].sum() + dd) / dd))
        open_.append(ex[k]); idx, c, lo, xr = TP[k]; r_ = idx - si; ok = r_ < m; P[r_[ok]] += w * c[ok]; PL[r_[ok]] += w * lo[ok]
        if ex[k] + 1 - si <= m: S[ex[k] + 1 - si] += w * xr
    cs = np.cumsum(S)[:m]; e_c = P + cs; e_l = PL + cs; di = dayid[si:hi]
    last = np.r_[np.where(np.diff(di) != 0)[0], m - 1]; prev = np.full(m, np.nan); pe = 0.0; st = np.r_[0, last[:-1] + 1]
    for a, b in zip(st, last): prev[a:b + 1] = pe; pe = e_c[b]
    brk = (e_l <= -dd) | (e_l <= prev - dl); pas = e_c >= tgt; bi = np.argmax(brk) if brk.any() else None; pi = np.argmax(pas) if pas.any() else None
    if bi is not None and (pi is None or bi <= pi): return 'breach', si + bi
    if pi is not None: return 'pass', si + pi
    return 'open', hi - 1
tds = pd.DatetimeIndex(sorted(set(days_))); starts = [tds[i] for i in range(0, len(tds), 10) if pd.Timestamp('2018-09-01') <= tds[i] <= pd.Timestamp('2026-03-31')]
DD, DL, FEEA = .08, .04, 155; rows = []
for u in ('U1', 'U2', 'U3', 'U4'):
    D = prep(U[u]); print(u, 'trades', len(D[0]), flush=True)
    for Ge in (1, 2, 3):
        for dyn in (False, True):
            for sd in starts:
                si = int(np.searchsorted(grid, np.datetime64(sd))); t0 = pd.Timestamp(grid[si]); o, ei = cycle(D, si, Ge, DD, DL, .10, dyn, 5200); tp = (pd.Timestamp(grid[ei]) - t0).days if o == 'pass' else np.nan; fp = np.nan; pays = 0
                if o == 'pass':
                    cur = ei + 1
                    while cur < NG - 30:
                        o2, e2 = cycle(D, cur, 1.0, DD, DL, .05, dyn, 3100); td = (pd.Timestamp(grid[e2]) - t0).days
                        if td - tp > 180: break
                        if o2 == 'pass':
                            pays += 1; fp = td if np.isnan(fp) else fp; cur = e2 + 1
                        else: break
                rows.append(dict(u=u, Ge=Ge, dyn=dyn, start=sd, out=o, tpass=tp, tfirst=fp, pays=pays))
            print(u, Ge, 'done', flush=True)
R = pd.DataFrame(rows); R['half'] = np.where(R.start.dt.year <= 2021, '18-21', '22+'); R.to_pickle(SP + 'r445.pkl')
print('\nU  Ge dyn | passed<=30/60/90d | 1st payout<=60/90/120d | eval breach% open% | net$/att all | 18-21 | 22+')
for (u, Ge, dyn), g in R.groupby(['u', 'Ge', 'dyn']):
    a = lambda x, c, d: (x[c] <= d).mean() * 100; net = lambda x: (x.pays * 450 - FEEA).mean(); h1, h2 = g[g.half == '18-21'], g[g.half == '22+']
    print(f"{u} {Ge} {'Y' if dyn else 'n'} | {a(g,'tpass',30):3.0f} {a(g,'tpass',60):3.0f} {a(g,'tpass',90):3.0f} | {a(g,'tfirst',60):3.0f} {a(g,'tfirst',90):3.0f} {a(g,'tfirst',120):3.0f} | {(g.out=='breach').mean()*100:3.0f} {(g.out=='open').mean()*100:3.0f} | {net(g):+6.0f} | {net(h1):+6.0f} | {net(h2):+6.0f}")
