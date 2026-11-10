"""Nov 10 2026: WHICH NOISE-AREA BREAKOUTS ARE WORTH TAKING? (Rey: 'add volatility models, distance from VWAP, volume / candle speed so we take the bigger trades, not a bunch of noise breakouts').
DECLARED BEFORE RESULTS. Real NQ 5m 2014 - Jan 2026, the paper noise-area entries (xs/nq_noise_onwide.py: bands from the prior 14 sessions, decisions at HH:00 / HH:30 10:00-15:30, next-5m-open fill,
1 pt cost), exit = close back through max(UB, VWAP) / min(LB, VWAP) or a 40-pt hard stop or the 15:55 close. One row per trade.
FEATURES at the decision bar (all known then):  BRK  how far the close is beyond the band in sigma units (breakout strength) | VWD  distance from VWAP in the trade's direction / 20-day average range |
  SPD  last 30 minutes' move in the trade's direction / 20-day average range (speed into the break) | BAR  the breakout bar's range / its 20-bar average (candle size) |
  RVOL  last 30 minutes' volume / the mean of the same 30 minutes over the prior 14 sessions | VIX  prior-day VIX level | VOLR  20-day average range / its 120-day median (vol regime) |
  WIDE  Globex range in its top third (xs/nq_edge_hunt.py) | GAP  trade direction agrees with the open gap | VEC  the 15m vector (QQQ premarket + RTH) agrees | TOD  entry before 11:00.
SELECTION: each feature's cut = its terciles on 2014-21 trades only; 'top third vs the rest' (VIX / VOLR also bottom third; WIDE / GAP / VEC / TOD binary). PASS = same sign in 2014-21 and 2022-Jan26 and |t| >= 2.0 in both.
Then: score = the number of passing features a trade has (direction fixed by the 2014-21 sign); take only score >= s, or size x(1 + score): pts per trade, trades per day, pts per day at 1 contract, by period."""
import numpy as np, pandas as pd, importlib.util, glob, sys, warnings; warnings.filterwarnings('ignore')
sys.argv = ['x', '2']; sys.path.insert(0, '/home/claude/fartpie/xs')
sp = importlib.util.spec_from_file_location('h', '/home/claude/fartpie/xs/nq_edge_hunt.py'); h = importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
_nw = open('/home/claude/fartpie/xs/nq_noise_onwide.py').read(); exec(_nw[:_nw.index('def run(day')])
src = open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0]; exec(src)
H = pd.read_parquet(glob.glob('/home/claude/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
mm = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(mm >= 240) & (mm < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); REG = vec(H.rename(columns=str.lower)); CL = (H.dt + pd.Timedelta(minutes=15)).values
def reg_day(day):
    t = pd.date_range(day + pd.Timedelta(hours=9, minutes=35), periods=78, freq='5min'); i = np.searchsorted(CL, t.values, 'right') - 1
    return np.where(i >= 0, REG[np.maximum(i, 0)], 0)
VIX = pd.read_parquet('/home/claude/fartpie/data/yahoo/daily_v/VIX.parquet').close
onr = D.OH - D.OL; WIDE = (onr > onr.shift(1).rolling(20).quantile(2 / 3)); VOLR = D.rng20 / D.rng.shift(1).rolling(120).median()
VOLK = {}                                                                                  # per day volume at each bar for the relative-volume feature
for dy in DAYS: VOLK[dy] = DAYS[dy].v.values.astype(float)
idx = list(DAYS); ROWS = []
for i, day in enumerate(idx):
    if i < 15: continue
    g = DAYS[day]; o, hi, lo, c = (g[x].values for x in 'ohlc'); hist = [MV[x] for x in idx[i - 14:i] if MV[x] is not None]
    if len(g) < 78 or len(hist) < 14: continue
    op = o[0]; pcv = D.PC.get(day, np.nan); pc = pcv if np.isfinite(pcv) else op; sig = np.mean(hist, axis=0); rg = reg_day(day); vol = VOLK[day]
    vw = np.cumsum((hi + lo + c) / 3 * vol) / np.maximum(np.cumsum(vol), 1); adr = D.rng20.get(day, np.nan); hv = np.mean([VOLK[x] for x in idx[i - 14:i]], axis=0)
    gap = op - pc; pos = 0; ent = 0.0; start = 0; rec = None
    def close_trade(pnl):
        global rec
        rec['pts'] = pnl; ROWS.append(rec); rec = None
    for k in range(78):
        if pos and k >= start:
            st = ent - 40 * pos
            if (pos == 1 and lo[k] <= st) or (pos == -1 and hi[k] >= st): close_trade(-41.0); pos = 0
        if k in K and k < 77:
            ub = max(op, pc) * (1 + sig[k]); lb = min(op, pc) * (1 - sig[k]); nx = o[k + 1]
            if pos == 1 and c[k] < max(ub, vw[k]): close_trade(nx - ent - 1); pos = 0
            elif pos == -1 and c[k] > min(lb, vw[k]): close_trade(ent - nx - 1); pos = 0
            if pos == 0 and np.isfinite(adr):
                d = 1 if c[k] > ub else (-1 if c[k] < lb else 0)
                if d:
                    pos, ent, start = d, nx, k + 1; w0 = max(0, k - 5)
                    rec = dict(day=day, k=k, dir=d, BRK=((c[k] - ub) if d == 1 else (lb - c[k])) / max(op * sig[k], 1e-9), VWD=(c[k] - vw[k]) * d / adr, SPD=(c[k] - c[max(0, k - 6)]) * d / adr,
                               BAR=(hi[k] - lo[k]) / max(np.mean(hi[max(0, k - 20):k] - lo[max(0, k - 20):k]), 1e-9) if k > 3 else np.nan, RVOL=vol[w0:k + 1].sum() / max(hv[w0:k + 1].sum(), 1.0),
                               VIX=VIX.reindex([day], method='ffill').iloc[0] if day in VIX.index or True else np.nan, VOLR=VOLR.get(day, np.nan), WIDE=bool(WIDE.get(day, False)),
                               GAP=bool(np.sign(gap) == d), VEC=bool(rg[k] == d), TOD=bool(k < 15))
    if pos: close_trade((c[77] - ent) * pos - 1)
X = pd.DataFrame(ROWS); X['A'] = X.day < '2022-01-01'; print(f'trades {len(X):,} on {X.day.nunique():,} days | avg {X.pts.mean():+.2f} pts / trade (2014-21 {X.pts[X.A].mean():+.2f}, 2022-26 {X.pts[~X.A].mean():+.2f}) | per day {X.pts.sum() / len(DAYS):+.2f}\n', flush=True)
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 10 and len(b) > 10 else np.nan
CONT = ['BRK', 'VWD', 'SPD', 'BAR', 'RVOL', 'VIX', 'VOLR']; BIN = ['WIDE', 'GAP', 'VEC', 'TOD']; FL = {}
for f in CONT:
    q1, q2 = X.loc[X.A, f].quantile([1 / 3, 2 / 3]); FL[f + ' top'] = X[f] >= q2
    if f in ('VIX', 'VOLR'): FL[f + ' bottom'] = X[f] <= q1
for f in BIN: FL[f] = X[f].astype(bool); FL['not ' + f] = ~X[f].astype(bool) if f in ('WIDE', 'TOD') else None
FL = {k: v for k, v in FL.items() if v is not None}; PASS = {}
print('=== flag | n A / B | avg pts per trade flag vs rest, 2014-21 | 2022-26 | t | PASS ===')
for k, f in FL.items():
    a1, b1, a2, b2 = X.pts[X.A & f], X.pts[X.A & ~f], X.pts[~X.A & f], X.pts[~X.A & ~f]; ta, tb = tt(a1, b1), tt(a2, b2); da, db = a1.mean() - b1.mean(), a2.mean() - b2.mean()
    ok = np.isfinite(ta) and np.isfinite(tb) and np.sign(da) == np.sign(db) and abs(ta) >= 2 and abs(tb) >= 2
    if ok: PASS[k] = f
    print(f'  {k:<12} n {len(a1):4}/{len(a2):4} | {a1.mean():+6.1f} vs {b1.mean():+6.1f} | {a2.mean():+6.1f} vs {b2.mean():+6.1f} | t {ta:+5.2f} {tb:+5.2f} | {"PASS" if ok else ""}')
print('\nPASSING:', list(PASS), flush=True)
if PASS:
    X['score'] = sum(f.astype(int) for f in PASS.values()); nd = len(DAYS); nA = (DAYS < '2022-01-01').sum()
    print('\n=== by score: trades | avg pts / trade 2014-21 | 2022-26 | trades per day ===')
    for s in sorted(X.score.unique()): z = X[X.score == s]; print(f'  score {s}: {len(z):4} | {z.pts[z.A].mean():+6.1f} | {z.pts[~z.A].mean():+6.1f} | {len(z) / nd:.2f}')
    print('\n=== take only score >= s, or size x(1 + score): pts / day at 1 contract, all | 2014-21 | 2022-26 | worst trade ===')
    for lab, w in [('all trades (N0)', X.pts)] + [(f'score >= {s}', X.pts.where(X.score >= s, 0.0)) for s in range(1, int(X.score.max()) + 1)] + [('size x(1 + score)', X.pts * (1 + X.score))]:
        print(f'  {lab:<20} {w.sum() / nd:+6.2f} | {w[X.A].sum() / nA:+6.2f} | {w[~X.A].sum() / (nd - nA):+6.2f} | {w.min():+6.0f}')
X.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/noise_trades.pkl')
