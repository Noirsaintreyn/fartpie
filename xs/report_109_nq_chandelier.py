"""Nov 15 2026 NQ: VECTOR + CHANDELIER EXIT + SUPERTREND AGREEMENT (Rey: 'try this on the prop or large cap since 1h doesn't work'). Declared before results. Data: nq5_drive_et.parquet (NQ 5m, 2017-04..2026-07, ET, continuous), resampled to 15m / 1H / 4H bars on the full 24h session (offset 0). Vector (vector_pine.run_state_machine, same params) flips both ways. Trade = 1 contract, enter at the CLOSE of the flip bar in the flip direction (long on bull flip, short on bear flip), cost 1.0 NQ point round trip (declared), NQ POINTS per contract. Exits: (A) vector reverse: hold until the opposite flip (always in market); (B) chandelier: exit when the close crosses the extreme-since-entry -/+ M x ATR(22) (M=3 primary, 4 and 5 sensitivity), flat until the next vector flip; the opposite vector flip is always a backstop. (C) SUPERTREND AGREEMENT filter: only take flips where SuperTrend(10,3) is on the same side at the flip bar close; exit rule A. TRAIN 2017-04..2021-12, TEST 2022-01..2026-07; report per timeframe, per era: trades, mean/median pts, win rate, total pts, max drawdown pts (closed-trade equity), long vs short mean, and the bar-count held. Lead bar: positive mean pts net of 1.0 on BOTH eras with TEST t >= 2 (trade-clustered). Not the prop algo: this is the plain vector on NQ, so it says whether the exit/filter ideas transfer to NQ, not how the prop strategy behaves."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
d = pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').set_index('dt').sort_index(); COST = 1.0
def rma(x, n):
    o = np.full(len(x), np.nan); o[n - 1] = np.nanmean(x[:n]); a = 1 / n
    for k in range(n, len(x)): o[k] = o[k - 1] * (1 - a) + x[k] * a
    return o
def tr_(h, l, c): pc = np.r_[c[0], c[:-1]]; return np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
def supertrend(h, l, c):
    n = len(c); atr = rma(tr_(h, l, c), 10); hl2 = (h + l) / 2; ub = hl2 + 3 * atr; lb = hl2 - 3 * atr; up = np.full(n, np.nan); lo = np.full(n, np.nan); dr = np.zeros(n)
    for k in range(1, n):
        if np.isnan(atr[k]): continue
        if np.isnan(lo[k - 1]): lo[k] = lb[k]; up[k] = ub[k]; dr[k] = 1; continue
        lo[k] = lb[k] if (lb[k] > lo[k - 1] or c[k - 1] < lo[k - 1]) else lo[k - 1]; up[k] = ub[k] if (ub[k] < up[k - 1] or c[k - 1] > up[k - 1]) else up[k - 1]
        dr[k] = 1 if (dr[k - 1] == -1 and c[k] > up[k - 1]) else (-1 if (dr[k - 1] == 1 and c[k] < lo[k - 1]) else dr[k - 1])
    return dr
def run(B, mode, M=3, st_filter=False):
    reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); h = B.high.values; l = B.low.values; c = B.close.values; n = len(B); a = rma(tr_(h, l, c), 22); sd = supertrend(h, l, c); idx = B.index; trades = []
    i = 1
    flips = [k for k in range(1, n) if reg[k] != 0 and reg[k] != reg[k - 1]]
    for fi, k in enumerate(flips):
        s = reg[k]
        if s not in (1, -1): continue
        if st_filter and sd[k] != s: continue
        nxt = flips[fi + 1] if fi + 1 < len(flips) else n - 1; e = c[k]; end = nxt
        if mode == 'chand':
            ext = h[k] if s == 1 else l[k]; end = nxt
            for q in range(k + 1, nxt + 1):
                ext = max(ext, h[q]) if s == 1 else min(ext, l[q])
                if not np.isnan(a[q]) and ((s == 1 and c[q] < ext - M * a[q]) or (s == -1 and c[q] > ext + M * a[q])): end = q; break
        pts = (c[end] - e) * s - COST; trades.append((idx[k], s, pts, end - k))
    return pd.DataFrame(trades, columns=['t', 's', 'pts', 'bars'])
def rep(T, lab):
    for era, z in (('TRAIN', T[T.t < '2022-01-01']), ('TEST ', T[T.t >= '2022-01-01'])):
        if len(z) < 10: print(f'   {lab} {era} n {len(z)}'); continue
        eq = z.pts.cumsum(); dd = (eq.cummax() - eq).max(); t = z.pts.mean() / (z.pts.std() / np.sqrt(len(z)))
        print(f'   {lab:22s} {era} n {len(z):5d} mean {z.pts.mean():+7.2f} med {z.pts.median():+7.2f} win {(z.pts>0).mean()*100:4.1f}% total {z.pts.sum():+9.0f} maxDD {dd:8.0f} t {t:+5.2f} | long {z[z.s==1].pts.mean():+7.2f} short {z[z.s==-1].pts.mean():+7.2f} | bars {z.bars.mean():6.1f}')
for tf in ('15min', '60min', '240min'):
    B = d.resample(tf).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open']); print(f'\n== NQ {tf} bars {len(B)} ({B.index[0].date()} to {B.index[-1].date()}) NQ points per contract, cost {COST} pt')
    rep(run(B, 'rev'), 'vector reverse (A)')
    for M in (3, 4, 5): rep(run(B, 'chand', M), f'chandelier {M}x (B)')
    rep(run(B, 'rev', st_filter=True), 'ST-agree + reverse (C)'); rep(run(B, 'chand', 3, True), 'ST-agree + chand3')
