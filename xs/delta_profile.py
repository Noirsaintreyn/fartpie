"""Nov 9 2026: VOLUME PROFILE vs DELTA PROFILE LEVELS on MNQ (Rey). DECLARED BEFORE RESULTS.
Data: MNQ TBBO 1-minute file, sign FIXED (true delta = -signed_vol), 18 Dec 2024 - 18 Nov 2025, RTH 09:30-16:00. Each minute's volume and delta go to its VWAP, 5-pt bins.
Levels from the PRIOR RTH session:
  VP POC (most volume) | VP HVN (2 biggest other volume peaks, > 15 pts from the POC) | VP LVN (2 deepest volume troughs in the middle 70% of the range)
  DP BUY node (bin with the most net buying) | DP SELL node (most net selling)
  VP POC/HVN split by the node's own delta (3 bins): buyers won the node vs sellers won it.
Test in the next RTH session (09:45-15:30; roll weeks skipped): the FIRST touch of each level (the previous minute fully on one side). From above = support test, from below =
resistance test. Fade at the level with a symmetric bracket (15/15 and 30/30, both in one minute = loss, 1 pt cost, out 15:55): hold rate and points per touch.
PLACEBO: the same level moved 10-40 pts (random side, 5 draws) - a level only matters if it beats its placebo. Halves: Dec 2024 - May 2025 / Jun - Nov 2025."""
import pandas as pd, numpy as np
M = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt')
M['delta'] = -M.signed_vol.fillna(M.buy_vol.fillna(0) - M.sell_vol.fillna(0)); M['vol'] = M.buy_vol.fillna(0) + M.sell_vol.fillna(0)
M['px'] = M.vwap.where(M.vwap > 0, M.last_price); M['m'] = M.dt.dt.hour * 60 + M.dt.dt.minute; M['day'] = M.dt.dt.normalize()
R = M[(M.m >= 570) & (M.m < 960)]; G = {pd.Timestamp(d): g for d, g in R.groupby('day') if len(g) > 300}; DAYS = sorted(G)
ROLL = [('2024-12-16', '2024-12-20'), ('2025-03-13', '2025-03-21'), ('2025-06-12', '2025-06-20'), ('2025-09-11', '2025-09-19')]
inroll = lambda d: any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in ROLL); BIN = 5
def levels(g):
    b = np.floor(g.px / BIN) * BIN; vp = g.vol.groupby(b).sum(); dp = g.delta.groupby(b).sum()
    idx = np.arange(vp.index.min(), vp.index.max() + BIN, BIN); vp = vp.reindex(idx, fill_value=0); dp = dp.reindex(idx, fill_value=0)
    if len(vp) < 12: return []
    v = vp.rolling(3, center=True, min_periods=1).mean().values; ix = vp.index.values; dv = dp.values; poc = int(np.argmax(vp.values)); out = []
    nd = lambda k: dv[max(0, k - 1):k + 2].sum()
    out.append(('VP POC', ix[poc] + BIN / 2)); out.append(('VP POC/HVN, buyers won' if nd(poc) > 0 else 'VP POC/HVN, sellers won', ix[poc] + BIN / 2))
    pk = sorted([k for k in range(2, len(v) - 2) if v[k] == v[k - 2:k + 3].max() and abs(ix[k] - ix[poc]) > 15], key=lambda k: -v[k])[:2]
    for k in pk: out.append(('VP HVN', ix[k] + BIN / 2)); out.append(('VP POC/HVN, buyers won' if nd(k) > 0 else 'VP POC/HVN, sellers won', ix[k] + BIN / 2))
    tr = sorted([k for k in range(3, len(v) - 3) if v[k] == v[k - 2:k + 3].min() and .15 * len(v) < k < .85 * len(v)], key=lambda k: v[k])[:2]
    for k in tr: out.append(('VP LVN', ix[k] + BIN / 2))
    out.append(('DP BUY node', ix[int(np.argmax(dv))] + BIN / 2)); out.append(('DP SELL node', ix[int(np.argmin(dv))] + BIN / 2)); return out
def touch(g, L, br):
    lp = g.last_price.values; px = g.px.values; mm = g.m.values; n = len(lp)
    hi = np.maximum.reduce([np.r_[lp[0], lp[:-1]], lp, px]); lo = np.minimum.reduce([np.r_[lp[0], lp[:-1]], lp, px])
    for k in range(1, n):
        if mm[k] < 585 or mm[k] > 930: continue
        if not (lo[k] <= L <= hi[k]): continue
        if lo[k - 1] > L: side = 1
        elif hi[k - 1] < L: side = -1
        else: continue
        T, S = L + side * br, L - side * br
        for j in range(k + 1, n):
            if mm[j] >= 955: return side, (lp[j - 1] - L) * side - 1.0
            st = (lo[j] <= S) if side > 0 else (hi[j] >= S); tg = (hi[j] >= T) if side > 0 else (lo[j] <= T)
            if st: return side, -br - 1.0
            if tg: return side, br - 1.0
        return side, (lp[-1] - L) * side - 1.0
    return None
if __name__ == '__main__':
    rng = np.random.default_rng(7); rows = []
    for a, b in zip(DAYS[:-1], DAYS[1:]):
        if (b - a).days > 4 or inroll(a) or inroll(b): continue
        for typ, L in levels(G[a]):
            for br in (15, 30):
                r_ = touch(G[b], L, br)
                if r_: rows.append((b, typ, 'real', br, r_[0], r_[1]))
                for _ in range(5):
                    r_ = touch(G[b], L + rng.choice([-1, 1]) * rng.uniform(10, 40), br)
                    if r_: rows.append((b, typ, 'placebo', br, r_[0], r_[1]))
    X = pd.DataFrame(rows, columns=['day', 'typ', 'kind', 'br', 'side', 'pts']); X['h'] = np.where(X.day < pd.Timestamp('2025-06-01'), '1st', '2nd'); X['held'] = X.pts > 0
    print(f'{X[X.kind == "real"].day.nunique()} sessions with touches | fade at the level: HOLD RATE, points per touch, 1st half / 2nd half | real vs placebo (same level moved 10-40 pts)', flush=True)
    for br in (15, 30):
        print(f'\n  bracket {br}/{br}', flush=True)
        for typ in ('VP POC', 'VP HVN', 'VP LVN', 'DP BUY node', 'DP SELL node', 'VP POC/HVN, buyers won', 'VP POC/HVN, sellers won'):
            for side, sl in ((1, 'from above (support)'), (-1, 'from below (resistance)')):
                cells = []
                for kind in ('real', 'placebo'):
                    g = X[(X.br == br) & (X.typ == typ) & (X.side == side) & (X.kind == kind)]; a, b = g[g.h == '1st'], g[g.h == '2nd']
                    cells.append(f'{kind} {a.held.mean():3.0%} {a.pts.mean():+5.1f} (n{len(a):3d}) / {b.held.mean():3.0%} {b.pts.mean():+5.1f} (n{len(b):3d})')
                print(f'    {typ:<24} {sl:<24} ' + ' | '.join(cells), flush=True)
