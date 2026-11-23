"""Nov 22 2026 report 372. Rey: 'try other strategies; an ORB or a noise band exists, my algo is price action too, stop only testing filters on the vector'.
Two published intraday price-action systems applied to the same 54 large-cap names, 15m RTH bars (26 per session, complete sessions only), 2019-2026, train 2019-21 | test 2022-26:
 ORB (Zarattini-Barbon-Aziz 'stocks in play' opening range breakout): bar0 = 09:30 bar.  Stock in play = bar0 volume / mean bar0 volume of the prior 14 sessions >= 1.0, the top 20 names per day by that ratio.  Direction = colour of bar0 (up = long, down = short); entry at the OPEN of bar1 (no look-ahead); protective stop = the other end of bar0 (low for longs, high for shorts) checked on bar highs/lows, filled at the stop; otherwise exit at the 15:45 bar close.  Reported long-only (his book is long-only) and long+short.
 NOISE BAND (Concretum 'Beat the market' intraday momentum): m_t = mean over the prior 14 sessions of |close_t / open_0 - 1| at the same bar t.  UB_t = max(today open, prior close) x (1 + m_t), LB_t = min(open, prior close) x (1 - m_t); VWAP of the session so far.  Long when the bar close is above UB and above VWAP (short mirror), entry at the next bar's open, one entry per name per day; exit at the next open once the close falls back below max(UB, VWAP) (longs; mirror for shorts) or at the 15:45 close.
 Both: cost 2bp per side (4bp round trip); a pessimistic 8bp round trip also shown.  Sleeve return per day = sum of trade returns / 20 (equal 1/20 gross per position, up to 20 positions, rest in cash), so the daily series is comparable with the vector book.
 Reported: per-trade mean (bp), win rate, train | test, sleeve Sharpe / return / DD, daily-return correlation with the live vector book, and a 50/50 blend with the book.  Declared pass: positive mean and sleeve Sharpe > 1 in BOTH halves after 4bp, and a correlation with the vector below 0.3 (the point of a second strategy is diversification)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); names = sorted(b.tk.unique()); NB = 26; K = 20; COST = .0004
orb_rows, nz_rows = [], []
for tk in names:
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    m = H.index.hour * 60 + H.index.minute; H = H[(m >= 570) & (m < 960)]; H['d'] = H.index.normalize(); cnt = H.groupby('d').size(); H = H[H.d.isin(cnt[cnt == NB].index)]
    days = H.d.unique(); n = len(days)
    if n < 100: continue
    O, Hh, L, C, V = (H[c].values.reshape(n, NB) for c in ('open', 'high', 'low', 'close', 'volume')); D = pd.DatetimeIndex(days)
    prevC = np.r_[np.nan, C[:-1, -1]]; v0 = V[:, 0]; rv0 = np.full(n, np.nan)
    for d in range(14, n): rv0[d] = v0[d] / v0[d - 14:d].mean() if v0[d - 14:d].mean() > 0 else np.nan
    ratio = np.abs(C / O[:, [0]] - 1); mt = np.full((n, NB), np.nan)
    for d in range(14, n): mt[d] = ratio[d - 14:d].mean(0)
    tp = (Hh + L + C) / 3; vw = np.cumsum(tp * V, 1) / np.maximum(np.cumsum(V, 1), 1)
    for d in range(15, n):
        # ORB
        if rv0[d] >= 1 and C[d, 0] != O[d, 0] and O[d, 1] > 0:
            side = 1 if C[d, 0] > O[d, 0] else -1; ent = O[d, 1]; stop = L[d, 0] if side == 1 else Hh[d, 0]; ex = C[d, -1]
            for t in range(1, NB):
                if (side == 1 and L[d, t] <= stop) or (side == -1 and Hh[d, t] >= stop): ex = stop; break
            orb_rows.append((tk, D[d], rv0[d], side, side * (ex / ent - 1)))
        # noise band
        if np.isnan(mt[d, 1]) or np.isnan(prevC[d]): continue
        hi_ref = max(O[d, 0], prevC[d]); lo_ref = min(O[d, 0], prevC[d]); pos = 0; ent = 0
        for t in range(1, NB - 1):
            ub = hi_ref * (1 + mt[d, t]); lb = lo_ref * (1 - mt[d, t])
            if pos == 0:
                if C[d, t] > ub and C[d, t] > vw[d, t]: pos, ent = 1, O[d, t + 1]
                elif C[d, t] < lb and C[d, t] < vw[d, t]: pos, ent = -1, O[d, t + 1]
                if pos: tr_t = t
            else:
                if t > tr_t and ((pos == 1 and C[d, t] < max(ub, vw[d, t])) or (pos == -1 and C[d, t] > min(lb, vw[d, t]))):
                    nz_rows.append((tk, D[d], rv0[d], pos, pos * (O[d, t + 1] / ent - 1))); pos = 0; break
        if pos != 0: nz_rows.append((tk, D[d], rv0[d], pos, pos * (C[d, -1] / ent - 1)))
    print(tk, len(orb_rows), len(nz_rows), end=' | ', flush=True)
cols = ['tk', 'date', 'rv0', 'side', 'gross']; ORB = pd.DataFrame(orb_rows, columns=cols); NZ = pd.DataFrame(nz_rows, columns=cols); ORB.to_parquet(SP + 'orb372.parquet'); NZ.to_parquet(SP + 'nz372.parquet'); print('\ntrades: ORB %d, noise %d' % (len(ORB), len(NZ)), flush=True)
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
def live(Bs): return Bs.assign(_k=Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable')
order = live; TKv, _ = sim_swap(B0, None); zv = spread(TKv); zv = zv[zv.index >= '2019-01-01']; zv.to_pickle(SP + 'zlive372.pkl')
def sleeve(T, cost, side=None, topk=True):
    T = T.copy()
    if side is not None: T = T[T.side == side]
    T['net'] = T.gross - cost
    if topk: T = T.sort_values(['date', 'rv0'], ascending=[True, False]).groupby('date').head(K)
    return T.groupby('date').net.sum() / K, T
def show(lab, T, side=None, cost=COST):
    s, X = sleeve(T, cost, side); s = s.reindex(zv.index).fillna(0); res = []
    for h, m_ in (('train', s.index < '2022-01-01'), ('test', s.index >= '2022-01-01')):
        x = s[m_]; tt = X[(X.date < '2022-01-01') if h == 'train' else (X.date >= '2022-01-01')]; res.append('%s: n %d mean %+.1fbp win %.0f%% | sleeve Sharpe %.2f ret %+.1f%%/yr' % (h, len(tt), tt.net.mean() * 1e4, (tt.net > 0).mean() * 100, x.mean() / x.std() * np.sqrt(252), x.mean() * 252 * 100))
    e = (1 + s).cumprod(); dd = (e / e.cummax() - 1).min(); corr = s.corr(zv); bl = .5 * s + .5 * zv
    print('%-30s %s | %s | DD %.1f%% corr(vector) %+.2f | 50/50 with vector Sharpe %.2f (vector alone %.2f)' % (lab, res[0], res[1], dd * 100, corr, bl.mean() / bl.std() * np.sqrt(252), zv.mean() / zv.std() * np.sqrt(252)), flush=True)
print('\n== ORB (top 20 stocks in play per day) ==')
show('ORB long-only, 4bp', ORB, 1); show('ORB long+short, 4bp', ORB); show('ORB long-only, 8bp', ORB, 1, .0008)
print('\n== Noise band (up to 20 names/day ranked by opening volume ratio) ==')
show('Noise long-only, 4bp', NZ, 1); show('Noise long+short, 4bp', NZ); show('Noise long-only, 8bp', NZ, 1, .0008)
print('ORB gross per trade (bp) by year long-only:', (ORB[ORB.side == 1].assign(y=lambda x: x.date.dt.year).groupby('y').gross.mean() * 1e4).round(1).to_dict())
print('Noise gross per trade (bp) by year long-only:', (NZ[NZ.side == 1].assign(y=lambda x: x.date.dt.year).groupby('y').gross.mean() * 1e4).round(1).to_dict())
