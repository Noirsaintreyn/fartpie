"""Nov 18 2026 report 295 (Rey: pick better flips than 'the random 6 in order + ratio'). Rebuild the live 15m-vector trades for the 54 names from lse_ohlcv_cache 15m bars (04:00-16:00 ET), live model: 15m bull flip -> entry at next RTH open,
exit 15m bear flip (sticky 1h stop once +2% with 1h bull seen), 5% peak lock at 50% giveback, 2bp. h1track=False (15m entry track) only. Feature table at the flip bar (all known before the entry):
  room   = (close - vector ratchet level)/close      (stop distance as the vector sees it)        DECLARED: LOW first (tight stop, more R)
  bearlen= bars in the bear regime before this flip  (compression length)                          DECLARED: HIGH first
  s1     = 1h vector already bull at the flip bar                                                  DECLARED: aligned first
  rvol   = flip-bar volume / 20-bar mean                                                           DECLARED: HIGH first
  rngatr = flip-bar range / ATR14(15m)                                                             DECLARED: HIGH first
  hi52   = prior close / 252-day high                                                              DECLARED: HIGH first
Tests: (a) per-trade mean return by feature third (train 2019-22 | test 2023-26); (b) the 6-slot sim with each feature as the pick order for simultaneous flips (live = ratio), chosen on 2019-22, judged on 2023-26."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def smr(x): r = run_state_machine(x[['open', 'high', 'low', 'close']].assign(datetime=x.dt)); return np.asarray(r[0]), np.asarray(r[1])
def build(f):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute
    H['reg'], H['lvl'] = smr(H)
    x = H[H.m < 960].set_index('dt')[['open', 'high', 'low', 'close']]; g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']
    s = pd.DataFrame({'known': r.dt + pd.Timedelta('1h'), 'r': smr(r)[0]}).sort_values('known')
    H['s1'] = pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), s, left_on='t', right_on='known', direction='backward').r.fillna(0).values
    H['day'] = H.dt.dt.normalize(); rth = H[(H.m >= 570)]; dc = rth.groupby('day').close.last(); H['hi52'] = (dc.shift(1) / dc.shift(1).rolling(252, min_periods=120).max()).reindex(H.day).values
    pc = dc.shift(1); H['pc'] = pc.reindex(H.day).values
    H['rv'] = H.volume / H.volume.rolling(20, min_periods=10).mean(); tr = np.maximum(H.high - H.low, np.maximum((H.high - H.close.shift(1)).abs(), (H.low - H.close.shift(1)).abs())); H['rngatr'] = (H.high - H.low) / tr.rolling(14, min_periods=7).mean()
    return H
def trades(H, tk):
    reg, s1, o, c, hi, d, m, lvl, rv, rg, h52 = (H[k].values for k in ('reg', 's1', 'open', 'close', 'high', 'dt', 'm', 'lvl', 'rv', 'rngatr', 'hi52'))
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False; KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    runlen = np.zeros(len(H)); cnt = 0; prev = 0; prevlen = np.zeros(len(H))
    for i in range(len(H)):
        if reg[i] == prev: cnt += 1
        else: prevlen[i] = cnt; cnt = 1; prev = reg[i]
        runlen[i] = cnt
    last_bear = 0; bl = np.zeros(len(H))
    for i in range(len(H)):
        if i > 0 and reg[i] == 1 and reg[i - 1] != 1: last_bear = prevlen[i]
        bl[i] = last_bear
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            if KR[p] == 1 and KR[p - 1] != 1:
                k = g[p - 1]; pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False
                feat = dict(room=(c[k] - lvl[k]) / c[k], bearlen=bl[k], s1=float(KS[p] == 1), rvol=rv[k], rngatr=rg[k], hi52=h52[k], flipm=m[k])
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = h1 or now
        ex = (KS[p] == -1) if h1 else (KR[p] == -1); pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if ex: T.append(dict(tk=tk, entry=d[e], exit=d[j], ret=o[j] / px - 1 - 2e-4, **feat)); pos = False
    return T
if __name__ == '__main__':
    W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); tks = sorted(W[W.sp].tk.unique()); fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    rows = []
    for tk in tks:
        if tk not in fs: continue
        rows += trades(build(fs[tk]), tk)
        if len(rows) % 500 < 40: print(tk, len(rows), flush=True)
    A = pd.DataFrame(rows); A.to_parquet(SP + 'flips295.parquet'); print('trades', len(A), A.entry.min(), A.entry.max())
    Wt = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')]; print('wide_T 15m-track base+sticky trades', len(Wt)); 
    k = A.merge(Wt, on=['tk', 'entry'], suffixes=('', '_w')); print('matched entries', len(k), 'corr of ret', k.ret.corr(k.ret_w).round(3), 'mean |diff|', (k.ret - k.ret_w).abs().mean().round(5))
