"""Nov 18 2026 report 312. MACRO/FLOW LAYER, first test: a CTA trend-following proxy (BofA monitor style: 3 trend windows x 4 stop sensitivities = 12 models per market; position = +1 uptrend and not stopped, -1 downtrend and not stopped, 0 stopped).
THEORY (declared before running). CTAs run ~$300bn of mechanical, vol-scaled trend positions. Their rebalancing is a forecastable flow: (H1) when the aggregate NQ trend position is rising or near a buy/sell trigger, next-day NQ RANGE/ADR is larger (flow adds to path, size not direction, as with IV/volume in 257); (H2) a SUPPORTED trend (CTA long, not stopped) has higher up-trend-day efficiency than an unsupported one, i.e. algo pts/day on trend days differ by CTA state; (H3) cross-asset: stretched rates/commodity trend states (TLT, crude) change the equity day-type distribution. Why it should work: positions are rule-based and public in structure; why it would stop: crowded so reflexive, our proxy windows/stops are guesses (BofA windows undisclosed), CTAs are one of many flows, and CTA AUM is small next to NQ turnover.
Proxy inputs: daily closes NQ, ES (futures), TLT (UST: long TLT trend = rates down), crude, gold, copper. Windows 21/63/126d; stops 3/4/5/6 x ATR14 below rolling-w peak (above trough for shorts). All features are known at yesterday's close (shift 1). Train 2019-21 | test 2018+2022-Nov 2026. Spearman, day-clustered nothing; terciles cut on train."""
import numpy as np, pandas as pd, glob, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def daily(path):
    H = pd.read_parquet(path); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H['day'] = H.dt.dt.normalize()
    g = H.groupby('day'); return pd.DataFrame({'c': g.close.last(), 'h': g.high.max(), 'l': g.low.min()})
U = {'NQ': SP + 'lsefut/lse_ohlcv_cache/NASDAQ100.F_15m_2017-04-01_2026-11-11.parquet', 'ES': SP + 'lsefut/lse_ohlcv_cache/SP500.F_15m_2017-04-01_2026-11-11.parquet', 'CL': SP + 'lsefut/lse_ohlcv_cache/CRUDEOIL.F_15m_2017-04-01_2026-11-11.parquet', 'GC': SP + 'lsefut/lse_ohlcv_cache/GOLD.F_15m_2017-04-01_2026-11-11.parquet', 'HG': SP + 'lsefut/lse_ohlcv_cache/COPPER.F_15m_2017-04-01_2026-11-11.parquet', 'TLT': glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/TLT_15m_*')[0]}
def cta(x):
    c = x.c; atr = (x.h - x.l).rolling(14).mean(); out = []
    for w in (21, 63, 126):
        tr = np.sign(c - c.shift(w)); pk = c.rolling(w).max(); tg = c.rolling(w).min()
        for k in (3, 4, 5, 6):
            long_ok = (tr > 0) & (c > pk - k * atr); short_ok = (tr < 0) & (c < tg + k * atr); out.append(long_ok.astype(float) - short_ok.astype(float))
    P = pd.concat(out, axis=1); return P.mean(axis=1), (P == 0).sum(axis=1), ((c - (c.rolling(63).max() - 4 * atr)) / atr)
F = pd.DataFrame(); 
for k, p in U.items():
    x = daily(p); pos, ns, dist = cta(x); F[k + '_pos'] = pos; F[k + '_dpos'] = pos.diff(); F[k + '_dist'] = dist; F[k + '_nstop'] = ns
F = F.shift(1)   # ex-ante
D = pd.read_parquet(SP + 'd305.parquet').join(F, how='inner'); D['rn'] = D.rng / D.adr; D['absnet_n'] = D.net.abs() / D.adr; D = D.dropna(subset=['NQ_pos', 'NQ_dpos', 'rn'])
tr = D.tr; q = lambda s: s[tr].quantile([1 / 3, 2 / 3]).values
print(f'days {len(D)} train {int(tr.sum())} test {int((~tr).sum())}; mean NQ CTA pos {D.NQ_pos.mean():+.2f}, share fully long {(D.NQ_pos > .8).mean():.0%}')
def row(f, y):
    a = D[tr][[f, y]].dropna(); b = D[~tr][[f, y]].dropna(); return f'{spearmanr(a[f], a[y])[0]:+.3f} | {spearmanr(b[f], b[y])[0]:+.3f}'
print('\n[H1] ex-ante CTA features -> today range/ADR (rho train | test)')
for f in ('NQ_pos', 'NQ_dpos', 'NQ_dist', 'NQ_nstop', 'ES_pos', 'TLT_pos', 'CL_pos', 'GC_pos'): D['a_' + f] = D[f].abs() if 'dpos' in f or 'pos' in f else D[f]; print(f'  {f:9s} signed {row(f, "rn")}   abs {row("a_" + f, "rn")}')
print('\n[H2] algo ALL pts/day by NQ CTA position tercile and by day efficiency (train | test)')
for lab, f in (('NQ_pos', 'NQ_pos'), ('NQ_dpos', 'NQ_dpos'), ('TLT_pos', 'TLT_pos'), ('CL_pos', 'CL_pos')):
    c = q(D[f]); D['b'] = np.where(D[f] <= c[0], 'low', np.where(D[f] >= c[1], 'high', 'mid'))
    for m in ('ALL', 'M1', 'V15a', 'ON+'): print(f'  {lab:8s} {m:5s} ' + '  '.join(f'{b} {D[tr & (D.b == b)][m].mean():+6.1f}|{D[~tr & (D.b == b)][m].mean():+6.1f}' for b in ('low', 'mid', 'high')))
print('\n[H2b] up-day probability and net/ADR by NQ CTA state (test only has all years): ')
for f in ('NQ_pos',):
    c = q(D[f]); D['b'] = np.where(D[f] <= c[0], 'low', np.where(D[f] >= c[1], 'high', 'mid'))
    for lab, m in (('train', tr), ('test', ~tr)): print(' ', lab, {b: f'net/ADR {D[m & (D.b == b)].net.div(D[m & (D.b == b)].adr).mean():+.2f} eff {D[m & (D.b == b)].effday.mean():.2f} n{int((m & (D.b == b)).sum())}' for b in ('low', 'mid', 'high')})
print('\n[2026] mean NQ CTA pos by month:'); print(D[D.index >= '2026-01-01'].NQ_pos.groupby(D[D.index >= '2026-01-01'].index.to_period('M')).mean().round(2).to_dict())
D.to_parquet(SP + 'd312.parquet')
