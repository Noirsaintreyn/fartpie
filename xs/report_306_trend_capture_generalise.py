"""Nov 18 2026 report 306. Does the algo's core mechanism -- trend-day capture -- exist outside NQ?  (Evidence so far: 282 says the raw vector flip is NQ/QQQ-only; report 275/283/284 say the binding problem is DROUGHTS and that copies of one stream clump; next lever named = uncorrelated streams.)
THEORY (declared before running). Persistent intraday drift arises from (a) basket/index flow (equity index futures) and (b) macro-news repricing that dealers/arbs unwind slowly (rates/metals/energy). A generic engine with NO fitted parameter should show positive expectancy where that drift is strong and should be uncorrelated across asset classes on a daily basis.
ENGINE (fixed, one trade/day): 15m bars. Opening range = first 30 min after the instrument's primary-session open (indices 09:30 ET, GC and SI 08:15 (pit open 08:20/08:25 rounded to the 15m grid), HG 08:00, CL 09:00). Enter at the close of the first 15m bar within the next 4h that closes beyond the OR (long above, short below). Stop = 0.20 x ADR14 (mean session range of the prior 14 sessions), NO target, exit at the session close (indices 16:00, metals 13:30, CL 14:30). Cost 1.5 bp round trip. Placebo = same entry bars with coin-flip direction (200 draws, expectation).
Why it would fail: strong mean-reversion microstructure (low-vol, range-bound years: 2019, 2023), mid-session liquidity gaps in commodities; the edge in NQ may be a megacap-momentum/0DTE artefact (282).
Split: train 2019-21 | test 2018 + 2022-Nov 2026. Day-clustered t."""
import numpy as np, pandas as pd, glob, os, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/lsefut/lse_ohlcv_cache/'
CFG = {'NASDAQ100.F': (570, 960), 'SP500.F': (570, 960), 'DOW.F': (570, 960), 'RUSSELL2000.F': (570, 960), 'GOLD.F': (495, 810), 'SILVER.F': (495, 810), 'COPPER.F': (480, 810), 'CRUDEOIL.F': (540, 870)}
rng = np.random.default_rng(7); OUT = {}; DAILY = {}
def load(sym):
    f = glob.glob(SP + f'{sym}_15m_*.parquet')
    if not f: return None
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H['day'] = H.dt.dt.normalize(); return H
for sym, (o, c) in CFG.items():
    H = load(sym)
    if H is None: print(sym, 'not available yet'); continue
    S = H[(H.m >= o) & (H.m < c) & (H.dt.dt.dayofweek < 5)]; rows = []; rngs = []
    for d, x in S.groupby('day'):
        x = x.sort_values('m')
        if len(x) < (c - o) / 15 * .85 or x.m.iloc[0] != o: continue
        rngs.append(x.high.max() - x.low.min()); adr = np.mean(rngs[-15:-1]) if len(rngs) > 14 else np.nan
        if np.isnan(adr): continue
        orh = x.high.iloc[:2].max(); orl = x.low.iloc[:2].min(); rest = x.iloc[2:]; cut = rest[rest.m < o + 270]; fin = x.close.iloc[-1]; ent = None
        for i, (_, b) in enumerate(cut.iterrows()):
            if b.close > orh: ent = (1, b.close, b.m); break
            if b.close < orl: ent = (-1, b.close, b.m); break
        if ent is None: continue
        dr, px, m0 = ent; after = rest[rest.m > m0]; stop = px - dr * .2 * adr; ex = fin
        for _, b in after.iterrows():
            if (dr == 1 and b.low <= stop) or (dr == -1 and b.high >= stop): ex = stop; break
        gross = (ex / px - 1) * dr * 1e4; ex_up = None
        # placebo: opposite direction on same bars (stop mirrored) -> expectation of coin flip = mean(long, short)
        stop2 = px + dr * .2 * adr; ex2 = fin
        for _, b in after.iterrows():
            if (dr == 1 and b.high >= stop2) or (dr == -1 and b.low <= stop2): ex2 = stop2; break
        gross2 = (ex2 / px - 1) * (-dr) * 1e4
        rows.append(dict(day=d, dir=dr, bps=gross - 1.5, plc=(gross + gross2) / 2 - 1.5, adr_bps=adr / px * 1e4, risk_bps=.2 * adr / px * 1e4))
    if not rows: print(sym, 'no trades'); continue
    T = pd.DataFrame(rows).set_index('day'); T['tr'] = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); OUT[sym] = T; DAILY[sym] = T.bps
def tt(s): return f'{s.mean():+6.1f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+.1f})' if len(s) > 5 else 'n/a'
print('bps per trade net of 1.5bp (placebo coin-flip direction in brackets); n; avg stop (bps)')
for sym, T in OUT.items():
    a, b = T[T.tr], T[~T.tr]
    print(f'{sym:14s} train {tt(a.bps)} [{a.plc.mean():+5.1f}] n{len(a):4d} | test {tt(b.bps)} [{b.plc.mean():+5.1f}] n{len(b):4d} | stop {T.risk_bps.mean():5.0f}bp win {100*(T.bps>0).mean():.0f}%')
print('\nby year, bps per trade:'); Y = pd.DataFrame({s: T.bps.groupby(T.index.year).mean() for s, T in OUT.items()}).round(1); print(Y.to_string())
D = pd.DataFrame(DAILY); print('\ndaily P&L correlation (days both traded):'); print(D.corr(min_periods=100).round(2).to_string())
D.to_parquet(SP.replace('lse_ohlcv_cache/', '') + 'd306.parquet')
