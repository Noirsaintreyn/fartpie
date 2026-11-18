"""Nov 18 2026 report 334. THE 5-MIN ORB (report 333, +0.14R/trade generic) CONDITIONED ON VOLATILITY / ATRP / GAP / ORDER FLOW / OPTIONS FLOW.  Declared BEFORE running.
Engine (unchanged): first RTH 5m candle high/low, R = candle range, stop at the other end, no target, exit 15:55 open, 0.75 pt cost, one trade/day, 2014-2026 on NQ 5m.
Every feature is known BEFORE the break (prior-day values, the overnight gap, or the first candle itself).  Cuts for terciles are from TRAIN (2019-21).  Directions fixed from mechanism:
 V1 compression = first-candle range / ADR14 (low = the candle is small relative to a normal day): LOW -> HIGHER R.   R = pts / candle range, so a small candle that precedes a normal day gives a big R by construction -> reported as R AND as points per trade so it is not a pure artefact.  Why it fails: tiny candles = tiny R denominators = costs/gaps dominate and stop-outs are noise.
 V2 ATRP = ADR14 / open in bps: HIGH -> higher POINTS per trade (range is forecastable, reports 253-270; a break has more room).  Fails when high ATRP is a mean-reverting shock regime.
 V3 VIX (prior close) tercile: HIGH -> higher points per trade; direction of R left open (stated: R would not change).
 G1 gap alignment: the break direction equals the sign of the overnight gap (open vs prior RTH close) -> HIGHER R (desk 'gap rule'; overnight order imbalance continues; Gao et al. intraday momentum).  Fails when gaps fill (2019-23 style).
 OF1 order-flow alignment (MNQ 1-min signed volume, 09:30-09:34): the break direction equals the sign of first-candle signed volume -> HIGHER R (aggressor flow = real demand, not a thin sweep).  DATA: only from 2024-12-18 -> internal split 2024-12..2025-09 | 2025-10..2026-09; cannot meet the 2019-21 train rule.
 OP1 options: QQQ net vanna (prior day) in its BOTTOM tercile (dealers short vol-sensitivity: spot/vol feedback amplifies) -> HIGHER R; OP2 net charm sign (time-decay delta flow into the close) -> sign of charm = break direction -> higher R; OP3 gex-wall ratio terciles, high -> LOWER R (pinning).  DATA to 2025-12.
Six families, ~9 tests => 1-2 spurious |t|>2 expected; admit only same sign in both halves, |t|>=2 in one, plus a mechanism."""
import numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
U = '/mnt/user-data/uploads/fartpie/'
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); rows = []; DAYR = {}; CL = {}
for d, x in N.groupby('day'):
    x = x.sort_values('m')
    if len(x) < 70 or x.m.iloc[0] != 570: continue
    DAYR[d] = x.high.max() - x.low.min(); CL[d] = x.close.iloc[-1]; hi, lo = x.high.iloc[0], x.low.iloc[0]; rg = hi - lo
    if rg < 2: continue
    rest = x.iloc[1:]; ex = rest[rest.m == 955].open.iloc[0] if (rest.m == 955).any() else rest.close.iloc[-1]
    for _, b in rest.iterrows():
        up, dn = b.high >= hi, b.low <= lo
        if up and dn: break
        if up or dn:
            dr = 1 if up else -1; ent = hi if up else lo; stop = lo if up else hi; xp = ex
            if (dr == 1 and b.low <= stop) or (dr == -1 and b.high >= stop): xp = stop
            else:
                for _, c in rest[rest.m > b.m].iterrows():
                    if (dr == 1 and c.low <= stop) or (dr == -1 and c.high >= stop): xp = stop; break
            rows.append(dict(day=d, dir=dr, rg=rg, pts=(xp - ent) * dr - .75, o=x.open.iloc[0])); break
T = pd.DataFrame(rows).set_index('day'); T['R'] = T.pts / T.rg
dr_ = pd.Series(DAYR); adr = dr_.rolling(14).mean().shift(1); T['comp'] = T.rg / adr.reindex(T.index); T['atrp'] = adr.reindex(T.index) / T.o * 1e4
cl = pd.Series(CL); T['gap'] = (T.o - cl.shift(1).reindex(T.index)) / adr.reindex(T.index); T['gap_al'] = np.sign(T.gap) == T.dir
vx = yf.download('^VIX', start='2013-06-01', end='2026-11-17', progress=False, auto_adjust=True)['Close'].squeeze(); T['vix'] = vx.shift(1).reindex(T.index, method='ffill')
tr = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); te = (T.index >= '2018-01-01') & ~tr
def f(m, col='R'):
    s = T.loc[m, col].dropna(); return '    n/a' if len(s) < 25 else f'{s.mean():+.3f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f},n{len(s):4d})'
def terc(col, lab, hi_better=None):
    q = T.loc[tr, col].quantile([1 / 3, 2 / 3]).values; lo_, mid_, hi_ = T[col] <= q[0], (T[col] > q[0]) & (T[col] < q[1]), T[col] >= q[1]
    print(f'  {lab:34s} R:   train low {f(tr & lo_)} mid {f(tr & mid_)} high {f(tr & hi_)}\n  {"":34s}       test  low {f(te & lo_)} mid {f(te & mid_)} high {f(te & hi_)}')
    print(f'  {"":34s} pts:  train low {f(tr & lo_, "pts")} high {f(tr & hi_, "pts")} | test low {f(te & lo_, "pts")} high {f(te & hi_, "pts")}')
print(f'ORB trades {len(T)} | all: train R {f(tr)} | test R {f(te)}')
print('\nVOLATILITY / ATRP (terciles, cuts from train)'); terc('comp', 'V1 compression (candle / ADR14)'); terc('atrp', 'V2 ATRP (ADR14/open, bps)'); terc('vix', 'V3 VIX prior close')
print('\nGAP ALIGNMENT'); print(f'  G1 break WITH the gap   R: train {f(tr & T.gap_al)} | test {f(te & T.gap_al)}   AGAINST: train {f(tr & ~T.gap_al)} | test {f(te & ~T.gap_al)}')
print(f'       pts: WITH train {f(tr & T.gap_al, "pts")} | test {f(te & T.gap_al, "pts")}   AGAINST train {f(tr & ~T.gap_al, "pts")} | test {f(te & ~T.gap_al, "pts")}')
# ---- order flow (2024-12 ->)
O = pd.read_parquet(U + 'mnq_orderflow_1min.parquet'); O['dt'] = pd.to_datetime(O.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); O['m'] = O.dt.dt.hour * 60 + O.dt.dt.minute; O['day'] = O.dt.dt.normalize()
F = O[(O.m >= 570) & (O.m <= 574)].groupby('day').agg(sv=('signed_vol', 'sum'), bv=('buy_vol', 'sum'), sl=('sell_vol', 'sum')); F['imb'] = F.sv / (F.bv + F.sl)
T = T.join(F[['sv', 'imb']]); T['of_al'] = np.sign(T.sv) == T.dir; Tf = T.dropna(subset=['sv']); print(f'\nORDER FLOW: {len(Tf)} trades with 1-min order flow ({Tf.index.min().date()} -> {Tf.index.max().date()}); internal split')
a, b = Tf.index < '2025-10-01', Tf.index >= '2025-10-01'
for lab, m in (('2024-12..2025-09', a), ('2025-10..2026-09', b)):
    s1, s0 = Tf[m & Tf.of_al].R, Tf[m & ~Tf.of_al].R; d = s1.mean() - s0.mean(); se = np.sqrt(s1.var() / len(s1) + s0.var() / len(s0)); print(f'  OF1 {lab}: break WITH first-candle aggressor flow R {s1.mean():+.3f} (n{len(s1)}) vs AGAINST {s0.mean():+.3f} (n{len(s0)}) diff {d:+.3f} (t{d / se:+.1f}); share aligned {Tf[m].of_al.mean():.0%}')
big = Tf.imb.abs() >= Tf.imb.abs().median()
for lab, m in (('first half', a), ('second half', b)):
    s1, s0 = Tf[m & Tf.of_al & big].R, Tf[m & ~(Tf.of_al & big)].R; print(f'  OF1b {lab}: aligned AND |imbalance| above median R {s1.mean():+.3f} (n{len(s1)}) vs everything else {s0.mean():+.3f} (n{len(s0)})')
# ---- options (to 2025-12)
V = pd.read_csv(U + 'options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date'); G = pd.read_csv(U + 'gex_walls_nq.csv', parse_dates=['date']).set_index('date')
T['vanna'] = V.net_vanna.shift(1).reindex(T.index, method='ffill'); T['charm'] = V.net_charm.shift(1).reindex(T.index, method='ffill'); T['gexr'] = G.ratio.shift(1).reindex(T.index, method='ffill')
T['ch_al'] = np.sign(T.charm) == T.dir; To = T[T.index <= '2025-12-15']
print('\nOPTIONS FLOW (QQQ vanna/charm, NQ gex-wall ratio; prior day)'); T2 = T; T = To
tr = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); te = (T.index >= '2018-01-01') & ~tr
terc('vanna', 'OP1 net vanna (low = dealers short)'); terc('gexr', 'OP3 gex wall ratio')
print(f'  OP2 break WITH charm sign R: train {f(tr & T.ch_al)} | test {f(te & T.ch_al)}   AGAINST: train {f(tr & ~T.ch_al)} | test {f(te & ~T.ch_al)}')
# ---- follow-up: gap-aligned ORB only (the one survivor): difference test, by year, effect on the stack
T = T2; tr = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); te = (T.index >= '2018-01-01') & ~tr; print('\nGAP-ALIGNED ORB: with-minus-against difference (R)')
for lab, m in (('train', tr), ('test ', te)):
    a, b = T[m & T.gap_al].R, T[m & ~T.gap_al].R; d = a.mean() - b.mean(); print(f'  {lab}: {d:+.3f}R (t{d / np.sqrt(a.var() / len(a) + b.var() / len(b)):+.1f}); share of days aligned {T[m].gap_al.mean():.0%}')
print('  by year, R/trade WITH the gap:', T[T.gap_al].R.groupby(T[T.gap_al].index.year).mean().round(2).to_dict()); print('  by year, R/trade AGAINST:     ', T[~T.gap_al].R.groupby(T[~T.gap_al].index.year).mean().round(2).to_dict())
print('  stressed cost 2.0 pts, WITH the gap: train', f(tr & T.gap_al, 'pts'), '(pts at .75) -> subtract another 1.25 pts per trade')
big = T.gap.abs() >= T.loc[tr, 'gap'].abs().median(); print(f'  WITH the gap and |gap| above median R: train {f(tr & T.gap_al & big)} | test {f(te & T.gap_al & big)};   with a small gap: train {f(tr & T.gap_al & ~big)} | test {f(te & T.gap_al & ~big)}')
SPp = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D = pd.read_parquet(SPp + 'd305.parquet'); M = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1']; D['STACK'] = D[M].sum(axis=1); D['ORBg'] = T.pts.where(T.gap_al).reindex(D.index).fillna(0)
print('  gap-aligned ORB (1 contract) corr with stack', round(D.ORBg.corr(D.STACK), 2), 'M1', round(D.ORBg.corr(D.M1), 2), 'V15a', round(D.ORBg.corr(D['V15a']), 2))
for lab, a, b in (('2019', '2019-01-01', '2020-01-01'), ('2024H1', '2024-01-01', '2024-07-01'), ('2026YTD', '2026-01-01', '2026-12-31'), ('all', '2018-01-01', '2027-01-01')):
    z = D[(D.index >= a) & (D.index < b)]; print(f'    {lab:8s} stack {z.STACK.sum():+7.0f} pts | gap-ORB {z.ORBg.sum():+6.0f} pts')
