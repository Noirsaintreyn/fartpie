"""Nov 24 2026 report 454. Rey: gamma isn't a daily sign, it's per strike; look at flips in negative gamma by STRIKE structure, flips that recently left positive gamma, flips with the most room below the zero-gamma level, and volatility. 15m AND 1h flips. DECLARED BEFORE RESULTS. CORRECTION (same day): the first run compared LSE split-adjusted prices with unadjusted theta zero-gamma levels (NVDA x40, TSLA x15, AAPL x4, AMZN x20 pre-split), which made BELOW nearly always true on those names; fixed by dividing the theta spot/ZG by the split factor (spot / LSE daily close, rounded). Rerun results supersede the first run.
DATA gamma rebuilt from raw ThetaData (xs/build_gex_theta2.py): per day zero-gamma level ZG (price where net dollar gamma crosses 0 when spot is shifted, OI and IV held), net gamma, ATM IV. Features for session s use the PRIOR session row (known at its close).
UNIVERSE QQQ SPY IWM DIA (ungated, no ratio exists) + TSLA NVDA AAPL AMD AMZN MSFT META COST with the G20 options-ratio gate. Long only, bull vector flips (xs/vector_pine) on 15m bars and on 1h bars (RTH 15m aggregated by hour). Underlying returns, entry = open of bar after the flip bar's next bar (as reports 451-453), same-session exit: first bear flip (next open) else 15:45 (15m) / 15:30 (1h) bar open; 15m entries 09:45-14:30, 1h entries 10:30-14:30; cost 4bp round trip. 2018-07..2026-09.
FEATURES (all from the prior session): BELOW = entry open < ZG_prev (if no ZG crossing within +-12%: BELOW = net gamma < 0). RECENT = BELOW now and prior close was in positive gamma within the 4 sessions before... precisely: previous session's regime negative and at least one of the 3 sessions before it positive; ESTABLISHED = previous 4 sessions all negative. ROOM = (ZG_prev - entry open)/entry open in units of the name's ATM IV / sqrt(252) (daily sigma). VOL = prior-session ATM IV above its own trailing-252-session median (min 60).
HYPOTHESES (each |t|>=2 month-clustered, same sign train 2019-21 and test 2018+2022+): HA BELOW beats ABOVE. HB within BELOW: RECENT beats ESTABLISHED. HC within BELOW: top-tercile ROOM beats bottom tercile. HD within BELOW: high VOL beats low VOL. Run per timeframe (8 tests: treat t~2.5 as the bar). Baseline for comparison: the daily net-gamma-sign split NEG vs POS on the same trades. Eras 18-19 | 20-21 | 22 | 23-26 for the passing ones."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
src = open('xs/report_440_neggamma_short_and_2025.py').read(); exec(src[:src.index("names = sorted")].split("SP = ")[0] + "SP = '" + SP + "'; UP = '" + UP + "'; ET = 'America/New_York'\n" + "\n".join(l for l in src[:src.index("names = sorted")].split("\n") if l.startswith("R = ") or l.startswith("G20 =") or l.startswith("Rr")))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
files = {os.path.basename(f).split('_')[0]: f for f in glob.glob(UP + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')}
IDX = ['QQQ', 'SPY', 'IWM', 'DIA']; STK = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'COST']; rows = []
def hourly(H):
    H = H.copy(); H['d'] = H.dt.dt.normalize(); H['b'] = (H.dt.dt.hour * 60 + H.dt.dt.minute - 570) // 60
    return H.groupby(['d', 'b'], sort=True).agg(dt=('dt', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
for tk in IDX + STK:
    if tk not in files: print('no bars', tk); continue
    X = pd.read_csv(UP + f'data/theta_gex2/{tk}.csv', parse_dates=['date']).set_index('date').sort_index()
    H0 = rth(pd.read_parquet(files[tk]).rename(columns={'datetime': 'ts'}), 'ts'); dcl = H0.groupby(H0.dt.dt.normalize()).close.last()
    fac = (X.spot / dcl.reindex(X.index)).rolling(5, center=True, min_periods=1).median(); fr = fac.round(); fac = pd.Series(np.where((fr >= 2) & ((fac - fr).abs() <= .15 * fr), fr, 1.0), index=X.index)   # theta spot/zero-gamma are unadjusted, LSE bars are split-adjusted: divide by the split factor
    X['spot'] = X.spot / fac; X['zg'] = X.zg / fac; print(tk, 'split factors seen', sorted(set(fac.round(0).astype(int))), flush=True)
    X['neg'] = np.where(X.zg.notna(), X.spot < X.zg, X.net_gex < 0).astype(float); X['ivmed'] = X.atm_iv.rolling(252, min_periods=60).median().shift(1); X['volhi'] = (X.atm_iv > X.ivmed).astype(float).where(X.ivmed.notna())
    X['neg_prev'] = X.neg; X['pos3'] = (1 - X.neg).shift(1).rolling(3, min_periods=3).max(); X['neg4'] = X.neg.rolling(4, min_periods=4).min()
    P = X[['zg', 'net_gex', 'atm_iv', 'neg', 'pos3', 'neg4', 'volhi']].shift(1)
    for TF in ('15m', '1h'):
        H = H0 if TF == '15m' else hourly(H0); n = len(H); o_ = H.open.values; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); mm = (H.dt.dt.hour * 60 + H.dt.dt.minute).values
        regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']]); gate = np.ones(n, bool) if tk in IDX else (G20[tk].reindex(days, method='ffill').values.astype(float) == 1)
        Pm = P.reindex(days, method='ffill', tolerance=pd.Timedelta('5D')); lo, hi, cut = (585, 870, 945) if TF == '15m' else (630, 870, 930)
        zg, ng, iv, neg, pos3, neg4, vh = (Pm[c].values for c in ('zg', 'net_gex', 'atm_iv', 'neg', 'pos3', 'neg4', 'volhi'))
        for i, dd in fls:
            if dd != 1: continue
            fb = i + 1; e = fb + 1
            if e + 1 >= n or not gate[fb - 1] or not (lo <= mm[e] <= hi) or day[e] != day[fb - 1]: continue
            t0 = H.dt[e]
            if t0 < pd.Timestamp('2018-07-01') or t0 > pd.Timestamp('2026-09-30') or np.isnan(neg[e]): continue
            x = None
            for j in range(e + 1, n):
                if day[j] != day[e]: x = j - 1; break
                if regs[j - 1] == -1 or mm[j] >= cut: x = j; break
            if x is None: continue
            below = (o_[e] < zg[e]) if np.isfinite(zg[e]) else bool(neg[e] == 1); room = ((zg[e] - o_[e]) / o_[e]) / (iv[e] / np.sqrt(252)) if np.isfinite(zg[e]) and np.isfinite(iv[e]) and iv[e] > 0 else np.nan
            rows.append(dict(tk=tk, tf=TF, entry=t0, below=bool(below), recent=bool(below and neg[e] == 1 and pos3[e] == 1), est=bool(below and neg4[e] == 1), room=room, volhi=vh[e], daily_neg=bool(ng[e] < 0), r=o_[x] / o_[e] - 1 - 4e-4))
    print(tk, 'trades so far', len(rows), flush=True)
T = pd.DataFrame(rows); T['mon'] = T.entry.dt.to_period('M'); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26'])
def ctest(A, B):
    if len(A) < 8 or len(B) < 8: return np.nan, np.nan
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.r[x.s == 1].sum(), s0=x.r[x.s == -1].sum()))); m1, m0 = A.r.mean(), B.r.mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 1e4, (m1 - m0) / se
def show(nm, A, B):
    d, t_ = ctest(A, B); tr = ctest(A[A.half == 'train'], B[B.half == 'train']); te = ctest(A[A.half == 'test'], B[B.half == 'test'])
    ok = 'PASS' if (abs(t_) >= 2 and np.sign(tr[0]) == np.sign(te[0]) == np.sign(d)) else '    '
    print(f"{ok} {nm:34s} A {A.r.mean()*1e4:6.1f}bp (n{len(A):5d}) B {B.r.mean()*1e4:6.1f}bp (n{len(B):5d}) diff {d:+6.1f} t {t_:+5.2f} | train {tr[0]:+6.1f} (t {tr[1]:+.1f}) test {te[0]:+6.1f} (t {te[1]:+.1f})")
    return d, t_
for TF in ('15m', '1h'):
    S = T[T.tf == TF]; print(f'\n===== {TF} flips: {len(S)} trades, mean {S.r.mean()*1e4:.1f}bp, BELOW share {S.below.mean():.2f}, daily-NEG share {S.daily_neg.mean():.2f}')
    show('BASE daily net gamma <0 vs >=0', S[S.daily_neg], S[~S.daily_neg])
    show('HA below ZG vs above', S[S.below], S[~S.below]); Bl = S[S.below]
    show('HB recent cross vs established', Bl[Bl.recent], Bl[Bl.est])
    q = Bl.room.quantile([1 / 3, 2 / 3]).values; show('HC room top vs bottom tercile', Bl[Bl.room >= q[1]], Bl[Bl.room <= q[0]])
    show('HD high vs low ATM IV (in below)', Bl[Bl.volhi == 1], Bl[Bl.volhi == 0])
    print('   by era (below vs above, bp / n): ' + ' | '.join(f"{e} {S[S.below&(S.era==e)].r.mean()*1e4:+.0f}({(S.below&(S.era==e)).sum()}) / {S[~S.below&(S.era==e)].r.mean()*1e4:+.0f}({(~S.below&(S.era==e)).sum()})" for e in ('18-19', '20-21', '22', '23-26')))
    print('   by name (below vs above bp): ' + ' | '.join(f"{t} {S[S.below&(S.tk==t)].r.mean()*1e4:+.0f}/{S[~S.below&(S.tk==t)].r.mean()*1e4:+.0f}" for t in IDX + STK))
