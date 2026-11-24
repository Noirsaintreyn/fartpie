"""Nov 24 2026 report 453 (report 452 rerun with gamma REBUILT from raw ThetaData, xs/build_gex_theta.py, for TSLA NVDA AAPL AMD AMZN MSFT META GOOGL(2022-03 only); gex older than 5 days = missing, not carried). Original 452 text follows. DECLARED BEFORE RESULTS: same H2 pass rule, now on 7-8 names. Rey: use the leveraged single-stock ETFs of the names that make the options-ratio (G20) cut; long-only; neg-gamma where real gex exists. DECLARED BEFORE RESULTS.
PAIRS underlying -> real 15m ETF bars (LSE vault, xs/pull_lse_levetf.py): TSLA-TSLL NVDA-NVDL AAPL-AAPU AMZN-AMZU GOOGL-GGLL MSFT-MSFU AMD-AMDL META-METU COIN-CONL MSTR-MSTU PLTR-PLTU AVGO-AVL NFLX-NFXL MU-MUU HOOD-HOOG GLD-UGL SLV-AGQ TSM-TSMU (all 2x).
SIGNAL underlying 15m vector bull flip (xs/vector_pine), entry next bar open if 09:45-14:30 ET, exit open after bear flip else 15:45 open (same-session, as report 451). Trade skipped if the ETF has no bar at entry or exit (counted). Cost 6bp round trip ETF, 4bp underlying. Only trades inside the ETF's own history.
H1 (ratio gate carries to the ETF): mean ETF trade return with G20==1 minus G20==0, month-clustered t >= 2, pooled over pairs, same sign in train (2019-21 is mostly before these ETFs exist -> train n reported, not required) and in 2023-24 / 2025-26.
H2 (neg gamma on the ETF): within G20==1, NEG (real gex prior-close net_gex<0) minus POS on ETF return, pairs with gex files only (TSLA NVDA AAPL AMD), month-clustered t >= 2, same sign in 2023-24 and 2025-26 and in 3 of 4 names.
H3 (does the ETF add anything): ETF mean / underlying mean ratio vs 2x (trade-level), and book Calmar of ETF vs underlying on the same gated trades.
BOOK: each concurrent trade gets 20% of equity (cap 5), realized by exit day, daily compounding; ETF vs underlying on identical trades. Eras 22 | 23-24 | 25-26 (ETFs launched 2022-24, so no 2018-21)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
src = open('xs/report_440_neggamma_short_and_2025.py').read(); exec(src[:src.index("names = sorted")].split("SP = ")[0] + "SP = '" + SP + "'; UP = '" + UP + "'; ET = 'America/New_York'\n" + "\n".join(l for l in src[:src.index("names = sorted")].split("\n") if l.startswith("R = ") or l.startswith("G20 =") or l.startswith("Rr")))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
PAIRS = dict(TSLA='TSLL', NVDA='NVDL', AAPL='AAPU', AMZN='AMZU', GOOGL='GGLL', MSFT='MSFU', AMD='AMDL', META='METU', COIN='CONL', MSTR='MSTU', PLTR='PLTU', AVGO='AVL', NFLX='NFXL', MU='MUU', HOOD='HOOG', GLD='UGL', SLV='AGQ', TSM='TSMU')
GEXN = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL']; files = {os.path.basename(f).split('_')[0]: f for f in glob.glob(UP + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')}
rows = []; skipped = {}
for u, etf in PAIRS.items():
    if u not in files or not os.path.exists(SP + f'{etf}_15m_lse.parquet'): print('missing', u, etf); continue
    H = rth(pd.read_parquet(files[u]).rename(columns={'datetime': 'ts'}), 'ts'); n = len(H); o_ = H.open.values; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); mm = (H.dt.dt.hour * 60 + H.dt.dt.minute).values
    E = rth(pd.read_parquet(SP + f'{etf}_15m_lse.parquet'), 'ts').set_index('dt').open; oE = E.reindex(H.dt).values
    regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']]); gate = (G20[u].reindex(days, method='ffill').values.astype(float) == 1) if u in G20.columns else np.zeros(n, bool)
    gp = None
    if u in GEXN: g = pd.read_csv(UP + f'data/theta_gex/{u}.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; gp = g.shift(1).reindex(days, method='ffill', tolerance=pd.Timedelta('5D')).values
    first_etf = E.index.min(); sk = 0
    for i, dd in fls:
        if dd != 1: continue
        fb = i + 1; e = fb + 1
        if e + 1 >= n or not (585 <= mm[e] <= 870) or day[e] != day[fb - 1]: continue
        t0 = H.dt[e]
        if t0 < first_etf + pd.Timedelta(days=3) or t0 > pd.Timestamp('2026-10-31'): continue
        x = None
        for j in range(e + 1, n):
            if day[j] != day[e]: x = j - 1; break
            if regs[j - 1] == -1 or mm[j] >= 945: x = j; break
        if x is None: continue
        if np.isnan(oE[e]) or np.isnan(oE[x]): sk += 1; continue
        rows.append(dict(u=u, etf=etf, entry=t0, exit=H.dt[x], day=pd.Timestamp(day[x]), gate=bool(gate[fb - 1]), neg=(bool(gp[fb - 1] < 0) if gp is not None and np.isfinite(gp[fb - 1]) else None), re=oE[x] / oE[e] - 1 - 6e-4, ru=o_[x] / o_[e] - 1 - 4e-4))
    skipped[u] = sk; print(u, etf, 'first ETF bar', str(first_etf)[:10], 'skipped (ETF bar missing)', sk, flush=True)
T = pd.DataFrame(rows); T['mon'] = T.entry.dt.to_period('M'); T['era'] = pd.cut(T.entry.dt.year + (T.entry.dt.month > 0) * 0, [2021, 2022, 2024, 2100], labels=['22', '23-24', '25-26'])
def ctest(A, B, c):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x[c][x.s == 1].sum(), s0=x[c][x.s == -1].sum()))); m1, m0 = A[c].mean(), B[c].mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 1e4, (m1 - m0) / se
print(f'\ntrades {len(T)} gated {T.gate.sum()} ungated {(~T.gate).sum()}'); G1, G0 = T[T.gate], T[~T.gate]
print('\nPER PAIR (gated flips): n | ETF bp win% | underlying bp | ETF/und | ungated ETF bp (n)')
for u in PAIRS:
    a = G1[G1.u == u]; b = G0[G0.u == u]
    if len(a) == 0 and len(b) == 0: continue
    print(f"{u:6s}{PAIRS[u]:5s} n {len(a):3d} | ETF {a.re.mean()*1e4:6.1f} win {(a.re>0).mean()*100:3.0f}% | und {a.ru.mean()*1e4:6.1f} | x{(a.re.mean()/a.ru.mean() if len(a) and a.ru.mean() else np.nan):5.1f} | ungated ETF {b.re.mean()*1e4:6.1f} (n {len(b)})")
print('\nH1 gate effect on ETF return (gated - ungated), month-clustered')
d, t_ = ctest(G1, G0, 're'); print(f"pooled {d:+.1f}bp t {t_:+.2f} | gated mean {G1.re.mean()*1e4:.1f} ungated {G0.re.mean()*1e4:.1f}")
for e in ('22', '23-24', '25-26'):
    a, b = G1[G1.era == e], G0[G0.era == e]
    if len(a) > 5 and len(b) > 5: d, t_ = ctest(a, b, 're'); print(f"  era {e:6s} gated {a.re.mean()*1e4:6.1f} (n{len(a)}) ungated {b.re.mean()*1e4:6.1f} (n{len(b)}) diff {d:+.1f} t {t_:+.2f}")
print('\nH2 neg gamma within gated flips (theta-gex names, ETF return)')
X = G1[G1.neg.notna()].copy(); X['neg'] = X.neg.astype(bool); A, B = X[X.neg], X[~X.neg]; print(f'n NEG {len(A)} POS {len(B)}')
if len(A) > 5 and len(B) > 5:
    d, t_ = ctest(A, B, 're'); print(f"pooled NEG {A.re.mean()*1e4:.1f}bp POS {B.re.mean()*1e4:.1f}bp diff {d:+.1f} t {t_:+.2f} | underlying diff {ctest(A, B, 'ru')[0]:+.1f}bp t {ctest(A, B, 'ru')[1]:+.2f}")
    for e in ('22', '23-24', '25-26'):
        a, b = A[A.era == e], B[B.era == e]; print(f"  era {e:6s} NEG {a.re.mean()*1e4:6.1f} (n{len(a)}) POS {b.re.mean()*1e4:6.1f} (n{len(b)})")
    for u in GEXN:
        a, b = A[A.u == u], B[B.u == u]; print(f"  {u:5s} NEG {a.re.mean()*1e4:6.1f} (n{len(a)}) POS {b.re.mean()*1e4:6.1f} (n{len(b)})")
def book(S, c, nm, cap=5, slot=.2):
    S = S.sort_values('entry'); open_ = []; keep = []
    for r in S.itertuples():
        open_ = [x for x in open_ if x > r.entry]
        if len(open_) < cap: keep.append(r); open_.append(r.exit)
    K = pd.DataFrame(keep); idx = pd.bdate_range('2022-03-01', '2026-10-31'); r = (K.groupby('day')[c].sum() * slot).reindex(idx).fillna(0); e = (1 + r).cumprod(); cg = e.iloc[-1] ** (252 / len(r)) - 1; dd = (e / e.cummax() - 1).min()
    print(f"{nm:34s} trades {len(K):4d} CAGR {cg*100:6.1f}% DD {dd*100:6.1f}% Calmar {cg/abs(dd):5.2f} Sh {r.mean()/r.std()*np.sqrt(252):5.2f}"); return cg, dd
print('\nBOOKS (20% slots, cap 5, from 2022-03)'); book(G1, 're', 'gated ETF'); book(G1, 'ru', 'gated underlying (same trades)'); book(G0, 're', 'ungated ETF')
if len(X): book(X[X.neg], 're', 'gated NEG-only ETF (4 gex names)'); book(X, 're', 'gated all (4 gex names) ETF'); book(X[X.neg], 'ru', 'gated NEG-only underlying'); book(X, 'ru', 'gated all (4 gex) underlying')
