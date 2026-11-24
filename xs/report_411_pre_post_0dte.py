"""Nov 23 2026 report 411. Rey: compare option-flow results PRE vs POST 0DTE (hedging pressure differs); add what is positive post-0DTE to the full market analysis.
CUTOFF (declared): 2022-05-11, the day SPX Tuesday and Thursday expiries were added, completing five-day 0DTE.  PRE = flips 2019-01..2022-05-10; POST-A = 2022-05-11..2024-06-30 (discovery); POST-B = 2024-07-01..2025-12-15 (validation, QQQ greeks end 2025-12-15).
FEATURES at the prior close, each as its PIT rolling percentile (prior 252 obs, min 126): STOCK level (names with option files): P1 put share of 5d volume, P2 opening-position imbalance (call opens - put opens)/(sum), P3 CALL share of open interest (= 1 - put share, same information as put share with the sign flipped), P4 own net GEX, P5 IV - 20d RV.  MARKET level (QQQ): MG net GEX = call_gex - put_gex, MV net vanna, MC net charm, MS 25-delta skew, MP put/call volume ratio.
OUTCOME: net flip return (live book's 12,320 gated 15m flips, -4bp): top tercile minus bottom tercile mean, month-clustered t.  A feature is ADDED to the market analysis only if discovered in POST-A (|t| >= 1.5) AND validated in POST-B (same sign, |t| >= 2).  Also monthly 21d Spearman ICs of the stock-level selectors (report 407) by the same three windows."""
import numpy as np
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
X = UP + 'xs/ext/'; import os
G = {f[:-4]: pd.read_csv(X + 'gex/' + f, parse_dates=['date']).set_index('date').sort_index() for f in os.listdir(X + 'gex')}
O = {f[:-4]: pd.read_csv(X + 'oi_daily/' + f, parse_dates=['date']).set_index('date').sort_index() for f in os.listdir(X + 'oi_daily')}
Q = pd.read_csv(UP + 'options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); Q2 = pd.read_csv(UP + 'options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
def prank(s):  # PIT percentile of the latest value within the prior 252 observations
    return s.rolling(253, min_periods=127).apply(lambda w: (w[:-1] < w[-1]).mean() + .5 * (w[:-1] == w[-1]).mean(), raw=True)
MK = pd.DataFrame({'MG': Q.call_gex - Q.put_gex, 'MV': Q2.net_vanna, 'MC': Q2.net_charm, 'MS': Q.skew_25d, 'MP': Q.put_volume / Q.call_volume}).apply(prank)
SK = {}
for tk in B0.tk.unique():
    d = pd.DataFrame(index=pd.DatetimeIndex([]))
    o = O.get(tk); g = G.get(tk); px = pd.Series(Cv[:, ci[tk]], index=idx) if tk in ci else None
    if o is not None:
        w5 = lambda col: o[col].rolling(5).sum(); d = pd.DataFrame({'P1': w5('put_vol') / (w5('put_vol') + w5('call_vol')), 'P2': (w5('c_dpos') - w5('p_dpos')) / (w5('c_dpos') + w5('p_dpos')).clip(lower=1), 'P3': o.call_oi / (o.call_oi + o.put_oi)})
    if g is not None and px is not None:
        rv = np.log(px).diff().rolling(20).std() * np.sqrt(252); d = d.join(pd.DataFrame({'P4': g.net_gex, 'P5': g.atm_iv - rv.reindex(g.index)}), how='outer')
    if len(d): SK[tk] = d.apply(prank)
f = B0[['tk', 'entry', 'ret']].copy(); f['d0'] = (pd.to_datetime(f.entry).dt.normalize() - pd.Timedelta(days=1)).astype('datetime64[ns]'); f['n'] = f.ret - 4e-4; f['mo'] = pd.to_datetime(f.entry).dt.to_period('M')
for c in ('MG', 'MV', 'MC', 'MS', 'MP'): f[c] = MK[c].reindex(f.d0, method='ffill', tolerance=pd.Timedelta(days=5)).values
for c in ('P1', 'P2', 'P3', 'P4', 'P5'): f[c] = [SK[t][c].reindex([d], method='ffill', tolerance=pd.Timedelta(days=5)).iloc[0] if (t in SK and c in SK[t]) else np.nan for t, d in zip(f.tk, f.d0)]
e = pd.to_datetime(f.entry); f['win'] = np.where(e < '2019-01-01', 'x', np.where(e < '2022-05-11', 'PRE', np.where(e < '2024-07-01', 'POST-A', np.where(e <= '2025-12-15', 'POST-B', 'x'))))
LAB = {'MG': 'MKT net GEX (QQQ)', 'MV': 'MKT net vanna', 'MC': 'MKT net charm', 'MS': 'MKT 25d skew', 'MP': 'MKT put/call volume', 'P1': 'put share of volume', 'P2': 'opening-pos imbalance', 'P3': 'CALL share of OI', 'P4': 'own net GEX', 'P5': 'IV - 20d RV'}
def tb(d, c):
    d = d[d[c].notna()]; hi = d[d[c] >= 2 / 3]; lo = d[d[c] <= 1 / 3]
    if len(hi) < 25 or len(lo) < 25: return np.nan, np.nan, len(hi) + len(lo)
    m = hi.groupby('mo').n.mean().sub(lo.groupby('mo').n.mean(), fill_value=np.nan).dropna(); return (hi.n.mean() - lo.n.mean()) * 100, (m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 4 else np.nan), len(hi) + len(lo)
print(f"{'feature (high tercile minus low, net flip %)':44s}| PRE pp (t) n | POST-A pp (t) n | POST-B pp (t) n | verdict"); ADD = []
for c, l in LAB.items():
    r = {w: tb(f[f.win == w], c) for w in ('PRE', 'POST-A', 'POST-B')}; a, b = r['POST-A'], r['POST-B']
    ok = (abs(a[1]) >= 1.5) and (np.sign(a[0]) == np.sign(b[0])) and (abs(b[1]) >= 2)
    if ok: ADD.append((c, l, 'high' if a[0] > 0 else 'low'))
    print(f"{l:44s}| " + ' | '.join(f"{r[w][0]:+5.2f} ({r[w][1]:+.1f}) {r[w][2]:5d}" for w in ('PRE', 'POST-A', 'POST-B')) + f" | {'ADD ' + ('high' if a[0] > 0 else 'low') + ' better' if ok else '-'}")
print('\nADDED to the market analysis (discovered POST-A, validated POST-B):', ADD if ADD else 'none')
# monthly 21d IC of the stock-level selectors by window (report 407 data)
F7 = pd.read_parquet(SP + 'opt407.parquet'); F7['t'] = pd.to_datetime(F7.t); from scipy.stats import spearmanr
W7 = lambda t: 'PRE' if t < pd.Timestamp('2022-05-11') else ('POST-A' if t < pd.Timestamp('2024-07-01') else 'POST-B')
print('\nMONTHLY 21d IC of stock-level selectors (raw sign), months: PRE | POST-A | POST-B'); 
for c, l in (('P1', 'put share of volume'), ('P2', 'opening-pos imbalance'), ('P3', 'put share of OI'), ('P4', 'own net GEX z'), ('P5', 'IV - 20d RV'), ('ratio', 'ratio'), ('Fp', 'fundamentals')):
    out = {}
    for w in ('PRE', 'POST-A', 'POST-B'):
        ics = [spearmanr(g[c], g.y)[0] for t, g in F7[F7.t.map(W7) == w].groupby('t') if g[c].notna().sum() >= 8 and g.y.notna().sum() >= 8 and g[c].nunique() > 2 for g in [g[g[c].notna() & g.y.notna()]]]
        ics = pd.Series(ics).dropna(); out[w] = (ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 3 else np.nan, len(ics))
    print(f"  {l:24s}" + ' | '.join(f"{out[w][0]:+.3f} (t {out[w][1]:+.1f}) n{out[w][2]}" for w in out))
