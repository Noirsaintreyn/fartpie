"""Nov 8 2026: DAY-TYPE ENGINE (DTE) - our own intraday system built from what has real edge. DECLARED BEFORE RESULTS.
Idea: dealer gamma sets the day type. NEGATIVE gamma -> moves extend -> trade MOMENTUM (noise-area momentum, Rule A vector). POSITIVE gamma -> moves get absorbed -> trade
ROTATION inside the prior day's value (the Market Profile 80% rule and a VWAP 2-sigma fade). Two uncorrelated engines should make more winning days than either alone.
Data: real NQ futures 5m bars Jan 2014 - Jan 2026 (xs/dte_prep.py), RTH 09:30 - 16:00 ET; QQQ net GEX at the prior close (2018+); 1 NQ contract, 1 pt round-trip cost.
Scale: ADR = 20-day average RTH range (NQ points).
  M1 NOISE    Zarattini-Aziz-Barbon noise-area momentum on 5m bars (checks HH:00 / HH:30 from 10:00, bound / VWAP trailing exit), hard stop 0.15 ADR, flat 15:55
  M2 RULE A   the prop system's trades (QQQ-built, 2018 - Oct 2026, live_data/nq_rule_a_trades.csv), as they are
  R1 80% RULE RTH opens outside the prior-day value area; once 6 straight 5m closes (30 min) are back inside it before 13:00, trade toward the opposite edge:
              target = the opposite edge, stop = 0.08 ADR beyond the edge it came in through, else out at 15:55; one a day; no roll days.  R1p = target POC instead.
  R2 VWAP FADE a 5m close beyond VWAP +/- 2 sd between 10:30 and 14:25 -> fade at the next open, target = VWAP at the signal, stop = 1 sd further, 60-minute limit; <= 2 a day.
Report per engine: trades, win, points per trade, by year, by gamma; daily correlations; then the DTE portfolio vs every engine every day: points a day, days >= +7.5
(a Lucid 50K qualifying day at 10 micros = 1 NQ), qualifying days a month, Sharpe, max drawdown."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); H = X[X.rth].reset_index(drop=True)
o, h, l, c, v, m, dd = tuple(H[k].values for k in ('open', 'high', 'low', 'close', 'volume', 'm')) + (H.day.values,)
days = list(pd.DatetimeIndex(sorted(H.day.unique()))); gi = H.groupby('day').indices; di_ = {d: i for i, d in enumerate(days)}
DY = H.groupby('day').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), roll=('roll', 'first'), n=('open', 'size'))
DY['adr'] = (DY.h - DY.l).rolling(20).mean().shift(1); r = np.log(DY.c).diff(); DY['rv5p'] = r.rolling(5).std().rolling(252, min_periods=120).rank(pct=True).shift(1)
G = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); G.index = G.index.normalize()
DY['gex'] = G.reindex(DY.index, method='ffill').shift(1); DY.loc[DY.index < G.index.min() + pd.Timedelta(days=2), 'gex'] = np.nan
tp = (h + l + c) / 3; cvv = pd.Series(v).groupby(dd).cumsum().values; vw = pd.Series(tp * v).groupby(dd).cumsum().values / np.maximum(cvv, 1)
vsd = np.sqrt(np.maximum(pd.Series(tp * tp * v).groupby(dd).cumsum().values / np.maximum(cvv, 1) - vw * vw, 0))
VA = {}
for d in days:
    ix = gi[d]; bs = o[ix[0]] * .0005; b0 = np.floor(l[ix].min() / bs); nb = int(np.floor(h[ix].max() / bs) - b0) + 1; prof = np.zeros(nb)
    for a, b, w in zip(np.floor(l[ix] / bs) - b0, np.floor(h[ix] / bs) - b0, v[ix]): a, b = int(a), int(b); prof[a:b + 1] += w / (b - a + 1)
    p = int(prof.argmax()); u = dn = p; tot = prof[p]; tgt = .7 * prof.sum()
    while tot < tgt and (u < nb - 1 or dn > 0):
        up = prof[u + 1] if u < nb - 1 else -1; down = prof[dn - 1] if dn > 0 else -1
        if up >= down: u += 1; tot += up
        else: dn -= 1; tot += down
    VA[d] = ((b0 + dn) * bs, (b0 + p + .5) * bs, (b0 + u + 1) * bs)
def ok(d): return d in DY.index and np.isfinite(DY.adr.get(d, np.nan)) and DY.n.get(d, 0) >= 70
# ---- M1 noise momentum
pos_ = np.concatenate([np.arange(len(gi[d])) for d in days]); mv = np.abs(c / np.concatenate([np.full(len(gi[d]), o[gi[d][0]]) for d in days]) - 1)
SIG = pd.DataFrame({'d': dd, 'p': pos_, 'mv': mv}).pivot_table(index='d', columns='p', values='mv').rolling(14, min_periods=14).mean().shift(1)
def m1():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or d not in SIG.index: continue
        ix = gi[d]; o0 = o[ix[0]]; pc = c[gi[days[k - 1]][-1]]; S = .15 * DY.adr[d]; pos = 0; e = None; pend = None
        for p, i in enumerate(ix):
            if pend is not None: pos, e = pend, o[i]; pend = None
            if pos != 0 and ((l[i] <= e - S) if pos > 0 else (h[i] >= e + S)): rows.append((d, -S - 1.0)); pos = 0; continue
            if p == len(ix) - 1: break
            if (m[i] + 5) % 30 != 0 or m[i] + 5 < 600 or m[i] + 5 > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); nxt = o[ix[p + 1]]
            if pos == 1 and c[i] < max(ub, vw[i]): rows.append((d, nxt - e - 1.0)); pos = 0
            elif pos == -1 and c[i] > min(lb, vw[i]): rows.append((d, e - nxt - 1.0)); pos = 0
            if pos == 0 and c[i] > ub: pend = 1
            elif pos == 0 and c[i] < lb: pend = -1
        if pos != 0: rows.append((d, (c[ix[-1]] - e) * pos - 1.0))
    return pd.DataFrame(rows, columns=['day', 'pts'])
def walk(ix, j, d_, e, T, S, until_m=955, maxb=None):
    for q, i in enumerate(ix[j:]):
        if m[i] >= until_m or (maxb and q >= maxb): return (c[ix[j + q - 1]] - e) * d_ if q > 0 else 0.0
        if ((l[i] <= S) if d_ > 0 else (h[i] >= S)): return (S - e) * d_
        if ((h[i] >= T) if d_ > 0 else (l[i] <= T)): return (T - e) * d_
    return (c[ix[-1]] - e) * d_
def r1(to_poc=False):
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        val, poc, vah = VA[days[k - 1]]; ix = gi[d]; o0 = o[ix[0]]; side = 1 if o0 > vah else (-1 if o0 < val else 0)
        if side == 0: continue
        run = 0
        for p, i in enumerate(ix[:-1]):
            if m[i] >= 780: break
            run = run + 1 if val <= c[i] <= vah else 0
            if run >= 6:
                d_ = -side; e = o[ix[p + 1]]; T = poc if to_poc else (val if d_ < 0 else vah); S = (vah + .08 * DY.adr[d]) if d_ < 0 else (val - .08 * DY.adr[d])
                if (T - e) * d_ <= 0 or (e - S) * d_ <= 0: break
                rows.append((d, walk(ix, p + 1, d_, e, T, S) - 1.0)); break
    return pd.DataFrame(rows, columns=['day', 'pts'])
def r2():
    rows = []
    for d in days:
        if not ok(d): continue
        ix = gi[d]; busy = -1; n = 0
        for p, i in enumerate(ix[:-1]):
            if n >= 2 or m[i] < 630 or m[i] > 865 or p <= busy or vsd[i] <= 0: continue
            z = (c[i] - vw[i]) / vsd[i]
            if abs(z) < 2: continue
            d_ = -int(np.sign(z)); e = o[ix[p + 1]]; T = vw[i]; S = e - d_ * vsd[i]
            if (T - e) * d_ <= 0: continue
            rows.append((d, walk(ix, p + 1, d_, e, T, S, maxb=12) - 1.0)); busy = p + 12; n += 1
    return pd.DataFrame(rows, columns=['day', 'pts'])
RA = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); RA = pd.DataFrame({'day': pd.to_datetime(RA.day).dt.normalize(), 'pts': RA.net_pts})
if __name__ == '__main__':
    E = {'M1 noise momentum': m1(), 'M2 Rule A': RA, 'R1 80% rule (opposite edge)': r1(), 'R1p 80% rule (to POC)': r1(True), 'R2 VWAP 2-sd fade': r2()}
    AD = pd.DatetimeIndex([d for d in days if ok(d)]); AD18 = AD[AD >= '2018-01-03']; gneg = (DY.gex < 0)
    print(f'days {len(AD)} ({AD[0]:%Y-%m} - {AD[-1]:%Y-%m}); with GEX {len(AD18)}; median ADR 2014-17 {DY.adr[:"2017"].median():.0f}, 2022-25 {DY.adr["2022":].median():.0f}', flush=True)
    P = {}
    for nm, T in E.items():
        T = T[T.day.isin(AD)]; ty = T.day.dt.year; P[nm] = T.groupby('day').pts.sum().reindex(AD).fillna(0.0); gn = T.day.map(gneg)
        print(f'\n  {nm}: {len(T)} trades ({len(T) / len(AD):.2f}/day) win {np.mean(T.pts > 0):.0%} avg {T.pts.mean():+.2f} | neg gamma {T.pts[gn == True].mean():+.2f} (n{(gn == True).sum()}) '
              f'pos gamma {T.pts[gn == False].mean():+.2f} (n{(gn == False).sum()}) | by year ' + ' '.join(f'{y % 100:02d} {T.pts[ty == y].mean():+.1f}' for y in sorted(ty.unique())), flush=True)
    A = pd.DataFrame(P).loc[AD18]; print('\n  daily correlation 2018+:\n' + A.corr().round(2).to_string(), flush=True)
    gn18 = gneg.reindex(AD18).values
    BOOKS = {'M1 noise only': A['M1 noise momentum'], 'M1 + M2': A['M1 noise momentum'] + A['M2 Rule A'],
             'everything every day': A['M1 noise momentum'] + A['M2 Rule A'] + A['R1 80% rule (opposite edge)'] + A['R2 VWAP 2-sd fade'],
             'DTE: neg gamma M1+M2, pos gamma R1+R2+M2': np.where(gn18, A['M1 noise momentum'], A['R1 80% rule (opposite edge)'] + A['R2 VWAP 2-sd fade']) + A['M2 Rule A'],
             'DTE-p: same with R1 to POC': np.where(gn18, A['M1 noise momentum'], A['R1p 80% rule (to POC)'] + A['R2 VWAP 2-sd fade']) + A['M2 Rule A']}
    print('\n  BOOK 2018 - Jan 2026, 1 NQ per engine     pts/day  days>=+7.5  qual days/mo  losing days  Sharpe  max DD  worst month | by year pts/day', flush=True)
    for nm, P_ in BOOKS.items():
        P_ = pd.Series(np.asarray(P_), index=AD18); cum = P_.cumsum(); mo = P_.groupby(P_.index.to_period('M')).sum()
        print(f'  {nm:<42} {P_.mean():+7.2f} {np.mean(P_ >= 7.5):10.0%} {np.mean(P_ >= 7.5) * 21:12.1f} {np.mean(P_ < 0):12.0%} {P_.mean() / P_.std() * np.sqrt(252):7.2f} {(cum - cum.cummax()).min():+7.0f} {mo.min():+11.0f} | '
              + ' '.join(f'{y % 100:02d} {x:+.1f}' for y, x in P_.groupby(P_.index.year).mean().items()), flush=True)
    for nm, T in E.items(): T.to_parquet(f'/tmp/claude-0/ml/dte_{nm.split()[0]}.parquet')
