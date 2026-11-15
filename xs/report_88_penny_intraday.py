"""Nov 14 2026 PENNY / LOW-PRICED INTRADAY (Rey: lots of money in pennies, try 1m/5m/other frames, enter on the day of the flip). DATA LIMIT: 60 days of 5m bars (Aug 20-Nov 12 2026), ~2,000 low-priced names; ONE regime, no era split, survivorship-flattered (names still listed) -> every result is a LEAD, not a validated finding. 1m is not available at depth (Yahoo gives 7 days). Declared before results. Universe: close 0.50-10, median daily RTH dollar volume >= $300k. RTH bars only (09:30-16:00). Costs: 100bp round trip primary (penny spreads/slippage), 50bp and 200bp sensitivity. PART 1 timeframe grid: vector run on 5m/15m/30m/1h bars (RTH 5m resampled), long flips only; entry next bar open; exits (a) fixed horizon (5m:12 bars, 15m:8, 30m:8, 1h:7 = about 1h/2h/4h/1 day) and (b) open after opposite flip. Control = unconditional open-to-open over the same horizon on every bar of the same names. Report n, gross mean/median, net at 100bp, control, excess, day-clustered t. Lead bar: excess net of 100bp > 0 with day-clustered t >= 2. PART 2 day-of-flip entry on the DAILY vector (2-year daily bars): trigger T = level(prev bar) + 0.35*ATR(prev bar) (approx; validated by how often daily bull flips have close > T). A) system: enter next-day open after the confirmed close flip; B) intraday: enter at max(T, open of first RTH bar whose high >= T) on the flip day; both exit at the open after the daily opposite flip (still-open trades marked at last close), 100bp both. Also FALSE TRIGGERS: days a bear/neutral name crosses T intraday but closes below it; return of entering at the cross and exiting that day's close."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb')); DD = pickle.load(open(SC + 'lowpx_daily.pkl', 'rb'))
def clean(x):
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York') if x.index.tz is not None else x.index; x = x[~x.index.duplicated()]
    t = x.index.time; return x[(x.index.hour * 60 + x.index.minute >= 570) & (x.index.hour * 60 + x.index.minute < 960)]
R = {}
for tk, x in M5.items():
    x = clean(x); x = x[x.volume > 0] if len(x) else x
    if len(x) < 400: continue
    dvd = (x.close * x.volume).groupby(x.index.date).sum().median()
    if dvd >= 3e5 and 0.5 <= x.close.median() < 10: R[tk] = x
print('names', len(R))
TF = {'5m': (None, 12), '15m': ('15min', 8), '30m': ('30min', 8), '1h': ('60min', 7)}; H = 100e-4 * 1.0
rows = []
for tk, x in R.items():
    for tf, (rs, hz) in TF.items():
        B = x if rs is None else x.resample(rs, offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open'])
        if len(B) < 150: continue
        B = B.copy(); reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); o = B.open.values; n = len(B); d = np.array(B.index.date)
        fh = np.r_[o[1 + hz:] / o[1:n - hz] - 1, np.full(hz + 1, np.nan)][:n]; fh = np.r_[o[hz + 1:] / o[1:n - hz] - 1, [np.nan] * (hz + 1)] if False else fh
        ctrl = pd.Series(np.r_[o[hz + 1:n] / o[1:n - hz] - 1, np.full(hz, np.nan)], index=range(n - 1 if False else n)) if False else None
        for i in range(1, n - hz - 2):
            fwd = o[i + 1 + hz] / o[i + 1] - 1 if i + 1 + hz < n else np.nan
            rows.append((tk, tf, d[i], 1 if (reg[i] == 1 and reg[i - 1] != 1) else 0, fwd))
        # hold-to-bear
A = pd.DataFrame(rows, columns=['tk', 'tf', 'd', 'flip', 'fwd']).dropna(); A['day'] = A.d.astype(str)
print('\nPART 1: vector long flips by timeframe (fixed horizon), 60 days, ONE regime')
for tf in TF:
    z = A[A.tf == tf]; f = z[z.flip == 1]; ctl = z.fwd.mean(); e = f.fwd - ctl
    c = f.assign(e=e).groupby('day').e.mean(); t = c.mean() / (c.std() / np.sqrt(len(c))) if len(c) > 5 else np.nan
    print(f' {tf:4s} n {len(f):6d} gross mean {f.fwd.mean()*100:+6.2f}% med {f.fwd.median()*100:+5.2f}% | control {ctl*100:+6.2f}% | excess gross {(f.fwd.mean()-ctl)*100:+6.2f}pp | net@50bp {(f.fwd.mean()-ctl-.005)*100:+6.2f} @100bp {(f.fwd.mean()-ctl-.01)*100:+6.2f} @200bp {(f.fwd.mean()-ctl-.02)*100:+6.2f} (excess vs control, costs on flip trades) | day-clustered t (gross excess) {t:+.1f} | winners>=+10% {(f.fwd>=.1).mean()*100:.1f}%')
# Part 2
print('\nPART 2: day-of-flip intraday entry vs confirmed-close entry (daily vector, 5m path for the intraday fill)')
res = []; fal = []; ok = tot = 0
for tk, x in R.items():
    q = DD.get(tk)
    if q is None or len(q) < 120: continue
    q = q[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); q.columns = ['open', 'high', 'low', 'close', 'volume']; q = q[~q.index.duplicated()]; q.index = pd.to_datetime(q.index).tz_localize(None).normalize()
    reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); o = q.open.values; c = q.close.values; n = len(q); xs = x.copy(); xs['date'] = pd.to_datetime(xs.index.date); g = dict(list(xs.groupby('date')))
    for i in range(2, n - 1):
        dts = q.index[i]
        if dts not in g or not (0.5 <= c[i] < 10): continue
        if reg[i - 1] == 1 or np.isnan(lev[i - 1]) or np.isnan(atr[i - 1]): continue
        T = lev[i - 1] + 0.35 * atr[i - 1]; b = g[dts]; hit = b[b.high >= T]
        flip = reg[i] == 1
        if flip: tot += 1; ok += c[i] > T
        if hit.empty: continue
        fill = max(T, hit.open.iloc[0])
        if flip:
            j = i + 1
            while j < n and reg[j] != -1: j += 1
            ex = o[j + 1] if j + 1 < n else c[-1]; marked = j + 1 >= n
            res.append((tk, dts, o[i + 1] * 1.0, fill, ex, marked, c[i]))
        else: fal.append((tk, dts, fill, c[i]))
print(f' trigger check: daily bull flips with close > T(prev level + 0.35*prev ATR): {ok}/{tot} = {ok/max(tot,1)*100:.0f}%')
P = pd.DataFrame(res, columns=['tk', 'd', 'A_in', 'B_in', 'ex', 'mk', 'cl']); Fa = pd.DataFrame(fal, columns=['tk', 'd', 'fill', 'cl'])
if len(P):
    P['A'] = P.ex / P.A_in - 1; P['B'] = P.ex / P.B_in - 1; P['gap'] = P.A_in / P.B_in - 1
    print(f' flips with both paths n {len(P)} (still-open marked at last close: {int(P.mk.sum())})')
    for lab, z in (('ALL', P), ('closed trades only', P[~P.mk])):
        print(f'  {lab:18s} A confirmed-close entry: mean gross {z.A.mean()*100:+6.1f}% med {z.A.median()*100:+5.1f}% net100 {(z.A.mean()-.01)*100:+6.1f}% | B intraday trigger entry: mean gross {z.B.mean()*100:+6.1f}% med {z.B.median()*100:+5.1f}% net100 {(z.B.mean()-.01)*100:+6.1f}% | mean entry advantage of B over A {(-z.gap).mean()*100:+.1f}% (paired B-A {(z.B-z.A).mean()*100:+.1f}pp, median {(z.B-z.A).median()*100:+.1f})')
if len(Fa):
    Fa['r'] = Fa.cl / Fa.fill - 1; print(f' FALSE triggers (cross T intraday, close back below, no flip): n {len(Fa)}  enter at cross, exit same-day close: mean {Fa.r.mean()*100:+.2f}% med {Fa.r.median()*100:+.2f}% net100 {(Fa.r.mean()-.01)*100:+.2f}%  -> ratio false:true triggers = {len(Fa)}:{len(P)}')
# robustness: every trigger traded (true -> system exit, false -> same-day close), winsorized
w = lambda s: s.clip(-.5, 1.0)
if len(P) and len(Fa):
    allB = pd.concat([w(P.B), w(Fa.r)]); print(f'\nTRADE EVERY INTRADAY TRIGGER (true flips held to system exit; false triggers exit same-day close), returns winsorized to [-50%,+100%]: n {len(allB)} mean gross {allB.mean()*100:+.2f}% med {allB.median()*100:+.2f}% net100 {(allB.mean()-.01)*100:+.2f}%')
    print(f' vs SYSTEM (confirmed-close entry, one trade per flip) winsorized: n {len(P)} mean gross {w(P.A).mean()*100:+.2f}% net100 {(w(P.A).mean()-.01)*100:+.2f}% | false-trigger leg alone winsorized: mean {w(Fa.r).mean()*100:+.2f}% med {Fa.r.median()*100:+.2f}%; share of false triggers worse than -20%: {(Fa.r<-.2).mean()*100:.1f}%')
    print(f' share of true-flip B entries that were still green at the close of the flip day: {(P.cl>P.B_in).mean()*100:.0f}%')
P.to_pickle(SC + 'ext/pennyB.pkl'); Fa.to_pickle(SC + 'ext/pennyFalse.pkl')
