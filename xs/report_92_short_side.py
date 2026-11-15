"""Nov 15 2026 SHORT SIDE OF THE VECTOR (Rey, GEVO 1H chart: short during the bear state was up before the long flip). Declared before results. The vector is a two-state classifier, so a stop-and-reverse book is long in BULL, short in BEAR. Tests (a) daily wide universe 2019-Mar 2026 (close>=$1, 20d median dollar volume >=$500k): SHORT every bear flip, entry next open, cover at the open after the opposite (bull) flip; (b) 1H penny names (60 days of 5m resampled, ONE regime): same. Costs: 30bp round trip daily / 100bp 1H, plus BORROW at 0%, 15%, 50% annualized (small caps are often hard to borrow; actual cost and availability are unknown from our data). Report per era and price bucket: n, mean, median, share of trades that LOSE >=+50% (squeeze / adverse), worst trade, ex-worst-3 mean, win rate. H: shorts on bear flips earn a positive net mean in BOTH eras at 15% borrow with a loss tail that is not dominated by a few squeezes (ex-worst-3 mean also > 0). Combined stop-and-reverse = long book (as before) + short book, equal trade weight; report mean per trade. Survivorship note: our names are survivors only, so delisted/zeroed names are missing, which biases SHORT results DOWN (real shorts would look better) but borrow/squeeze realism is not in the data."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SC + 'ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); dv = pd.Series(c * v).rolling(20).median().values
    for side, a, b in (('SHORT', -1, 1), ('LONG', 1, -1)):
        i = 130
        while i < n - 2:
            if reg[i] == a and reg[i - 1] != a:
                j = i + 1
                while j < n and reg[j] != b: j += 1
                if j + 1 < n and '2019-01-01' <= str(idx[i])[:10] < '2026-03-01' and c[i] >= 1 and dv[i] >= 5e5: rows.append((side, tk, idx[i], o[j + 1] / o[i + 1] - 1, j - i, c[i]))
                i = j
            i += 1
A = pd.DataFrame(rows, columns=['side', 'tk', 'd', 'px_ret', 'days', 'px']); A['era'] = np.where(A.d < '2022-01-01', '2019-21', '2022-26Q1'); A['mo'] = A.d.dt.to_period('M')
def net(z, borrow): 
    s = np.where(z.side == 'SHORT', -z.px_ret - .003 - borrow * z.days / 252, z.px_ret - .003); return pd.Series(s, index=z.index)
def st(x): 
    w = x.sort_values(); return f'n {len(x):5d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% win {(x>0).mean()*100:3.0f}% loss<=-50% {(x<=-.5).mean()*100:4.1f}% worst {x.min()*100:+7.0f}% ex-worst3 {w.iloc[3:].mean()*100:+6.1f}%'
print('(a) DAILY, SHORT bear flips (cover at open after bull flip), net of costs; borrow 0 / 15 / 50% annualized')
for lab, m in (('ALL', A.px > 0), ('$1-10', A.px < 10), ('$10+', A.px >= 10)):
    for en in ('2019-21', '2022-26Q1'):
        z = A[(A.side == 'SHORT') & m & (A.era == en)]; print(f' {lab:6s} {en:9s} avg hold {z.days.mean():.0f}d | b0: {st(net(z,0))}\n {"":16s} b15: {st(net(z,.15))}\n {"":16s} b50: {st(net(z,.5))}')
print('\nCOMPARE LONG (same universe, hold to bear flip):')
for en in ('2019-21', '2022-26Q1'): z = A[(A.side == 'LONG') & (A.era == en)]; print(f' ALL {en:9s} avg hold {z.days.mean():.0f}d | {st(net(z,0))}')
print('\nCOMBINED stop-and-reverse, mean per trade (15% borrow on shorts):')
for en in ('2019-21', '2022-26Q1'):
    z = A[A.era == en]; r = net(z, .15); print(f' ALL {en:9s} long+short per trade {r.mean()*100:+.2f}%  (long {r[z.side=="LONG"].mean()*100:+.2f}%, short {r[z.side=="SHORT"].mean()*100:+.2f}%)')
    zz = A[(A.era == en) & (A.px < 10)]; r = net(zz, .15); print(f' $1-10 {en:9s} long+short per trade {r.mean()*100:+.2f}%  (long {r[zz.side=="LONG"].mean()*100:+.2f}%, short {r[zz.side=="SHORT"].mean()*100:+.2f}%)')
# (b) 1H pennies
M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb')); rows = []
for tk, x in M5.items():
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York'); x = x[~x.index.duplicated()]; m = x.index.hour * 60 + x.index.minute; x = x[(m >= 570) & (m < 960)]; x = x[x.volume > 0]
    if len(x) < 400 or not .5 <= x.close.median() < 10 or (x.close * x.volume).groupby(x.index.date).sum().median() < 3e5: continue
    B = x.resample('60min', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open'])
    if len(B) < 120: continue
    reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); o = B.open.values; n = len(B)
    for side, a, b in (('SHORT', -1, 1), ('LONG', 1, -1)):
        i = 1
        while i < n - 2:
            if reg[i] == a and reg[i - 1] != a:
                j = i + 1
                while j < n and reg[j] != b: j += 1
                if j + 1 < n: rows.append((side, tk, B.index[i], o[j + 1] / o[i + 1] - 1, (j - i) / 7))
                i = j
            i += 1
P = pd.DataFrame(rows, columns=['side', 'tk', 'd', 'px_ret', 'days']); print('\n(b) 1H PENNIES 60d, closed trades, 100bp round trip')
for side in ('SHORT', 'LONG'):
    z = P[P.side == side]
    for br in (0, .15, .5): s = pd.Series(np.where(z.side == 'SHORT', -z.px_ret - .01 - br * z.days / 252, z.px_ret - .01)); print(f' {side:5s} borrow {int(br*100):2d}% avg hold {z.days.mean():.1f} sessions | {st(s)}')
