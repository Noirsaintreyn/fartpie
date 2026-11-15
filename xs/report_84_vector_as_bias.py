"""Nov 14 2026 VECTOR AS A BIAS / CONFIRMATION LAYER FOR OTHER SIGNALS (Rey: instead of trading the flips, use 'recently flipped / bullish' as the context for other strategies). Declared before results. Universe: wide daily (1,656 names, price>=$1, 20d median dollar volume >=$500k at the signal bar, survivorship-biased). State at the close of signal bar t is known (no look-ahead): BULL_NEW = vector bullish and <=10 trading days since its bull flip, BULL_MID = 11-60 days, BULL_OLD = >60 days, BEAR = vector bearish. Four generic entry signals that have nothing to do with the vector, all causal at the close of bar t: PULLBACK = 3 consecutive lower closes; BREAKOUT = close above the prior 20-day high; THRUST = up day >=3% on volume >=2x its 20d mean; DROP = 5-day return <= -8%; plus ALL = every stock-day (baseline for the state itself). Entry next open, exit 10 trading days later at the open (open to open). Outcome = EXCESS return vs the equal-weight universe over the same window (removes market beta; flips cluster in good markets), minus 30bp costs. Report per signal x state x era (2019-21 / 2022-26Q1): n, mean excess %, median, hit rate. Month-clustered t for BULL_NEW minus BEAR and BULL_ANY minus BEAR per signal, pooled and per era. PASS (vector adds bias value) = for >=3 of the 4 signals, the BULL states beat BEAR in both eras by >=0.5pp per 10 days and pooled clustered t >= 2. Not run: 1h bars; vector as a long-only bias for shorts."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); H = 10
opn = pd.DataFrame({tk: B.open[~B.index.duplicated()] for tk, B in D.items()}).sort_index(); opn = opn[opn.index >= '2018-06-01']
ro = opn.pct_change(fill_method=None).clip(-.5, 1.0); Lm = np.log1p(ro.mean(axis=1).fillna(0)).cumsum(); mk = (Lm.shift(-(H + 1)) - Lm.shift(-1)); mk = np.expm1(mk)  # open t+1 -> open t+H+1 universe return
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']; n = len(B)
    if n < 400: continue
    o = B.open.values; h = B.high.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index))
    dv = pd.Series(c * v).rolling(20).median().values; vm = pd.Series(v).rolling(20).mean().shift(1).values; hh = pd.Series(h).rolling(20).max().shift(1).values
    last = -10 ** 6; age = np.zeros(n)
    for i in range(1, n):
        if reg[i] == 1 and reg[i - 1] != 1: last = i
        age[i] = i - last if reg[i] == 1 else -1
    r1 = np.r_[np.nan, c[1:] / c[:-1] - 1]; r5 = np.r_[[np.nan] * 5, c[5:] / c[:-5] - 1]
    pb = np.zeros(n, bool); pb[3:] = (c[3:] < c[2:-1]) & (c[2:-1] < c[1:-2]) & (c[1:-2] < c[:-3])
    sig = dict(ALL=np.ones(n, bool), PULLBACK=pb, BREAKOUT=c > hh, THRUST=(r1 >= .03) & (v >= 2 * vm), DROP=r5 <= -.08)
    for i in range(131, n - H - 2):
        if c[i] < 1 or not dv[i] >= 5e5 or reg[i] == 0: continue
        st = 'BEAR' if reg[i] == -1 else ('BULL_NEW' if age[i] <= 10 else ('BULL_MID' if age[i] <= 60 else 'BULL_OLD')); ex = o[i + 1 + H] / o[i + 1] - 1 - 0.003
        for k, m in sig.items():
            if m[i] and (k == 'ALL' and i % 5 == 0 or k != 'ALL'): rows.append((tk, idx[i], k, st, ex))
A = pd.DataFrame(rows, columns=['tk', 'd', 'sig', 'st', 'raw']); A['mk'] = mk.reindex(A.d).values; A = A.dropna(subset=['mk']); A['ex'] = A.raw - A.mk; A['era'] = np.where(A.d < '2022-01-01', '2019-21', '2022-26Q1'); A = A[(A.d >= '2019-01-01') & (A.d < '2026-03-01')]
A['mo'] = A.d.dt.to_period('M'); A.to_pickle(X + 'bias_events.pkl'); print('events', len(A))
ORD = ['BEAR', 'BULL_NEW', 'BULL_MID', 'BULL_OLD']
def mct(z, a, b):
    x = z[z.st.isin(a)].groupby('mo').ex.mean(); y = z[z.st.isin(b)].groupby('mo').ex.mean(); d = (x - y).dropna(); return d.mean() / (d.std() / np.sqrt(len(d))) * 1.0 if len(d) > 5 else np.nan, d.mean() * 100
print('\nMEAN EXCESS RETURN PER 10-DAY ENTRY (%, net of 30bp, vs equal-weight universe) | median | hit rate | n')
for k in ['ALL', 'PULLBACK', 'BREAKOUT', 'THRUST', 'DROP']:
    for en in ('2019-21', '2022-26Q1'):
        z = A[(A.sig == k) & (A.era == en)]; print(f'  {k:9s} {en:9s}: ' + ' | '.join(f'{s[:9]:9s} {z[z.st==s].ex.mean()*100:+5.2f} ({z[z.st==s].ex.median()*100:+5.2f}, {(z[z.st==s].ex>0).mean()*100:3.0f}%, n {int((z.st==s).sum()):6d})' for s in ORD))
print('\nBIAS VALUE: BULL minus BEAR (pp per 10 days) with month-clustered t'); ok = {}
for k in ['ALL', 'PULLBACK', 'BREAKOUT', 'THRUST', 'DROP']:
    out = []
    for lab, a in (('NEW-BEAR', ['BULL_NEW']), ('ANYBULL-BEAR', ['BULL_NEW', 'BULL_MID', 'BULL_OLD'])):
        r = {}
        for en in ('2019-21', '2022-26Q1', 'POOLED'):
            z = A[A.sig == k] if en == 'POOLED' else A[(A.sig == k) & (A.era == en)]; t, dm = mct(z, a, ['BEAR']); r[en] = (dm, t)
        out.append(f'{lab}: ' + ' / '.join(f'{r[e][0]:+.2f}pp (t {r[e][1]:+.1f})' for e in ('2019-21', '2022-26Q1', 'POOLED'))); ok[(k, lab)] = r['2019-21'][0] >= .5 and r['2022-26Q1'][0] >= .5 and r['POOLED'][1] >= 2
    print(f'  {k:9s} ' + ' | '.join(out))
for lab in ('NEW-BEAR', 'ANYBULL-BEAR'): print(f'  {lab}: signals passing both eras >=0.5pp & pooled t>=2: {[k for k in ["PULLBACK","BREAKOUT","THRUST","DROP"] if ok[(k, lab)]]}')
