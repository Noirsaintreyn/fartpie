"""Nov 22 2026 report 367. Rey: 'choose 5 random stocks that flipped on the 15m AND 1h vector in the last week, use the indicator, and see if it would have helped pick good trades vs bad trades.'
Window: flips on Nov 16-20 2026 (sessions Mon-Fri), bull flips only, entry = next bar open, outcome = exit at the next bear flip of the SAME timeframe (open of the bar after it) if it already happened, otherwise marked to the last bar (Nov 20 15:45 close); 4bp round trip.  1h bars = the 15m RTH bars aggregated from 09:30 into 60-min bars, same vector settings.
INDICATOR = the volatility z-score from report 366: z = 5-day log return ending at the PRIOR daily close / (sigma x sqrt5) with sigma measured on the 20 days before that 5-day window; Yang-Zhang sigma (z_yz) and plain std (z_reg).  Oversold = z < -2 (report 366: such flips did worse in both halves; skip rule not admitted).
Random pick: seed 7, among names with >=1 bull flip on BOTH timeframes in the window.  n is tiny: this is an illustration, not a test; the same indicator is also scored on every 15m flip of the week (all 54 names) as a slightly bigger sample."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
W0 = pd.Timestamp('2026-11-16'); W1 = pd.Timestamp('2026-11-21'); COST = .0004
def agg1h(H):
    g = H.resample('60min', origin='start_day', offset='30min'); o = pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last(), 'volume': g.volume.sum()}).dropna(); return o
def daily(H):
    g = H.groupby(H.index.normalize()); return pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()})
def zs(d):
    r = np.log(d.c).diff(); n = 20; k = .34 / (1.34 + (n + 1) / (n - 1)); on = np.log(d.o / d.c.shift()); oc = np.log(d.c / d.o); rs = np.log(d.h / d.c) * np.log(d.h / d.o) + np.log(d.l / d.c) * np.log(d.l / d.o)
    yz = np.sqrt((on.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()).clip(lower=1e-10)); reg = r.rolling(20).std(); r5 = np.log(d.c / d.c.shift(5))
    return pd.DataFrame({'z_reg': r5 / (reg.shift(5) * np.sqrt(5)), 'z_yz': r5 / (yz.shift(5) * np.sqrt(5)), 'r5': r5})
def flips(H, tf):
    regs, lv, atr, fl = run_state_machine(H); out = []; last = len(H) - 1
    for i, d in fl:
        if d != 1 or i + 1 > last: continue
        t = H.index[i]
        if not (W0 <= t < W1): continue
        ent = H.open.values[i + 1]; nxt = [j for j, dd in fl if dd == -1 and j > i]
        if nxt and nxt[0] + 1 <= last: ix = nxt[0] + 1; px = H.open.values[ix]; st = 'closed %s' % H.index[ix].strftime('%a %H:%M')
        else: px = H.close.values[last]; st = 'open (marked)'
        lo = H.low.values[i + 1:(nxt[0] + 1 if nxt and nxt[0] + 1 <= last else last + 1)].min()
        out.append(dict(tf=tf, flip=t, entry=ent, ret=px / ent - 1 - COST, mae=lo / ent - 1, status=st))
    return out
rows = []
for f in sorted(glob.glob(SP + 'lastweek/*.parquet')):
    tk = f.split('/')[-1][:-8]; H = pd.read_parquet(f); Z = zs(daily(H)); day = Z.index
    for tf, X in (('15m', H), ('1h', agg1h(H))):
        for r in flips(X, tf):
            pdays = Z.index[Z.index < r['flip'].normalize()]; z = Z.loc[pdays[-1]] if len(pdays) else None
            rows.append(dict(tk=tk, **r, z_yz=z.z_yz if z is not None else np.nan, z_reg=z.z_reg if z is not None else np.nan, r5=z.r5 if z is not None else np.nan))
A = pd.DataFrame(rows); pd.set_option('display.width', 250)
both = sorted(set(A[A.tf == '15m'].tk) & set(A[A.tf == '1h'].tk)); print('names with a bull flip on both 15m and 1h Nov 16-20: %d (%s)' % (len(both), ' '.join(both)))
rng = np.random.default_rng(7); pick = list(rng.choice(both, size=min(5, len(both)), replace=False)); print('random 5 (seed 7):', pick, '\n')
S = A[A.tk.isin(pick)].sort_values(['tk', 'tf', 'flip']); S['verdict'] = np.where(S.z_yz < -2, 'OVERSOLD (avoid)', 'ok'); S['trade'] = np.where(S.ret > 0, 'GOOD', 'bad')
print(S[['tk', 'tf', 'flip', 'entry', 'ret', 'mae', 'status', 'z_reg', 'z_yz', 'r5', 'verdict', 'trade']].round(3).to_string(index=False))
print('\nindicator flagged OVERSOLD (z_yz<-2): n=%d, mean ret %+.2f%%, good %d | not flagged: n=%d, mean ret %+.2f%%, good %d' % ((S.z_yz < -2).sum(), S[S.z_yz < -2].ret.mean() * 100, (S[S.z_yz < -2].ret > 0).sum(), (S.z_yz >= -2).sum(), S[S.z_yz >= -2].ret.mean() * 100, (S[S.z_yz >= -2].ret > 0).sum()))
print('\n== bigger sample: every bull flip of the week, all 54 names ==')
for tf in ('15m', '1h'):
    T = A[A.tf == tf].dropna(subset=['z_yz']); o = T[T.z_yz < -2]; n_ = T[T.z_yz >= -2]
    print('%s: %d flips | oversold n=%d mean %+.2f%% good %.0f%% | not oversold n=%d mean %+.2f%% good %.0f%% | Spearman(z_yz, ret) %+.2f, Spearman(z_reg, ret) %+.2f' % (tf, len(T), len(o), o.ret.mean() * 100, (o.ret > 0).mean() * 100 if len(o) else np.nan, len(n_), n_.ret.mean() * 100, (n_.ret > 0).mean() * 100, T[['z_yz', 'ret']].corr('spearman').iloc[0, 1], T[['z_reg', 'ret']].corr('spearman').iloc[0, 1]))
