"""Nov 15 2026 WHICH BEAR FLIPS ARE WORTH SHORTING (Rey after looking at the 1H losers: JMIA/ABEV/AEHL shorts looked good, funds like VVR/IGD are choppy). Declared before results. Same short book as report_92 (daily wide universe, bear flip, entry next open, cover open after the bull flip, 30bp + 15% annualized borrow). Features at the bear-flip close: EXT = close / 60d low - 1 (blow-off if high), R60 = 60d return (already in a downtrend if low), ATRP = ATR14 / close (chop/fund-like if low), LEG = close at the bear flip / close at the preceding bull flip - 1 (size of the bull leg just ended; AEHL-type if large). H_S1: high EXT or high LEG (blow-off tops) -> better shorts. H_S2: low R60 (persistent downtrend) -> better shorts. H_S3: low ATRP (funds / low-vol chop) -> worse shorts. Terciles within month. PASS = top-vs-bottom tercile spread in the predicted direction in BOTH eras, pooled month-clustered t >= 2, AND the best tercile's mean net short return > 0 in both eras. Survivorship biases shorts down; borrow availability unmodeled."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; c = B.close.values; h = B.high.values; l = B.low.values; v = B.volume.values.astype(float); idx = B.index; n = len(B); reg, _, atr, _ = run_state_machine(B.assign(datetime=B.index)); dv = pd.Series(c * v).rolling(20).median().values; lo60 = pd.Series(l).rolling(60).min().values; lastbull = 0
    i = 130
    while i < n - 2:
        if reg[i] == 1 and reg[i - 1] != 1: lastbull = i
        if reg[i] == -1 and reg[i - 1] != -1:
            j = i + 1
            while j < n and reg[j] != 1: j += 1
            if j + 1 < n and '2019-01-01' <= str(idx[i])[:10] < '2026-03-01' and c[i] >= 1 and dv[i] >= 5e5:
                rows.append((tk, idx[i], -(o[j + 1] / o[i + 1] - 1) - .003 - .15 * (j - i) / 252, c[i] / lo60[i] - 1, c[i] / c[i - 60] - 1, atr[i] / c[i], c[i] / c[lastbull] - 1 if lastbull else np.nan, c[i]))
            i = j
        i += 1
F = pd.DataFrame(rows, columns=['tk', 'd', 's', 'EXT', 'R60', 'ATRP', 'LEG', 'px']).replace([np.inf, -np.inf], np.nan).dropna(); F['era'] = np.where(F.d < '2022-01-01', '2019-21', '2022-26Q1'); F['mo'] = F.d.dt.to_period('M'); print('bear flips', len(F), 'base short mean', round(F.s.mean() * 100, 2))
for f in ('EXT', 'R60', 'ATRP', 'LEG'): F[f + '_t'] = F.groupby('mo')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if len(s) >= 6 else np.nan)
st = lambda x: f'n {len(x):5d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% win {(x>0).mean()*100:3.0f}%'
pred = {'EXT': 1, 'LEG': 1, 'R60': -1, 'ATRP': -1}
for f, sg in pred.items():
    print(f'\n {f}: H predicts {"HIGH" if sg>0 else "LOW"} tercile is the better short'); ok = {}
    for en in ('2019-21', '2022-26Q1', 'POOLED'):
        z = F if en == 'POOLED' else F[F.era == en]; t0 = z[z[f + '_t'] == 0]; t1 = z[z[f + '_t'] == 1]; t2 = z[z[f + '_t'] == 2]; a, b = (t2, t0) if sg > 0 else (t0, t2); d = (a.groupby('mo').s.mean() - b.groupby('mo').s.mean()).dropna(); t = d.mean() / (d.std() / np.sqrt(len(d))); ok[en] = (a.s.mean() - b.s.mean(), t, a.s.mean())
        print(f'  {en:9s} T0 {st(t0.s)} | T1 {st(t1.s)} | T2 {st(t2.s)} || best-minus-worst {(a.s.mean()-b.s.mean())*100:+.1f}pp t {t:+.1f}')
    print('  ->', 'PASS' if all(ok[e][0] > 0 and ok[e][2] > 0 for e in ('2019-21', '2022-26Q1')) and ok['POOLED'][1] >= 2 else 'no pass')
# combined: blow-off AND downtrend persistence AND not low-vol
z = F[(F.EXT_t == 2) & (F.ATRP_t >= 1)]; print('\nCOMBO high-EXT & not-low-ATRP:', {en: st(z[z.era == en].s) for en in ('2019-21', '2022-26Q1')})
z = F[(F.R60_t == 0) & (F.ATRP_t >= 1)]; print('COMBO low-R60 & not-low-ATRP :', {en: st(z[z.era == en].s) for en in ('2019-21', '2022-26Q1')})
