"""Oct 26 2026: an intraday strategy off the range x vector context. Signal: a 15m vector flip that confirms the day's extreme
(bull flip after the RTH low so far; mirror for shorts), 10:00-14:45. Variants by context: 15m only / 1h also agrees.
Entry next 15m open. Levels (HAR budget b = 1.7 sigma, what the site draws):
  TP at the expected high  = low so far x (1 + b)          SL at the expected low = high so far x (1 - b)
  alt SL at the day's low so far (the confirmed low)        out at the 15:45 bar close otherwise. Both hit in one bar -> SL.
python3 xs/range_intraday.py BARS_DIR OUT_FILE"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUTF = A; C2 = 2e-4; rows = []
def har(x): return 1.7 * (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)
for f in sorted(glob.glob(f'{BARS}/*_15m_*.parquet')):
    tk = os.path.basename(f).split('_15m_')[0]
    if tk in ('SPY', 'QQQ'): continue
    H = load(f); reg, s1 = states(H, vec); H['r'], H['r1'] = reg, s1
    R = H[(H.m >= 570) & (H.m < 960)].copy(); R['d'] = R.dt.dt.normalize()
    D = R.groupby('d').agg(h=('high', 'max'), l=('low', 'min')); B_ = har(np.log(np.log(D.h / D.l).clip(lower=1e-5)))
    for d, g in R.groupby('d'):
        b = B_.get(d, np.nan)
        if d < pd.Timestamp('2019-01-01') or not b > 0 or len(g) < 24: continue
        o, h, l, c, m, r, r1 = (g[k].values for k in ('open', 'high', 'low', 'close', 'm', 'r', 'r1'))
        lo, hi = np.minimum.accumulate(l), np.maximum.accumulate(h); prev = H.r.values[g.index[0] - 1]
        done = set()
        for k in range(len(g) - 2):
            if not 600 <= m[k] + 15 <= 885: continue
            pr = r[k - 1] if k else prev
            for side in (1, -1):
                if side in done or not (r[k] == side and pr != side): continue
                done.add(side); both = r1[k] == side; px = o[k + 1]
                if side == 1: tp, sl, sl2 = lo[k] * (1 + b), hi[k] * (1 - b), lo[k]
                else: tp, sl, sl2 = hi[k] * (1 - b), lo[k] * (1 + b), hi[k]
                used = np.log(hi[k] / lo[k]) / b
                for nm, s_ in (('sl_exp', sl), ('sl_day', sl2), ('no_sl', None)):
                    for tpn, t_ in (('tp', tp), ('no_tp', None)):
                        x = c[-1]; xt = len(g) - 1
                        for j in range(k + 1, len(g)):
                            if s_ is not None and (l[j] <= s_ if side == 1 else h[j] >= s_): x = min(o[j], s_) if side == 1 else max(o[j], s_); xt = j; break
                            if t_ is not None and (h[j] >= t_ if side == 1 else l[j] <= t_): x = max(o[j], t_) if side == 1 else min(o[j], t_); xt = j; break
                        rows.append((tk, d, g.dt.values[k + 1], g.dt.values[xt] + np.timedelta64(15, 'm'), side, both, used, nm, tpn,
                                     side * (x / px - 1) - C2, (tp / px - 1) * side, (px / sl - 1) * side if True else 0))
    print(tk, flush=True)
pd.DataFrame(rows, columns=['tk', 'd', 'entry', 'exit', 'side', 'both', 'used', 'sl', 'tp', 'ret', 'to_tp', 'to_sl']).to_parquet(OUTF)
