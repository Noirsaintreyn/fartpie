"""Oct 27 2026 (v2: adds a tighter stop - the flipping timeframe's own vector break level - and reward/risk stats). Trade the 'extreme is in' confirmation as its own strategy (intraday). Signal = the vector flip on 5m / 15m / 1h that comes
after the day's current low (long) or high (short), 10:00-15:00. Entry: next 5m open. Stop: the day's low (high) so far (5m close beyond it
-> next open). Targets (HAR budget b = 1.7 sigma):
  tp_budget  low x (1 + b)          (the site's expected high)          tp_proj  high so far x exp(q90 x b) from the time-of-day x vector table
  tp_none    no target, out at the 15:55 close
One trade per name per side per day; 2 bp cost. Book: up to 6 concurrent at 1/6, days independent (all flat by the close).
python3 xs/extreme_trade.py BARS_DIR OUT"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
src = open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]; exec(src)
v = src[src.index('def vec('):src.index('def donch(')].replace('def vec(', 'def vec_lv(')
v = v.replace("regs = np.zeros(n, int)", "regs = np.zeros(n, int); LV = np.full(n, np.nan); AT = np.full(n, np.nan)")
v = v.replace("regs[i] = regime; continue", "regs[i] = regime; LV[i] = level; AT[i] = a; continue").replace("        regs[i] = regime\n    return regs", "        regs[i] = regime; LV[i] = level; AT[i] = a\n    return regs, LV, AT")
exec(v); BA = BASE['BREAK_ATR']
BARS, OUTF = A_; rows = []; C2 = 2e-4
Q90 = {0: .68, 1: .45, 2: .26, 3: .13}                         # remaining move beyond the other extreme, 90th pct, after a 15m-confirmed extreme
def har(x): return 1.7 * (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)
for f5 in sorted(glob.glob(f'{BARS}/*_5m_*.parquet')):
    tk = os.path.basename(f5).split('_5m_')[0]; f15 = glob.glob(f'{BARS}/{tk}_15m_*.parquet')
    if not f15 or tk in ('QQQ',): continue
    H5, H15 = load(f5), load(f15[0]); H1 = agg(H15, '1h', '9h30min'); S = {}
    for nm, H, ln in (('5m', H5, 5), ('15m', H15, 15), ('1h', H1, 60)):
        st, lv, at = vec_lv(H[['dt', 'open', 'high', 'low', 'close']]); ix = H.dt + pd.Timedelta(minutes=ln)
        k = pd.DataFrame({'s': st, 'brk': np.where(st == 1, lv - BA * at, lv + BA * at)}, index=ix); S[nm] = k[~k.index.duplicated(keep='last')]
    R = H5[(H5.m >= 570) & (H5.m < 960)].copy(); R['d'] = R.dt.dt.normalize()
    D = R.groupby('d').agg(h=('high', 'max'), l=('low', 'min')); B = har(np.log(np.log(D.h / D.l).clip(lower=1e-5)))
    for d, g in R.groupby('d'):
        b = B.get(d, np.nan)
        if d < pd.Timestamp('2019-01-01') or not b > 0 or len(g) < 70: continue
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values; te = g.dt.values + np.timedelta64(5, 'm'); m = g.m.values
        lo, hi = np.minimum.accumulate(l), np.maximum.accumulate(h)
        tl = pd.Series(np.where(l <= lo, np.arange(len(g)), np.nan)).ffill().values; th = pd.Series(np.where(h >= hi, np.arange(len(g)), np.nan)).ffill().values
        for nm, s in S.items():
            sd = s[(s.index > g.dt.iloc[0]) & (s.index <= g.dt.iloc[-1])]; prev = s[s.index <= g.dt.iloc[0]]; pv = prev.s.iloc[-1] if len(prev) else 0; done = set()
            for t, v, brk in zip(sd.index, sd.s.values, sd.brk.values):
                if v == pv: continue
                pv_old, pv = pv, v; k = np.searchsorted(te, np.datetime64(t))
                if k >= len(g) - 2 or te[k] != np.datetime64(t) or not 600 <= m[k] + 5 <= 900: continue
                side = 1 if v == 1 else -1
                if side in done: continue
                if side == 1 and not (tl[k] < k + 1 and c[k] > lo[k]): continue
                if side == -1 and not (th[k] < k + 1 and c[k] < hi[k]): continue
                done.add(side); px = o[k + 1]; stop_ext = lo[k] if side == 1 else hi[k]; tb = 0 if m[k] < 660 else 1 if m[k] < 780 else 2 if m[k] < 900 else 3
                tgts = {'tp_budget': lo[k] * (1 + b) if side == 1 else hi[k] * (1 - b), 'tp_proj': hi[k] * np.exp(Q90[tb] * b) if side == 1 else lo[k] * np.exp(-Q90[tb] * b), 'tp_none': None}
                for (tn, tg), (sn, stop) in [(a_, b_) for a_ in tgts.items() for b_ in (('ext', stop_ext), ('vlev', brk))]:
                    if tg is not None and (tg - px) * side <= 0: continue
                    if not (stop - px) * side < 0: continue
                    x = c[-1]; xi = len(g) - 1
                    for j in range(k + 1, len(g)):
                        if tg is not None and (h[j] >= tg if side == 1 else l[j] <= tg): x = max(o[j], tg) if side == 1 else min(o[j], tg); xi = j; break
                        if (c[j] < stop if side == 1 else c[j] > stop) and j + 1 < len(g): x = o[j + 1]; xi = j + 1; break
                    rows.append((tk, d, nm, side, tn, sn, g.dt.values[k + 1], g.dt.values[xi], side * (x / px - 1) - C2, abs(px / stop - 1), abs(tg / px - 1) if tg is not None else np.nan))
    print(tk, flush=True)
pd.DataFrame(rows, columns=['tk', 'd', 'tf', 'side', 'tp', 'stop', 'entry', 'exit', 'ret', 'risk', 'reward']).to_parquet(OUTF)
