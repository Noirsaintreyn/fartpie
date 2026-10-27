"""Oct 27 2026: can the 'low/high is in' alert fire EARLIER and still carry edge? Replays the live time-of-day x vector model (RNG_Q, range_states)
bar by bar on 2019-26 15m bars. For each rule, the FIRST bar per side per day that qualifies -> alert at that bar's close.
Outcomes from the alert price c: held (extreme never broken later), fav = best move toward the other side, adv = move to the extreme
(risk if the stop sits at the low/high), close = signed move to 16:00. Also a no-signal baseline: the same side at a fixed clock time."""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault('SITE_SNAPSHOT', '0')
import live_screener as LS
RTH0, RTH1, Q = 570, 960, LS.RNG_Q
out = []
files = sorted(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_*.parquet'))
for f in files:
    tk = os.path.basename(f).split('_15m_')[0]
    if tk in ('QQQ', 'SPY'): continue
    H = pd.read_parquet(f); H['dt'] = H.datetime.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
    H = H[['dt', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    r15, r1 = LS.range_states(H)
    m = (H.dt.dt.hour * 60 + H.dt.dt.minute).values; rth = (m >= RTH0) & (m < RTH1)
    R = H[rth].copy(); R['m'] = m[rth]; R['r15'] = r15[rth]; R['r1'] = r1[rth]; R['d'] = R.dt.dt.normalize()
    D = R.groupby('d').agg(h=('high', 'max'), l=('low', 'min')); x = np.log(np.log(D.h / D.l).clip(lower=1e-5))
    B = (1.7 * np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)
    for d, g in R.groupby('d'):
        if d < pd.Timestamp('2019-01-01') or len(g) != 26 or not B.get(d, 0) > 0: continue
        b = B[d]; o, h, l, c, mm, s15, s1 = (g[k].values for k in ('open', 'high', 'low', 'close', 'm', 'r15', 'r1'))
        rlo, rhi, tl, th = np.inf, -np.inf, 0, 0; fu15 = fd15 = fu1 = fd1 = -1; fired = {}
        for k in range(26):
            if l[k] < rlo: rlo, tl = l[k], k
            if h[k] > rhi: rhi, th = h[k], k
            p15 = s15[k - 1] if k else 0; p1 = s1[k - 1] if k else 0
            if s15[k] == 1 and p15 != 1: fu15 = k
            if s15[k] == -1 and p15 != -1: fd15 = k
            if s1[k] == 1 and p1 != 1: fu1 = k
            if s1[k] == -1 and p1 != -1: fd1 = k
            def cx(a15, a1, f15, f1, t0):
                A, Z = a15 == 1 and f15 >= t0, a1 == 1 and f1 >= t0
                return 'both' if A and Z else '15m' if A else '1h' if Z else ('old' if k - t0 >= 3 else 'fresh')
            tb = LS._tb(mm[k] + 15); cl = cx(s15[k], s1[k], fu15, fu1, tl); ch = cx(-s15[k], -s1[k], fd15, fd1, th)
            if k == 25: break
            for side, ctx, lvl in (('low', cl, rlo), ('high', ch, rhi)):
                if side in fired: continue
                p = Q[f'{tb}{ctx}'][0] * 100
                sg = 1 if side == 'low' else -1
                fav = (h[k + 1:].max() / c[k] - 1) if sg == 1 else (1 - l[k + 1:].min() / c[k])
                held = (l[k + 1:].min() >= lvl) if sg == 1 else (h[k + 1:].max() <= lvl)
                rec = dict(tk=tk, d=d, side=side, t=int(mm[k] + 15), ctx=ctx, p=p, b=b, held=held, fav=fav, adv=abs(c[k] / lvl - 1),
                           close=sg * (c[-1] / c[k] - 1), yr=d.year)
                out.append(rec)
    print(tk, len(out), flush=True)
pd.DataFrame(out).to_parquet('/tmp/claude-0/xt_early.parquet')
