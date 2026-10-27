"""Oct 27 2026: which timeframe's vector flip best confirms that the day's low (high) is in? For every RTH day, each bull (bear) flip of the
vector on 5m / 15m / 1h bars that comes AFTER the current low (high) of the day, with price above (below) it, is a 'confirmation'.
Measured per confirmation: does the low hold to the close, minutes from the low to the confirmation, and how much of the day's eventual
move away from the low (in expected ranges) was still ahead. 25 names with 5m history, 2019-2026; split 2019-21 / 2022-26.
python3 xs/extreme_tf.py BARS_DIR OUT"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUTF = A_; rows = []
def har(x): return 1.7 * (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)
for f5 in sorted(glob.glob(f'{BARS}/*_5m_*.parquet')):
    tk = os.path.basename(f5).split('_5m_')[0]; f15 = glob.glob(f'{BARS}/{tk}_15m_*.parquet')
    if not f15: continue
    H5, H15 = load(f5), load(f15[0]); H1 = agg(H15, '1h', '9h30min')
    S = {}
    for nm, H, ln in (('5m', H5, 5), ('15m', H15, 15), ('1h', H1, 60)):
        st = np.asarray(vec(H[['dt', 'open', 'high', 'low', 'close']])); k = pd.Series(st, index=H.dt + pd.Timedelta(minutes=ln))   # known at bar end
        S[nm] = k[~k.index.duplicated(keep='last')]
    R = H5[(H5.m >= 570) & (H5.m < 960)].copy(); R['d'] = R.dt.dt.normalize()
    D = R.groupby('d').agg(h=('high', 'max'), l=('low', 'min')); B = har(np.log(np.log(D.h / D.l).clip(lower=1e-5)))
    for d, g in R.groupby('d'):
        b = B.get(d, np.nan)
        if d < pd.Timestamp('2019-01-01') or not b > 0 or len(g) < 70: continue
        t_end = g.dt.values + np.timedelta64(5, 'm'); lo = np.minimum.accumulate(g.low.values); hi = np.maximum.accumulate(g.high.values)
        tl = pd.Series(np.where(g.low.values <= lo, g.dt.values, np.datetime64('NaT')), dtype='datetime64[ns]').ffill().values
        th = pd.Series(np.where(g.high.values >= hi, g.dt.values, np.datetime64('NaT')), dtype='datetime64[ns]').ffill().values
        LO, HI = lo[-1], hi[-1]; c = g.close.values
        for nm, s in S.items():
            sd = s[(s.index > g.dt.iloc[0]) & (s.index <= g.dt.iloc[-1] + pd.Timedelta(minutes=5))]
            if not len(sd): continue
            prev = s[s.index <= g.dt.iloc[0]]; p0 = prev.iloc[-1] if len(prev) else 0; vals = np.r_[p0, sd.values]
            for (t, v), pv in zip(sd.items(), vals[:-1]):
                if v == pv or v == 0: continue
                k = np.searchsorted(t_end, np.datetime64(t)) ;
                if k >= len(g) or t_end[k] != np.datetime64(t): continue
                m_ = (pd.Timestamp(t) - d).total_seconds() / 60
                if not 600 <= m_ <= 930: continue
                if v == 1 and np.datetime64(tl[k]) < np.datetime64(t) and c[k] > lo[k]:
                    rows.append((tk, d, nm, 'low', m_, LO >= lo[k], (pd.Timestamp(t) - pd.Timestamp(tl[k])).total_seconds() / 60, np.log(hi[-1] / max(c[k], hi[k] if False else c[k])) / b))
                if v == -1 and np.datetime64(th[k]) < np.datetime64(t) and c[k] < hi[k]:
                    rows.append((tk, d, nm, 'high', m_, HI <= hi[k], (pd.Timestamp(t) - pd.Timestamp(th[k])).total_seconds() / 60, np.log(c[k] / lo[-1]) / b))
    print(tk, flush=True)
pd.DataFrame(rows, columns=['tk', 'd', 'tf', 'side', 'tmin', 'holds', 'lag', 'ahead']).to_parquet(OUTF)
