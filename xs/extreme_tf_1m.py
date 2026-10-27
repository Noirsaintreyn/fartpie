"""Oct 27 2026: the 1m check for xs/extreme_tf.py - yfinance keeps only ~30 days of 1m bars, so this compares 1m / 5m / 15m / 1h vector
confirmations of the day's low / high on the same recent days (bars all built from the same 1m feed). Small sample: a sanity check.
python3 xs/extreme_tf_1m.py OUT TK..."""
import sys, os, numpy as np, pandas as pd, warnings, yfinance as yf; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
OUTF, TKS = A_[0], A_[1:]; rows = []
def bars(tk):
    end = pd.Timestamp.now(tz='UTC'); P = []
    for k in range(4):
        e = end - pd.Timedelta(days=7 * k); s = e - pd.Timedelta(days=7)
        try: P.append(yf.Ticker(tk).history(start=s, end=e, interval='1m', prepost=True))
        except Exception: pass
    y = pd.concat(P).sort_index(); y = y[~y.index.duplicated()]; y.index = y.index.tz_convert('America/New_York').tz_localize(None)
    H = pd.DataFrame({'dt': y.index, 'open': y.Open.values, 'high': y.High.values, 'low': y.Low.values, 'close': y.Close.values})
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < 960) & (H.dt.dt.dayofweek < 5)].reset_index(drop=True); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; return H
def rs(H, f, off='0min'):
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]; g = x.groupby([x.index.normalize(), pd.Grouper(freq=f, origin='start_day', offset=off)])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); r.index = r.index.get_level_values(1); return r.reset_index().rename(columns={'index': 'dt'})
for tk in TKS:
    try: H1m = bars(tk)
    except Exception as e: print(tk, 'no data', e); continue
    S = {}
    for nm, H, ln in (('1m', H1m, 1), ('5m', rs(H1m, '5min'), 5), ('15m', rs(H1m, '15min'), 15), ('1h', rs(H1m[H1m.m >= 570], '1h', '9h30min'), 60)):
        H = H.rename(columns={H.columns[0]: 'dt'}); st = np.asarray(vec(H[['dt', 'open', 'high', 'low', 'close']])); k = pd.Series(st, index=pd.to_datetime(H.dt) + pd.Timedelta(minutes=ln)); S[nm] = k[~k.index.duplicated(keep='last')]
    R = H1m[(H1m.m >= 570) & (H1m.m < 960)].copy(); R['d'] = R.dt.dt.normalize()
    for d, g in R.groupby('d'):
        if len(g) < 300: continue
        te = g.dt.values + np.timedelta64(1, 'm'); lo = np.minimum.accumulate(g.low.values); hi = np.maximum.accumulate(g.high.values); c = g.close.values
        tl = pd.Series(np.where(g.low.values <= lo, g.dt.values, np.datetime64('NaT')), dtype='datetime64[ns]').ffill().values
        th = pd.Series(np.where(g.high.values >= hi, g.dt.values, np.datetime64('NaT')), dtype='datetime64[ns]').ffill().values
        for nm, s in S.items():
            sd = s[(s.index > g.dt.iloc[0]) & (s.index <= g.dt.iloc[-1] + pd.Timedelta(minutes=1))]; prev = s[s.index <= g.dt.iloc[0]]; pv = prev.iloc[-1] if len(prev) else 0
            for t, v in sd.items():
                if v == pv or v == 0: pv = v; continue
                pv = v; k = np.searchsorted(te, np.datetime64(t))
                if k >= len(g) or te[k] != np.datetime64(t): continue
                m_ = (pd.Timestamp(t) - d).total_seconds() / 60
                if not 600 <= m_ <= 930: continue
                if v == 1 and tl[k] < np.datetime64(t) and c[k] > lo[k]: rows.append((tk, d, nm, 'low', m_, lo[-1] >= lo[k], (pd.Timestamp(t) - pd.Timestamp(tl[k])).total_seconds() / 60))
                if v == -1 and th[k] < np.datetime64(t) and c[k] < hi[k]: rows.append((tk, d, nm, 'high', m_, hi[-1] <= hi[k], (pd.Timestamp(t) - pd.Timestamp(th[k])).total_seconds() / 60))
    print(tk, flush=True)
pd.DataFrame(rows, columns=['tk', 'd', 'tf', 'side', 'tmin', 'holds', 'lag']).to_parquet(OUTF)
