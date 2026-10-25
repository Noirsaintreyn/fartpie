"""Open type at entry, vs the vector level the flip crossed (15m trades: 15m vector; 1h trades: 1h vector).
V = vector level (bear-state line) at the prior session's close, A = its ATR.  z = (entry-day RTH open - V) / A
  gap_thru   z > 0.35  (opened past the trigger: flipped on the open)
  open_above 0 < z <= 0.35 (opened over the line but under the trigger)
  just_below -1 < z <= 0   (opened just under, crossed in the session)
  crossed    z <= -1       (opened well under, ran through it)
  was_bull   the vector was already bull at the prior close (flip came after a premarket/intraday whipsaw)
Same with the week: W = level at the prior week's last close, z_w from the week's first RTH open (for trades entered that week).
Stage 1 TK..: xs/ext/otype/<tk>.parquet ; stage 2: averages + book sims."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
def cut(z):
    return np.where(np.isnan(z), 'was_bull', np.where(z > .35, 'gap_thru', np.where(z > 0, 'open_above', np.where(z > -1, 'just_below', 'crossed'))))
if sys.argv[1] == '1':
    os.makedirs('xs/ext/otype', exist_ok=True)
    T = pd.read_parquet('xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5)]
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:] or sorted(set(T.tk)):
        out = f'xs/ext/otype/{tk}.parquet'
        if os.path.exists(out) or tk not in fs: continue
        try:
            H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            m = H.dt.dt.hour * 60 + H.dt.dt.minute
            H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
            H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute
            reg, lv, atr, _ = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt))
            H['reg'], H['lv'], H['atr'] = np.asarray(reg), np.asarray(lv, float), np.asarray(atr, float)
            x = H[H.m >= 570].set_index('dt')[['open', 'high', 'low', 'close']]
            g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')])
            r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']
            r2, l2, a2, _ = run_state_machine(r[['open', 'high', 'low', 'close']].assign(datetime=r.dt))
            S1 = pd.DataFrame({'known': r.dt + pd.Timedelta('1h'), 'reg': np.asarray(r2), 'lv': np.asarray(l2, float), 'atr': np.asarray(a2, float)})
            R = H[H.m >= 570]; day = R.dt.dt.normalize()
            DO = R.groupby(day).open.first(); DC_t = R.groupby(day).dt.last()               # RTH open per day, last RTH bar per day
            wk = DO.index.to_period('W'); WO = DO.groupby(wk).first(); WD = pd.Series(DO.index, index=DO.index).groupby(wk).first()
            last15 = H.set_index('dt'); rows = []
            Tk = T[T.tk == tk]
            def lvl(h1, t):                                                                 # vector state/level/atr as known at time t (bar ending at t)
                if h1:
                    k = S1.known.searchsorted(t, 'right') - 1
                    return (np.nan,) * 3 if k < 0 else (S1.reg.iloc[k], S1.lv.iloc[k], S1.atr.iloc[k])
                k = H.dt.searchsorted(t - pd.Timedelta('15min'), 'right') - 1
                return (np.nan,) * 3 if k < 0 else (H.reg.iloc[k], H.lv.iloc[k], H.atr.iloc[k])
            for e, h1 in zip(Tk.entry, Tk.h1):
                d = e.normalize(); k = DO.index.searchsorted(d)
                if k == 0 or k >= len(DO) or DO.index[k] != d: rows.append((tk, h1, e, np.nan, np.nan, np.nan)); continue
                pc_t = DC_t.iloc[k - 1] + pd.Timedelta('15min'); s, v, a = lvl(h1, pc_t)
                z = np.nan if s != -1 or not a > 0 else (DO.iloc[k] - v) / a
                w = d.to_period('W'); wd = WD.loc[w]; kw = DO.index.searchsorted(wd); zw = np.nan
                if kw > 0:
                    s2, v2, a2_ = lvl(h1, DC_t.iloc[kw - 1] + pd.Timedelta('15min'))
                    zw = np.nan if s2 != -1 or not a2_ > 0 else (WO.loc[w] - v2) / a2_
                rows.append((tk, h1, e, z, zw, float(d == wd)))
            pd.DataFrame(rows, columns=['tk', 'h1', 'entry', 'z', 'zw', 'wk1']).to_parquet(out); print(tk, end=' ', flush=True)
        except Exception as ex: print(f'\n{tk} ERR {ex}', flush=True)
else:
    exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])
    F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/otype/*.parquet'))
    T = T.merge(F, on=['tk', 'h1', 'entry'], how='left').sort_values('entry').reset_index(drop=True)
    T['ot'] = cut(T.z.values); T['ow'] = cut(T.zw.values); T.loc[T.z.isna() & T.zw.isna() & T.wk1.isna(), ['ot', 'ow']] = 'n/a'
    T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21'); T['trk'] = np.where(T.h1, '1h', '15m')
    for col, nm in (('ot', 'DAY open vs vector'), ('ow', 'WEEK open vs vector')):
        g = T.groupby([col, 'per']).ret.agg(['mean', 'count', lambda r: (r > 0).mean()]); g.columns = ['avg%', 'n', 'win']; g['avg%'] *= 100
        print(f'\n{nm}: avg trade %, count, win rate'); print(g.unstack().round(3).to_string())
        g = T.groupby([col, 'trk']).ret.mean().mul(100).unstack().round(3); print(' by track (all years):'); print(g.to_string())
    print('\nWEEK first-day entries only (opened the week):')
    g = T[T.wk1 == 1].groupby(['ow', 'per']).ret.agg(['mean', 'count']); g['mean'] *= 100; print(g.unstack().round(3).to_string())
    head('OPEN TYPE book tests (skip a type / prefer a type)')
    for col in ('ot', 'ow'):
        for t in ('gap_thru', 'open_above', 'just_below', 'crossed', 'was_bull'):
            show(f'skip {col}={t} ({(T[col] == t).sum()})', T[T[col] != t])
    for col in ('ot', 'ow'):
        for t in ('gap_thru', 'crossed', 'just_below'):
            Z = T.copy(); Z['ratio'] = Z.ratio + np.where(Z[col] == t, 100, 0); show(f'prefer {col}={t} first', Z)
