"""Nov 9 2026: per screener flip, the stock's entry price (open of the entry 15m bar) and that session's close (last RTH 15m close) - for the 0DTE / short-dated
call test (xs/opt_0dte.py). Run where lse_clean exists (the Mac). Writes flip_intraday.csv next to this file (or to argv[1])."""
import glob, os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = pd.read_parquet(os.path.join(ROOT, 'xs/ext/wide_T.parquet')); T = T[T.sp & (T.ratio >= 1.5)]
out = []
for tk, g in T.groupby('tk'):
    f = glob.glob(os.path.join(ROOT, f'lse_clean/{tk}_15m_*.parquet'))
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').set_index('dt').sort_index()
    cl = H.close.groupby(H.index.normalize()).last(); k = H.index.get_indexer(g.entry.values)
    pe = np.where(k >= 0, H.open.values[np.maximum(k, 0)], np.nan); pc = cl.reindex(g.entry.dt.normalize().values).values
    out.append(pd.DataFrame({'tk': tk, 'entry': g.entry.values, 'px_e': pe, 'px_c': pc}))
O = pd.concat(out); dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flip_intraday.csv')
O.to_csv(dst, index=False); print(len(O), 'rows ->', dst, '| entry price found', f'{O.px_e.notna().mean():.0%}')
