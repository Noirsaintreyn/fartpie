"""Nov 9 2026: export the per-flip accuracy features (the Mac's 15m bars + theta_feat) for the options-on-best-flips test (xs/opt_best.py).
Columns: tk, entry, PVWD (entry - yesterday's VWAP, / ATR), ECOMP (vs the 5-session composite value), VREC (vs today's VWAP), OTMC (short-term OTM call share
of call volume, prior close), NFLIP, GX. Writes flip_features.csv to argv[1]."""
import os, sys, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_c = open(os.path.join(_HERE, 'vwap_flip.py')).read(); exec(_c[:_c.index("\nif __name__ == '__main__':")], globals())
z = pd.read_csv(P('data/theta_feat/all_eod.csv')); z['d'] = pd.to_datetime(z.d); T['OTMC'] = np.nan
for tk, g in z.groupby('tk'):
    if tk not in TKS: continue
    s = (g.st_otm_c / g.cvol.where(g.cvol > 0)).set_axis(g.d).sort_index(); m = T.tk == tk; i = s.index.searchsorted(T.loc[m, 'day'].values, 'left') - 1
    T.loc[m, 'OTMC'] = np.where(i >= 0, s.values[np.maximum(i, 0)], np.nan)
cols = ['tk', 'entry', 'ratio', 'PVWD', 'WVWD', 'ECOMP', 'EVA', 'VREC', 'OTMC', 'NFLIP', 'GX']
T[[c for c in cols if c in T]].to_csv(sys.argv[1], index=False); print(len(T), 'rows ->', sys.argv[1], '|', ', '.join(f'{c} {T[c].notna().mean():.0%}' for c in cols[2:] if c in T))
