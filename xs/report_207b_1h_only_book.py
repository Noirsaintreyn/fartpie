"""Report 207b: large-cap book (6 slots, 8% breaker, 2bp, ratio>=1.5) with 1H-only, 15m-only, or both tracks (live). CAGR / maxDD 2019-21 | 2022-26 | 2019-26 | 2026 YTD."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.argv=['x','/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet']; src=open('report_196_one_slot_per_name.py').read(); src=src[:src.index("P = [(np.datetime64")]; ns={}; exec(src,ns)
T0=ns['T'].copy(); P=[(np.datetime64(a),np.datetime64(b)) for a,b in (('2019-01-01','2022-01-01'),('2022-01-01','2027-01-01'),('2019-01-01','2027-01-01'),('2026-01-01','2027-01-01'))]
for nm,m in (('both tracks (live)',np.ones(len(T0),bool)),('1H only',T0.h1.values),('15m only',~T0.h1.values)):
    ns['T']=T0[m].reset_index(drop=True); res=[ns['sim'](a,b) for a,b in P]; print(f'  {nm:20s}'+''.join(f'{c:>+9.1%} /{d:>5.0%}' for c,d,_ in res),f' trades {len(ns["T"])}',flush=True)
