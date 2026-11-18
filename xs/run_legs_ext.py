import pandas as pd, numpy as np, sys, pickle
_rp = pd.read_parquet
pd.read_parquet = lambda p, *a, **k: _rp(str(p).replace('nq5_real.parquet', 'nq5_real_ext.parquet'), *a, **k)
src = open('/home/claude/fartpie/xs/dump_legs280.py').read().replace('legs280.pkl', 'legs287_ext.pkl')
exec(compile(src, 'dump_legs280', 'exec'))
