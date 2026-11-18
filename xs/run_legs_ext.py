import pandas as pd, numpy as np, sys, pickle
_rp = pd.read_parquet
_map = [('nq5_real.parquet', 'nq5_real_ext.parquet'), ('rules_ds2.parquet', 'rules_ds2_ext.parquet'), ('rules_ex.parquet', 'rules_ex_ext.parquet'), ('nq5_drive_et.parquet', 'nq5_drive_et_ext.parquet')]
def _rd(p, *a, **k):
    p = str(p)
    for a_, b_ in _map: p = p.replace(a_, b_)
    return _rp(p, *a, **k)
pd.read_parquet = _rd
src = open('/home/claude/fartpie/xs/dump_legs280.py').read().replace('legs280.pkl', 'legs287_ext.pkl').replace('s1_trades162.pkl', 's1_trades287_ext.pkl')
exec(compile(src, 'dump_legs280', 'exec'))
