import pandas as pd
_rp = pd.read_parquet
pd.read_parquet = lambda p, *a, **k: _rp(str(p).replace('nq5_real.parquet', 'nq5_real_ext.parquet').replace('nq5_drive_et.parquet','nq5_drive_et_ext.parquet'), *a, **k)
s = open('/home/claude/fartpie/xs/report_161_regime_s1.py').read().replace("2026-01-30", "2026-10-31").replace("s1_trades162.pkl", "s1_trades287_ext.pkl")
exec(compile(s, 'r161', 'exec'))
