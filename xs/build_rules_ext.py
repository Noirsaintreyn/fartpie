"""Nov 18 2026: extend the RULES feature tables (rules_ds2 / rules_ex) past 2026-01-30 on the spliced NQ file, to 2026-10-31. Dealer features after the greeks files end are NaN (HistGB handles NaN); own-GEX sign available to Oct 8."""
import pandas as pd, numpy as np
_rp = pd.read_parquet
pd.read_parquet = lambda p, *a, **k: _rp(str(p).replace('nq5_real.parquet', 'nq5_real_ext.parquet'), *a, **k)
_a = open('/home/claude/fartpie/xs/rules_algo2.py').read(); exec(_a[:_a.rindex("if __name__ == '__main__':")])
_ok = ok; LO, HI = pd.Timestamp('2026-01-31'), pd.Timestamp('2026-10-31')
ok = lambda d: _ok(d) and LO <= d <= HI
NEW = build(); EXN = extra()
old = _rp('/tmp/claude-0/rules_ds2.parquet'); olx = _rp('/tmp/claude-0/rules_ex.parquet')
pd.concat([old[old.day < LO], NEW], ignore_index=True).to_parquet('/tmp/claude-0/rules_ds2_ext.parquet')
pd.concat([olx[olx.day < LO], EXN], ignore_index=True).to_parquet('/tmp/claude-0/rules_ex_ext.parquet')
print('new rows', len(NEW), len(EXN), NEW.day.min(), NEW.day.max())
