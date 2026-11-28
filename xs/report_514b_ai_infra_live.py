"""Current daily vector state for the AI-infra basket, bars rebuilt from LSE 15m (RTH). Key read at runtime by lse_data (LSE_ENV_FILE)."""
import sys, os, numpy as np, pandas as pd, concurrent.futures as cf
sys.path.insert(0, '.'); import lse_data; from vector_pine import run_state_machine
exec(open('xs/report_514_ai_infra_flips.py').read().split("D = pd.read_pickle")[0])
assert lse_data.available()
def one(t):
    try: return t, lse_data.daily_rth(t, years=1.0)
    except Exception as e: return t, None
with cf.ThreadPoolExecutor(2) as ex: B = dict(ex.map(one, tk))
rows = []; miss = []
for t in tk:
    q = B.get(t)
    if q is None or len(q) < 150: miss.append(t); continue
    reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); fl = np.where((reg[1:] == 1) & (reg[:-1] != 1))[0] + 1
    rows.append(dict(tk=t, grp=grp[t], px=round(q.close.iat[-1], 2), reg=int(reg[-1]), ATRP=round(atr[-1] / q.close.iat[-1] * 100, 1), last_flip=q.index[fl[-1]].date() if len(fl) else None,
                     days_since=(len(q) - 1 - fl[-1]) if len(fl) else None, asof=q.index[-1].date()))
R = pd.DataFrame(rows).sort_values('ATRP', ascending=False); R.to_csv('xs/report_514b_ai_infra_live.csv', index=False)
print('asof', R['asof'].max(), 'n', len(R), 'missing', miss)
print('\nIN REGIME (long), by ATRP:'); print(R[R.reg == 1].to_string(index=False))
print('\nOUT of regime top ATRP:'); print(R[R.reg != 1].head(20).to_string(index=False))
