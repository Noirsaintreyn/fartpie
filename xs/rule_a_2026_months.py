"""Nov 10 2026: 2026 ONLY - an account started on the first session of each month (Rule A final system S1: Rule A trades, Yang-Zhang + VWAP sizing, eval skips flips against the 09:30 gap).
The full current system (S6, 7 engines) only has trade data to 30 Jan 2026 (real NQ file), so Rule A is the one engine with 2026 data to the end (QQQ proxy, through Oct 2026).
Lucid 50K as in xs/nq_final_system.py: eval fee, activation fee, real-time drawdown. Per start: result, sessions to pass, funded payouts (day, $), fees paid, net = payouts - fees, still open at the data end."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import yfinance as yf
Dq = yf.download('QQQ', start='2015-06-01', interval='1d', progress=False, auto_adjust=False); Dq.columns = [c[0] if isinstance(c, tuple) else c for c in Dq.columns]; Dq.index = Dq.index.normalize()
_fs = open('xs/nq_final_system.py').read(); exec(_fs[:_fs.rindex("if __name__ == '__main__':")])
f = LUC; De, Dpre, Dpost = DS['GAPg'], DS['base'], DS['base']; print('fee keys:', {k: f[k] for k in f if k in ('fee', 'act', 'name')}, '| data ends', D_[NS - 1].date(), flush=True)
rows = []
for mth in range(1, 11):
    s = int(np.searchsorted(D_, pd.Timestamp(f'2026-{mth:02d}-01')))
    if s >= NS - 3: continue
    res, k, fees = evS(f, s, NS, De); r = dict(start=D_[s].date(), res=res, fees=fees, pass_days=None, pays=[])
    if res == 'pass':
        r['pass_days'] = k - s + 1; r['fees'] = fees + f['act']; fr, k2, pays = fuS(f, k + 1, NS, Dpre, Dpost); r['pays'] = [(D_[kk].date(), a) for kk, a in pays]; r['fund'] = fr
    rows.append(r)
print('\nstart | eval result | sessions to pass | funded payouts (date, $) | fees | net (payouts - fees)')
for r in rows:
    got = sum(a for _, a in r['pays']); print(f"  {r['start']} | {r['res']:<5} | {r['pass_days']} | {[(str(d), round(a)) for d, a in r['pays']]} {r.get('fund', '')} | ${r['fees']:,.0f} | ${got - r['fees']:+,.0f}")
paid = [r for r in rows if r['pays']]; print(f"\nstarts {len(rows)} | passed {sum(r['res'] == 'pass' for r in rows)} | with at least one payout {len(paid)} | total payouts ${sum(a for r in rows for _, a in r['pays']):,.0f} | total fees ${sum(r['fees'] for r in rows):,.0f}")
