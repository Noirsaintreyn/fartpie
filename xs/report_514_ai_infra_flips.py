"""AI-infrastructure bottleneck basket (own theme mapping, NOT gstang57's list): current daily vector state + ATRP rank + flip history."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, '.')
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TH = {
 'memory/storage': 'MU SNDK WDC STX PSTG NTAP SIMO',
 'optics/network': 'COHR LITE AAOI CIEN FN CRDO ALAB ANET MRVL AVGO CLS APH GLW CALX',
 'semis/foundry/equip': 'TSM NVDA AMD INTC ASML AMAT LRCX KLAC ONTO ACLS AEHR AXTI POET WOLF NVTS ICHR UCTT AMKR',
 'power/cooling': 'VRT ETN GEV VST CEG NRG TLN OKLO SMR BE FLNC NVT POWL PWR AAON MOD SPXC',
 'servers/EMS': 'SMCI DELL HPE JBL CLS FLEX CLMB',
 'neocloud/DC': 'CRWV NBIS IREN APLD CIFR WULF CORZ EQIX DLR',
 'industrial/materials': 'FIX EME STRL MTZ MYRG FCX SCCO TECK COPX',
}
tk = sorted({t for v in TH.values() for t in v.split()}); grp = {t: g for g, v in TH.items() for t in v.split()}
D = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl'); rows = []; miss = []
for t in tk:
    q = D.get(t)
    if q is None or len(q) < 150: miss.append(t); continue
    q = q.dropna(subset=['close']).copy(); q.index = pd.DatetimeIndex(q.index)
    reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index))
    fl = np.where((reg[1:] == 1) & (reg[:-1] != 1))[0] + 1
    last = q.index[fl[-1]] if len(fl) else None
    rows.append(dict(tk=t, grp=grp[t], px=round(q.close.iat[-1], 2), reg=int(reg[-1]), ATRP=round(atr[-1] / q.close.iat[-1] * 100, 1),
        last_flip=last.date() if last is not None else None, days_since=(len(q) - 1 - fl[-1]) if len(fl) else None, asof=q.index[-1].date()))
R = pd.DataFrame(rows).sort_values('ATRP', ascending=False); R.to_csv('xs/report_514_ai_infra_flips.csv', index=False)
print('asof', R['asof'].max(), 'missing', miss)
print('FRESH (reg=1, flipped <=5d):'); print(R[(R.reg == 1) & (R.days_since <= 5)].to_string(index=False))
print('\nIN-REGIME older:'); print(R[(R.reg == 1) & (R.days_since > 5)].head(25).to_string(index=False))
print('\nNOT IN REGIME top ATRP:'); print(R[R.reg != 1].head(15).to_string(index=False))
