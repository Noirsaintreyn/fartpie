"""Daily market-state series for the wide universe from the vector itself: breadth (share of names in bull regime), equal-weight index from daily returns, and the vector regime of that index. Output scratchpad/ext/market_state.pkl"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); regs = {}; rets = {}
for tk, B in D.items():
    B = B[~B.index.duplicated()]
    if len(B) < 400: continue
    regs[tk] = pd.Series(run_state_machine(B.assign(datetime=B.index))[0], B.index); rets[tk] = B.close.pct_change().clip(-0.5, 1.0)
R = pd.DataFrame(regs); br = (R > 0).where(R.notna()).sum(axis=1) / R.notna().sum(axis=1); ew = pd.DataFrame(rets).mean(axis=1).fillna(0); idx = (1 + ew).cumprod(); I = pd.DataFrame({'open': idx.shift(1).fillna(1), 'high': idx * 1.002, 'low': idx * 0.998, 'close': idx}); I = I.iloc[:-0 or None]
ir = pd.Series(run_state_machine(I.assign(datetime=I.index))[0], I.index)
S = pd.DataFrame({'breadth': br, 'idx': idx, 'idx_reg': ir}); S['idx_dd'] = idx / idx.cummax() - 1; S['ma200'] = idx / idx.rolling(200).mean() - 1
chg = ir.ne(ir.shift()).cumsum(); S['idx_days_in_reg'] = ir.groupby(chg).cumcount() + 1; S.to_pickle(SC + 'market_state.pkl'); print(S.describe().T[['mean', 'min', 'max']]); print('index bull share', (ir > 0).mean(), 'regime flips', int(ir.ne(ir.shift()).sum()))
