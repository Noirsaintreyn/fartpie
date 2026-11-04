"""Nov 4 2026: book test of the two context cells that held in both periods (declared before running):
 S semi names get priority in the slot queue (ratio + 1000); R bull+stress entries (SPY>200d and VIX/VIX3M>1) get priority; SR both; E drop energy (OXY, ENPH; only 2022-26 evidence, exploratory).
 40 matched runs vs live, both periods (stop_1h_ev harness)."""
import sys
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
C = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet')[['tk', 'entry', 'h1', 'sec', 'regime']]
Z = XL.merge(C, on=['tk', 'entry', 'h1'], how='left'); print('joined', len(Z), 'missing sec', Z.sec.isna().sum())
B0 = ev(Z); print('40 matched runs, 2019-21 / 2022-26'); line('L live (rebuilt)', B0)
def var(nm, boost=None, drop=None):
    Y = Z.copy()
    if boost is not None: Y['ratio'] = Y.ratio + 1000 * boost(Y)
    if drop is not None: Y = Y[~drop(Y)].reset_index(drop=True)
    line(nm, ev(Y), B0)
var('S semi first', lambda y: (y.sec == 'semi'))
var('R bull+stress first', lambda y: (y.regime == 'bull+stress'))
var('SR both first', lambda y: (y.sec == 'semi') | (y.regime == 'bull+stress'))
var('E drop energy', drop=lambda y: y.sec == 'energy')
