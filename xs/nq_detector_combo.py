"""Nov 4 2026, POST-HOC: combine the variance-ratio detector, the last-40-trades detector and 2026 mode (half size against the 1h vector), incl. eval-only use.
Base = final system (Rule A, Yang-Zhang + VWAP sizing, 50 / 75, gap-gated eval). Lucid 50K ($88) + Topstep.
  normal | VR (2026 mode when VR > 1) | LAST40 (when the last 40 trades favour WITH-1h) | AND (both) | OR (either) | VOTE (against-1h x0.75 with one, x0.5 with both)
  EVAL-VR: VR sizing in the eval only, funded normal | EVAL-2026: 2026 mode in the eval only, funded normal | EVAL-OR: OR in the eval only, funded normal"""
_rg = open('xs/nq_regime_gate.py').read(); exec(_rg[:_rg.rindex("if __name__ == '__main__':")])
vrt, pft = TREND['VR'], TREND['PERF']
M = {'normal': MULT['always normal'], 'VR': MULT['VR-1'], 'LAST40': MULT['PERF-1'], 'AND': pd.Series(np.where(vrt & pft & ag1, .5, 1.0), index=XA.index),
     'OR': pd.Series(np.where((vrt | pft) & ag1, .5, 1.0), index=XA.index), 'VOTE': pd.Series(np.where(ag1, (vrt.astype(int) + pft.astype(int)).map({0: 1.0, 1: .75, 2: .5}), 1.0), index=XA.index),
     '2026': MULT['always 2026 mode']}
for nm, m in M.items(): DS[f'e_{nm}'] = build(~AG['GAP'], mult=m); DS[f'f_{nm}'] = build(mult=m)
VAR = {nm: (f'e_{nm}', f'f_{nm}', f'f_{nm}') for nm in ('normal', 'VR', 'LAST40', 'AND', 'OR', 'VOTE')}
VAR.update({'EVAL-VR / funded normal': ('e_VR', 'f_normal', 'f_normal'), 'EVAL-2026 / funded normal': ('e_2026', 'f_normal', 'f_normal'), 'EVAL-OR / funded normal': ('e_OR', 'f_normal', 'f_normal')})
for nm in ('AND', 'OR', 'VOTE'):
    usd = RLx.net * 20 * RLx.bs * M[nm].reindex(RLx.index); print(f'  {nm:<5} $ by year: ' + ' | '.join(f'{y} {usd[RLx.y == y].sum():+7,.0f}' for y in ('2018-21', '2022', '2023', '2024', '2025', '2026')), flush=True)
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nDETECTOR COMBINATIONS', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<26} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} (10th {r["k_p10"]:+5,.0f}) positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
