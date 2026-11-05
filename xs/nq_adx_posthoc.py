"""Nov 5 2026, POST-HOC (found in nq_context_guards: 'ADX < 25 and rising' was the best label in all four periods, 'ADX >= 25 and falling' the worst).
Shapes chosen AFTER seeing that table, so this is an upper bound, not a test: ADX-UP x1.5 on 'no trend rising' | ADX-TF skip 'trend falling' | ADX-BOTH both |
NEWSUP+SHADOW the two declared variants that did a little better, combined. Lucid 50K."""
_cg = open('xs/nq_context_guards.py').read(); exec(_cg[:_cg.rindex("if __name__ == '__main__':")])
up = pd.Series(np.where(ADX == 'no trend rising', 1.5, 1.0), index=XA.index); tf = pd.Series(np.where(ADX == 'trend falling', 0.0, 1.0), index=XA.index)
VAR = {'BASE': VAR['BASE']}
for nm, fm, mode in (('ADX-UP', up, 'step'), ('ADX-TF', tf, 'step'), ('ADX-BOTH', up * tf, 'step'), ('NEWSUP+SHADOW', FM['NEWS-UP'], 'shadow'), ('ADX-BOTH+SHADOW', up * tf, 'shadow')):
    DS[f'e_{nm}'] = to_D(ev_res, eqm(.3, mode) * fm); DS[f'f_{nm}'] = to_D(full, eqm(.7, mode) * fm); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
if __name__ == '__main__':
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, [(nm, 'LUCID 50K') for nm in VAR])
    print('\nPOST-HOC ADX / COMBOS (Lucid 50K)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<16} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
