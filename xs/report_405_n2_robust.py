"""Nov 23 2026 report 405: neighbours of N2 (name 15m realized-vol relief). DECLARED before running: windows (short, long) in {(3,20),(5,10),(5,40),(10,20)} plus the original (5,20); robust if >=3 of the 4 neighbours cut MDD AND raise Calmar vs live in BOTH halves."""
import numpy as np
_t = open('xs/report_404_name_vol_relief.py').read(); exec(_t[:_t.rindex("rows = []; TK0, s0")])
TK0, s0 = live_fn(B0); b = score('LIVE', TK0, s0); ok = 0
for sw, lw in ((5, 20), (3, 20), (5, 10), (5, 40), (10, 20)):
    Mdf = RVd.rolling(sw).mean() <= RVd.rolling(lw).mean(); Bf = run_gate_name(Mdf); TKs, s = live_fn(Bf); d = score(f'N2 ({sw}d <= {lw}d mean)', TKs, s); p = d['ca'][2] > b['ca'][2] and d['cb'][2] > b['cb'][2] and d['ca'][1] > b['ca'][1] and d['cb'][1] > b['cb'][1]; ok += (p and (sw, lw) != (5, 20))
    print(line(d) + f" | kept {len(Bf)/len(B0):.0%} | CAGR halves {d['ca'][0]:.1%} | {d['cb'][0]:.1%} -> {'PASS' if p else 'fail'}", flush=True)
print(f'neighbours passing: {ok}/4 (live: CAGR halves {b["ca"][0]:.1%} | {b["cb"][0]:.1%}, MDD halves {b["ca"][1]:.1%} | {b["cb"][1]:.1%})')
