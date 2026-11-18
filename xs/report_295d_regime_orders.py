"""Report 295d: regime explanation test. Within-bar regression (295c) says beta/low-vol picked winners in 2019-22 (stimulus bull) and 20d momentum in 2023-26 (narrow leadership). Declared: pick order by beta HIGH-first and by mom20 HIGH-first in the 6-slot sim; the regime hypothesis predicts beta wins in 2019-22 and mom20 in 2023-26. Not a live rule - a mechanism check."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
_c = open('/home/claude/fartpie/xs/report_295c_within_bar.py').read().split("def fe(")[0].replace('B = B[B.n >= 3].dropna(subset=[\'beta\', \'vol\', \'ldv\', \'mom20\']).copy()', 'B = B.copy()'); exec(_c)
base = B.copy()
exec(open('/home/claude/fartpie/xs/report_295b_flip_rank.py').read().split("def st2")[1].join(["def st2", ""]) if False else "")
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    return ' | '.join(out)
base['h1'] = False; END_ = END
for lab, key in (('ratio (live)', -base.ratio), ('beta high first', -base.beta), ('mom20 high first', -base.mom20), ('low vol first', base.vol), ('ratio + mom20 ranks', base.groupby('entry').ratio.rank(ascending=False) + base.groupby('entry').mom20.rank(ascending=False))):
    T0 = base.assign(_k=key.values).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); print(f'  {lab:22s} {st2(s_)}', flush=True)
