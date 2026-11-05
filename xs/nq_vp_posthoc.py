"""Nov 4 2026, POST-HOC (chosen after reading nq_vp_nodes.log): the one label that held in all three periods on the monthly and 60-day profiles was
'LVN path ahead' = WORSE (the opposite of the leave-value-through-thin-profile idea). Realistic months 2022 - Oct 2026 for skipping or half-sizing those trades."""
_s = open('xs/nq_vp_nodes.py').read(); exec(_s[:_s.index("R = run_l(C); Z = ")])
R = run_l(C)
def mline(nm, dv):
    M_ = months(dv); M6 = M_[M_.month >= '2026-01']
    print(f'  {nm:<52} passed {M_.res.eq("passed").mean():4.0%} lost {M_.res.eq("lost").mean():4.0%} neither {M_.res.eq("neither").mean():4.0%} | month $ mean {M_.pnl.mean():+6,.0f} worst {M_.pnl.min():+7,.0f} | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}', flush=True)
mline('base', (R.net * 20).groupby(R.day).sum())
lvn = {'M': C.AHEAD_M == 'LVN path ahead', '60D': C['AHEAD_60D'] == 'LVN path ahead'}; lvn['M or 60D'] = lvn['M'] | lvn['60D']; lvn['M and 60D'] = lvn['M'] & lvn['60D']
for nm, m in lvn.items():
    Rr = run_l(C[~m]); post = Rr[Rr.day >= P0]; mline(f'skip LVN path ahead ({nm}) (n{len(post)}, avg {post.net.mean():+.1f})', (Rr.net * 20).groupby(Rr.day).sum())
    w = np.where(m.loc[R.index], 10, 20); mline(f'0.5 NQ when LVN path ahead ({nm}), else 1 NQ', (R.net * w).groupby(R.day).sum())
