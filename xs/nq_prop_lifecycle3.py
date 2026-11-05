"""Nov 4 2026, POST-HOC (picked after nq_prop_lifecycle2.log): combine the two best single sizers, volume profile THIN and HAR high, in the funded stage."""
_l2 = open('xs/nq_prop_lifecycle2.py').read(); exec(_l2[:_l2.rindex("rows = {")])
both = F.THIN.astype(int) + F.HARH.astype(int)
rows = {'B  0.5 on THIN': summarize(dvA, dv(np.where(F.THIN, .5, 1.0))),
        'V7 0.5 on THIN or HARH': summarize(dvA, dv(np.where(F.THIN | F.HARH, .5, 1.0))),
        'V8 1 / 0.5 / 0.3 by THIN + HARH count': summarize(dvA, dv(np.select([both == 0, both == 1], [1.0, .5], .3)))}
exec(_lc[_lc.index("T = pd.DataFrame(rows).T"):])
