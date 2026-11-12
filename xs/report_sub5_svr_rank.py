"""Use svr20 (persistent short-sale share) WITHOUT cutting trades: (a) slot ORDER: when several flips compete for free slots, take low-svr20 first (default order is higher flip volume first); (b) SIZE: high-svr20 trades (train top third) get 0.5x slot, others 1x. Both on E+V and on B. Cuts frozen from TRAIN."""
exec(open('xs/report_sub5_shortvol.py').read().split("FS = ['svr_dev'")[0])
EV = lambda d: (d.vr >= 1.5) & ~(d.dte <= 3) & ~(d.vwx > 1); yrs = lambda d: max((d.et.max() - d.et.min()).days / 365, .5)
cut = tr.svr20.dropna().quantile(2 / 3)
def simx(d, order, size):
    key = {'vol': lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0)), 'svr': lambda r: (r['et'], r['svr20'] if r['svr20'] == r['svr20'] else .5)}[order]
    rws = sorted(d.to_dict('records'), key=key); eq = 1.0; op = []; cv = [1.0]
    def close(p):
        nonlocal eq
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq)
    for r in rws:
        st = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: st.append(p)
        op = st
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K * (.5 if size and r['svr20'] > cut else 1.0); op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min()
an = lambda r, d: ((1 + r) ** (1 / yrs(d)) - 1) * 100 if r > -1 else -100
print(f"{'set':8s}{'variant':34s}| {'TRAIN ann':>9s}{'DD':>5s} | {'TEST ann':>8s}{'DD':>5s}")
for nm, f in (('E+V', EV), ('B', lambda d: d.vr >= 1.5)):
    for lab, order, size in (('baseline (vol order, equal size)', 'vol', False), ('order: low svr20 first', 'svr', False), ('size: high svr20 half size', 'vol', True), ('order low svr20 + half size high', 'svr', True)):
        o = []
        for d in (tr, te): x = d[f(d).fillna(False)]; r, dd = simx(x, order, size); o += [an(r, d), dd * 100]
        print(f'{nm:8s}{lab:34s}| {o[0]:+8.0f}%{o[1]:5.0f} | {o[2]:+7.0f}%{o[3]:5.0f}')
