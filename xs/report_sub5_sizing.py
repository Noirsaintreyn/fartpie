"""Does volatility-aware SIZING (not selection) fix the book? Inverse-ATR% weights, w = clip(train median atr_pct / atr_pct, .4, 2.5), applied to each slot (slot = equity/8 * w, i.e. up to 2.5x slot for calm names, 0.4x for wild ones). Cut/medians frozen from TRAIN. Compares ann return, DD and a return/DD ratio on TRAIN and TEST."""
exec(open('xs/report_sub5_discover2.py').read().split("def sim(d):")[0].replace("print(", "(lambda *a, **k: None)("))
med = tr.atr_pct.median()
def simw(d, mode):
    rws = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]
    def close(p):
        nonlocal eq
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq)
    for r in rws:
        st = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: st.append(p)
        op = st
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]:
            w = np.clip(med / r['atr_pct'], .4, 2.5) if mode == 'inv' and r['atr_pct'] == r['atr_pct'] else 1.0; r['sz'] = eq / K * w; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min()
yrs = lambda d: max((d.et.max() - d.et.min()).days / 365, .5)
print(f"{'set':22s}{'sizing':9s} | {'TRAIN ann':>9s}{'DD':>5s}{'ratio':>7s} | {'TEST ann':>8s}{'DD':>5s}{'ratio':>7s}")
for nm, f in (('all flips', lambda d: d.ret == d.ret), ('B vol>=1.5x', lambda d: d.vr >= 1.5), ('B & E+V', lambda d: (d.vr >= 1.5) & ~(d.dte <= 3) & ~(d.vwx > 1))):
    for mode in ('equal', 'inv'):
        o = []
        for d in (tr, te):
            x = d[f(d).fillna(False)]; r, dd = simw(x, mode); an = ((1 + r) ** (1 / yrs(d)) - 1) * 100 if r > -1 else -100; o += [an, dd * 100, an / abs(dd * 100) if dd else np.nan]
        print(f'{nm:22s}{mode:9s} | {o[0]:+8.0f}%{o[1]:5.0f}{o[2]:7.1f} | {o[3]:+7.0f}%{o[4]:5.0f}{o[5]:7.1f}')
