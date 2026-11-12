"""Follow-up to report_sub5_discover.py: the rank-IC effects that replicated in BOTH halves (atr_pct, body_atr, rng_atr, dist60h, vr, dil, gap2). Rank IC says what orders winners vs losers, NOT what the mean does (book lives on a right tail), so show mean / median / win / P(>10%) / P(<-8%) by TRAIN-cut terciles, in TRAIN and TEST, then 8-slot books on TEST with cut points and direction frozen from TRAIN."""
import re, sys; src = open('xs/report_sub5_discover.py').read().split("res = []")[0]; exec(src)
FS = ['atr_pct', 'body_atr', 'dist60h', 'vr', 'dil', 'gap2']; cuts = {}
for f in FS:
    q = tr[f].dropna().quantile([1 / 3, 2 / 3]).values; cuts[f] = q; print(f'\n{f}  (train tercile cuts {q.round(3)})'); print(f"   {'':8s}{'period':7s}{'n':>6s}{'mean':>8s}{'median':>8s}{'win':>6s}{'P>10%':>7s}{'P<-8%':>7s}")
    for nm, d in (('train', tr), ('test', te)):
        b = pd.cut(d[f], [-np.inf, q[0], q[1], np.inf], labels=['low', 'mid', 'high'])
        for lab, x in d.groupby(b):
            n = x.net; print(f'   {lab:8s}{nm:7s}{len(x):6d}{n.mean() * 100:+7.2f}%{n.median() * 100:+7.2f}%{(n > 0).mean() * 100:5.0f}%{(n > .1).mean() * 100:6.1f}%{(n < -.08).mean() * 100:6.1f}%')
def sim(d):
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
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min()
def yrs(d): return max((d.et.max() - d.et.min()).days / 365, .5)
EV = lambda d: (d.vr >= 1.5) & ~(d.dte <= 3) & ~(d.vwx > 1)
RULES = {'all flips': lambda d: d.ret == d.ret, 'B vol>=1.5x': lambda d: d.vr >= 1.5, 'B & E+V (hand rule)': EV, 'atr_pct in low-or-mid third': lambda d: d.atr_pct <= cuts['atr_pct'][1], 'atr_pct in low third': lambda d: d.atr_pct <= cuts['atr_pct'][0], 'dist60h not in low third': lambda d: d.dist60h > cuts['dist60h'][0], 'body_atr not in high third': lambda d: d.body_atr <= cuts['body_atr'][1],
         'E+V & atr_pct<=mid': lambda d: EV(d) & (d.atr_pct <= cuts['atr_pct'][1]), 'E+V & atr_pct<=low': lambda d: EV(d) & (d.atr_pct <= cuts['atr_pct'][0]), 'E+V & atr<=mid & dist60h>low': lambda d: EV(d) & (d.atr_pct <= cuts['atr_pct'][1]) & (d.dist60h > cuts['dist60h'][0]), 'all flips & atr<=mid & dist60h>low & body<=mid': lambda d: (d.atr_pct <= cuts['atr_pct'][1]) & (d.dist60h > cuts['dist60h'][0]) & (d.body_atr <= cuts['body_atr'][1])}
print('\n8-slot books, cut points FROZEN from TRAIN (Dec23-Jun25). ann = annualised over the period; TEST = Jul25-Nov26')
print(f"{'rule':50s}{'TRAIN n':>8s}{'ret':>7s}{'DD':>5s} |{'TEST n':>7s}{'ret':>7s}{'ann':>6s}{'DD':>5s}{'mean/tr':>9s}{'win':>5s}")
for nm, f in RULES.items():
    a = tr[f(tr).fillna(False)]; b = te[f(te).fillna(False)]; ra, da = sim(a); rb, db = sim(b); an = ((1 + rb) ** (1 / yrs(te)) - 1) * 100
    print(f'{nm:50s}{len(a):8d}{ra * 100:+6.0f}%{da * 100:5.0f} |{len(b):7d}{rb * 100:+6.0f}%{an:+6.0f}{db * 100:5.0f}{(b.ret - COST).mean() * 100:+8.2f}%{((b.ret - COST) > 0).mean() * 100:5.0f}')
