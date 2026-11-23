"""Nov 23 2026 report 386. Slot allocation audit of the live book (b302, 6 slots, 8% breaker, 2+2bp), nothing else changed.
Part A opportunity cost: for every flip the live sim SKIPPED (slots full or breaker), compare its full-vector return with the flip taken; per-name displacement counts; train 2019-21 | test 2022-26.
Part B ranking variants (only the order in which same-timestamp candidates get slots changes): live (brank + 1e-6 ratio), ratio high, ratio low, ldv high (dollar-volume leadership proxy), ldv low, brank only, oldest-name alphabetical, random (40 seeds).
Declared pass for a variant: Sharpe above live in BOTH halves.  Live book is the locked benchmark."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
LIVE = order
def run(lab, ordf, quiet=False):
    global order; order = ordf; TK, _ = sim_swap(B0, None); order = LIVE; z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    r = dict(lab=lab, n=len(TK), cagr=cagr(z), dd=mdd(z), sh=sharpe(z.values), sh_tr=sharpe(a.values), sh_te=sharpe(b.values), cg_tr=cagr(a), cg_te=cagr(b), TK=TK)
    if not quiet: print('%-26s trades %4d CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f' % (lab, r['n'], r['cagr'] * 100, r['dd'] * 100, r['sh'], r['cg_tr'] * 100, r['sh_tr'], r['cg_te'] * 100, r['sh_te']), flush=True)
    return r
base = run('LIVE (locked benchmark)', LIVE)
# ---- Part A
taken = pd.DataFrame(base['TK'], columns=['tk', 'entry', 'exit', 'ret', 'ratio']); key = set(zip(taken.tk, taken.entry))
B0['taken'] = [(t, e) in key for t, e in zip(B0.tk, B0.entry)]; B0['tr'] = B0.entry < '2022-01-01'
print('\n== PART A: taken vs skipped flips (full-vector return, per flip)')
for h, m in (('all', B0.tr | ~B0.tr), ('train', B0.tr), ('test', ~B0.tr)):
    g = B0[m]; t, s = g[g.taken], g[~g.taken]
    print('  %-5s taken n %4d mean %+.2f%% win %.0f%% | skipped n %4d mean %+.2f%% win %.0f%% | gap %+.2fpp' % (h, len(t), t.ret.mean() * 100, (t.ret > 0).mean() * 100, len(s), s.ret.mean() * 100, (s.ret > 0).mean() * 100, (t.ret.mean() - s.ret.mean()) * 100))
# same-day best skipped vs taken
B0['day_'] = B0.entry.dt.normalize(); rows = []
for d, g in B0.groupby('day_'):
    t, s = g[g.taken], g[~g.taken]
    if len(t) and len(s): rows.append((d, t.ret.mean(), s.ret.max(), s.ret.mean(), len(t), len(s)))
Dd = pd.DataFrame(rows, columns=['d', 'taken_mean', 'best_skipped', 'skipped_mean', 'nt', 'ns']); Dd['tr'] = Dd.d < '2022-01-01'
for h, g in (('train', Dd[Dd.tr]), ('test', Dd[~Dd.tr])): print('  days with both a taken and a skipped flip, %s: n %d | mean taken %+.2f%% | mean skipped %+.2f%% | best skipped %+.2f%% | taken beats skipped mean on %.0f%% of days' % (h, len(g), g.taken_mean.mean() * 100, g.skipped_mean.mean() * 100, g.best_skipped.mean() * 100, (g.taken_mean > g.skipped_mean).mean() * 100))
nm = B0.groupby('tk').agg(n=('ret', 'size'), taken=('taken', 'sum'), sk_ret=('ret', lambda s: s[~B0.loc[s.index, 'taken']].mean())); nm['skip_rate'] = 1 - nm.taken / nm.n
print('  most-skipped names (skip rate, n flips):', nm.sort_values('skip_rate', ascending=False).head(8)[['skip_rate', 'n']].round(2).values.tolist(), list(nm.sort_values('skip_rate', ascending=False).head(8).index))
print('  skipped flips with return > +3%%: %d of %d skipped (%.1f%%); taken flips > +3%%: %.1f%%' % ((B0[~B0.taken].ret > .03).sum(), (~B0.taken).sum(), (B0[~B0.taken].ret > .03).mean() * 100, (B0[B0.taken].ret > .03).mean() * 100))
# ---- Part B
print('\n== PART B: ranking variants (identical flips, filters, slots, breaker)')
def mk(col, asc=False, tie=True):
    def f(Bs):
        k = Bs[col].rank(ascending=asc).values if col in Bs else 0
        return Bs.assign(_k=k + (1e-6 * Bs.ratio.rank(ascending=False).values if tie else 0)).sort_values(['entry', '_k', 'tk'], kind='stable')
    return f
res = [run('ratio high first', mk('ratio', False, False)), run('ratio low first', mk('ratio', True, False)), run('ldv high (leadership)', mk('ldv', False)), run('ldv low', mk('ldv', True)), run('brank only', mk('brank', False, False)), run('alphabetical', lambda Bs: Bs.sort_values(['entry', 'tk'], kind='stable'))]
rs = []
for sd in range(40):
    rng = np.random.default_rng(sd); rs.append(run('rand%d' % sd, (lambda Bs, rng=rng: Bs.assign(_k=rng.random(len(Bs))).sort_values(['entry', '_k'], kind='stable')), True))
R = pd.DataFrame(rs); print('%-26s Sharpe median %.2f [5%% %.2f, 95%% %.2f] | train Sh median %.2f | test Sh median %.2f | CAGR median %.1f%%' % ('random order x40', R.sh.median(), R.sh.quantile(.05), R.sh.quantile(.95), R.sh_tr.median(), R.sh_te.median(), R.cagr.median() * 100))
print('live Sharpe percentile among random orders: %.0f%% (train %.0f%%, test %.0f%%)' % ((R.sh < base['sh']).mean() * 100, (R.sh_tr < base['sh_tr']).mean() * 100, (R.sh_te < base['sh_te']).mean() * 100))
for r in res: print('  %-24s beats live in both halves: %s' % (r['lab'], r['sh_tr'] > base['sh_tr'] and r['sh_te'] > base['sh_te']))
