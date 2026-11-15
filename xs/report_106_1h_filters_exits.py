"""Nov 15 2026 FILTERS AND EXITS FOR 1H FLIPS (Rey: 'look for models to improve or filter 1h'). Declared before results. Same 1H flip set as report_105 (entry at the 1H close of the flip bar, 100bp round trip, 60 days Aug 31-Nov 12 2026, funds excluded, $1-10, dollar volume >= $500k). ONE regime, 3,436 flips, survivorship-flattered -> leads only. SPLIT inside the window: SELECT on first half (flip date < Oct 6), JUDGE on second half (>= Oct 6); nothing is chosen on the second half. PART 1 hypothesis-first filters with the sign declared in advance (a pass needs the declared sign on BOTH halves): F1 daily vector bull at prior close (+: multi-timeframe agreement); F2 chop = number of 1H bull flips of the same name in the prior 70 bars (-: whipsaw names lose); F3 pre-flip run = close/close 20 bars earlier - 1 (-: late entries give it back); F4 risk to ratchet = (close - level)/close at the flip bar (-: tight ratchet = small stop distance and cheap failure); F5 flip bar close location in its range (+: strong close); F6 daily breadth = share of universe in daily bull state the prior day (+); F7 flip in the first hour of session (declared 0 sign, reported only). Report Spearman IC to net return, and mean net of the good half vs the bad half of each filter by split at the TRAIN median. PART 2 ridge logistic (target net>0, rank features, C=0.5) fit on first half, scored on second half: IC, top-quintile vs rest on the test half. PART 3 exits (instead of waiting for the 1H bear flip): time stop at 7/14/35 bars or earlier bear flip; trailing close stop 5%/10% from the peak close; hard stop at -4%/-6% from entry (fills at the stop price); all with the bear flip as the backstop. Report mean/median/win/>=10% net, plus first vs second half. Lead bar for any filter or exit: improves mean net on BOTH halves and the second-half day-clustered t of (filtered minus all) >= 2."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb')); DD = pickle.load(open(SC + 'lowpx_daily.pkl', 'rb')); sic = pd.read_csv(SC + 'ext/sic.csv').drop_duplicates('tk').set_index('tk'); COST = .01; SPLIT = pd.Timestamp('2026-10-06')
def cl(x):
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York') if x.index.tz is not None else x.index; x = x[~x.index.duplicated()]; m = x.index.hour * 60 + x.index.minute; return x[(m >= 570) & (m < 960)]
rows = []; dreg = {}
for tk, x in M5.items():
    if sic.sic.get(tk, 0) // 100 == 67: continue
    q = DD.get(tk)
    if q is None or len(q) < 200: continue
    q = q[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); q.columns = ['open', 'high', 'low', 'close', 'volume']; q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize()
    c = q.close.values; v = q.volume.values.astype(float); dv = pd.Series(c * v).rolling(20).median().values; r_, lv_, at_, _ = run_state_machine(q.assign(datetime=q.index)); Dreg = pd.Series(r_, index=q.index).shift(1); Dage = pd.Series(r_, index=q.index)
    dreg[tk] = Dreg; DF = pd.DataFrame(dict(cc=c, dvv=dv), index=q.index).shift(1)
    x = cl(x); x = x[x.volume > 0]
    if len(x) < 300: continue
    B = x.resample('60min', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open']); B = B[B.index.hour * 60 + B.index.minute >= 570]
    if len(B) < 150: continue
    reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); cc = B.close.values; hi = B.high.values; lo = B.low.values; n = len(B); dts = pd.to_datetime(B.index.date); isf = np.array([(reg[k] == 1 and reg[k - 1] != 1) for k in range(1, n)]); isf = np.r_[False, isf]; cumf = np.cumsum(isf)
    for i in range(21, n - 1):
        if not isf[i]: continue
        d = dts[i]
        if d not in DF.index: continue
        f = DF.loc[d]
        if not (1 <= f.cc < 10 and f.dvv >= 5e5): continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        e = cc[i]; end = min(j, n - 1); closes = cc[i + 1:end + 1]; lows = lo[i + 1:end + 1]; highs = hi[i + 1:end + 1]
        def ex_time(K):
            k = min(K, len(closes)); return closes[k - 1]
        def ex_trail(x_):
            pk = e
            for k in range(len(closes)):
                if closes[k] <= pk * (1 - x_): return closes[k]
                pk = max(pk, closes[k])
            return closes[-1]
        def ex_hard(x_):
            st = e * (1 - x_)
            for k in range(len(closes)):
                if lows[k] <= st: return st
            return closes[-1]
        rng = hi[i] - lo[i]
        rows.append(dict(tk=tk, t=B.index[i], d=d, entry=e, ex_base=closes[-1], ex_t7=ex_time(7), ex_t14=ex_time(14), ex_t35=ex_time(35), ex_tr5=ex_trail(.05), ex_tr10=ex_trail(.10), ex_h4=ex_hard(.04), ex_h6=ex_hard(.06), dbull=float(Dreg.get(d, np.nan) == 1), chop=float(cumf[i] - cumf[max(0, i - 70)] - 1), pre=cc[i] / cc[i - 20] - 1, risk=(e - lev[i]) / e if lev[i] == lev[i] else np.nan, clv=(cc[i] - lo[i]) / rng if rng > 0 else .5, first=float(B.index[i].hour * 60 + B.index[i].minute == 570)))
A = pd.DataFrame(rows).dropna(); print('flips', len(A), 'names', A.tk.nunique())
br = pd.DataFrame({k: v for k, v in dreg.items()}); breadth = (br == 1).sum(axis=1) / br.notna().sum(axis=1).clip(lower=1); A['breadth'] = A.d.map(breadth)
for k in ('base', 't7', 't14', 't35', 'tr5', 'tr10', 'h4', 'h6'): A['r_' + k] = A['ex_' + k] / A.entry - 1 - COST
A['r'] = A.r_base; A['half'] = np.where(A.d < SPLIT, 'train', 'test'); tr = A[A.half == 'train']; te = A[A.half == 'test']; print('train', len(tr), 'test', len(te))
def clus(z, mask, col='r'):
    a = z.assign(day=z.d.astype(str)); m = a[mask.loc[a.index]].groupby('day')[col].mean(); allm = a.groupby('day')[col].mean(); dd_ = (m - allm.reindex(m.index)).dropna(); return dd_.mean() * 100, (dd_.mean() / (dd_.std() / np.sqrt(len(dd_))) if len(dd_) > 5 and dd_.std() > 0 else np.nan)
print('\nPART 1 single filters (declared sign), IC to net return and good-half vs bad-half (split at TRAIN median), mean net')
decl = dict(dbull='+', chop='-', pre='-', risk='-', clv='+', breadth='+', first='0'); good = {}
for f, sg in decl.items():
    s_tr = spearmanr(tr[f], tr.r)[0]; s_te = spearmanr(te[f], te.r)[0]; med = tr[f].median(); gi = (lambda z: z[f] > med) if sg in '+0' else (lambda z: z[f] <= med)
    if f in ('dbull', 'first'): gi = (lambda z, f=f: z[f] == 1) if sg != '-' else (lambda z, f=f: z[f] == 0)
    a, b = tr[gi(tr)], tr[~gi(tr)]; c_, d_ = te[gi(te)], te[~gi(te)]; ok = (np.sign(s_tr) == (1 if sg == '+' else -1)) and (np.sign(s_te) == (1 if sg == '+' else -1)) if sg != '0' else None; good[f] = gi
    m, t = clus(te, gi(te)); print(f' {f:8s} declared {sg}  IC train {s_tr:+.3f} test {s_te:+.3f} sign-ok {ok} | TRAIN good n {len(a)} {a.r.mean()*100:+5.2f}% vs bad n {len(b)} {b.r.mean()*100:+5.2f}% | TEST good n {len(c_)} {c_.r.mean()*100:+5.2f}% vs bad n {len(d_)} {d_.r.mean()*100:+5.2f}% | test good minus all {m:+.2f}pp t {t:+.2f}')
print('\nPART 1b combo declared: dbull & chop<=train median & pre<=train median'); cm = lambda z: (z.dbull == 1) & (z.chop <= tr.chop.median()) & (z.pre <= tr.pre.median())
for lab, z in (('train', tr), ('test', te)): g = z[cm(z)]; print(f' {lab} n {len(g)} mean {g.r.mean()*100:+.2f}% med {g.r.median()*100:+.2f}% win {(g.r>0).mean()*100:.1f}% >=10% {(g.r>=.1).mean()*100:.1f}% | rest mean {z[~cm(z)].r.mean()*100:+.2f}%')
m, t = clus(te, cm(te)); print(f' test combo minus all {m:+.2f}pp t {t:+.2f}')
print('\nPART 2 ridge logistic (target net>0) fit on train, scored on test')
fs = ['dbull', 'chop', 'pre', 'risk', 'clv', 'breadth', 'first']; Xt = tr[fs].rank(pct=True); Xe = te[fs].rank(pct=True); lr = LogisticRegression(C=.5, max_iter=500).fit(Xt, tr.r > 0); print(' coefs', dict(zip(fs, lr.coef_[0].round(2)))); st_ = lr.decision_function(Xe); te = te.assign(sc=st_); ic = spearmanr(te.sc, te.r); print(f' test IC {ic[0]:+.3f} (p {ic[1]:.3f})'); te['q'] = pd.qcut(te.sc, 5, labels=False)
for k in range(5): z = te[te.q == k]; print(f'  test quintile {k+1}: n {len(z)} mean {z.r.mean()*100:+5.2f}% med {z.r.median()*100:+5.2f}% win {(z.r>0).mean()*100:4.1f}% >=10% {(z.r>=.1).mean()*100:4.1f}%')
m, t = clus(te, te.q == 4); print(f' top quintile minus all (test, day-clustered) {m:+.2f}pp t {t:+.2f}')
print('\nPART 3 exits (net of 100bp, entry at 1H close of flip bar)')
for k in ('base', 't7', 't14', 't35', 'tr5', 'tr10', 'h4', 'h6'):
    col = 'r_' + k; a = A[col]; print(f' {k:5s} ALL mean {a.mean()*100:+6.2f}% med {a.median()*100:+6.2f}% win {(a>0).mean()*100:4.1f}% >=10% {(a>=.1).mean()*100:4.1f}% | train {tr[col].mean()*100:+6.2f}% test {te[col].mean()*100:+6.2f}% | worst 1% {a.quantile(.01)*100:+6.1f}%')
A.to_pickle(SC + 'ext/flips_1h_filters.pkl')
