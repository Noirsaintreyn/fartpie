"""Nov 3 2026: since Oct 13, SAME live entries (the 39 trades the book took), exits swapped for each rule; overlays: tilt by vol, drop low-vol (slot in cash).
Entries the variant would not have taken (still holding the earlier trade on that ticker) count as cash. Open marked at site last price, else last parquet close."""
import sys, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
s0 = open(HERE + '/stop_1h.py').read(); exec("def run2" + s0.split("def run2")[1].split("if __name__")[0])
lc = open(HERE + '/live_compare.py').read().split("T = pd.DataFrame(rows)")[0]; exec(lc)
T = pd.DataFrame(rows).sort_values('t0').reset_index(drop=True); T['grp'] = np.where(T.rv <= lo, 'low', np.where(T.rv > hi, 'high', 'mid'))
def variant_exit(tk, track, t0, kw):
    H = load(f'{L}/{tk}_15m.parquet'); reg0, s10 = states(H, vec)
    if track == '1h': tr = run(H, reg0, s10, True)[0]
    else: tr = run2(H, reg0, s10, final=True, **kw)
    for a, b, r in tr:
        if pd.Timestamp(a) == t0:
            if b == 'open': xp = last.get((tk, track), float(H.close.values[-1])); return xp / r - 1 - 2e-4, True
            return r, True
    return np.nan, False
RULES = {'current exit': dict(), '1h-confirm no hard stop': dict(mode='A', hard=None), '1h-confirm 5% hard stop': dict(mode='A', hard=.05)}
res = {}
for nm, kw in RULES.items():
    ret = []; miss = 0
    for _, t in T.iterrows():
        if nm == 'current exit': ret.append(t.ret); continue
        if t.track == '1h': ret.append(t.ret); continue
        r, ok = variant_exit(t.tk, t.track, t.t0, kw)
        if ok: ret.append(r)
        else:
            # trade closed in the variant before data end? then it was not found as a closed trade => still open at data end (use last close) or not taken
            ret.append(np.nan); miss += 1
    res[nm] = (np.array(ret, float), miss)
def book(r, w): return np.expm1(np.nansum(np.log1p(np.where(np.isnan(r), 0, r) * w)))
W0 = np.full(len(T), 1 / 6); Wt = np.where(T.grp == 'high', 1.25, np.where(T.grp == 'low', .75, 1)) / 6; Wd = np.where(T.grp == 'low', 0, 1) / 6
print(f'{len(T)} live entries since Oct 13 (site: book {snap["history"]["book"]:+.2f}%  QQQ {snap["history"]["qqq"]:+.2f}%  SPY {snap["history"]["spy"]:+.2f}%)')
for nm, (r, miss) in res.items():
    print(f'  {nm:<26} plain {book(r, W0):+.2%}   size tilt {book(r, Wt):+.2%}   drop low-vol {book(r, Wd):+.2%}   (entries the variant would not have taken, counted as cash: {miss})')
