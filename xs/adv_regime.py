"""Nov 10 2026: REGIME SPLIT (bull / bear / chop), NOT A FORECAST. DECLARED BEFORE RESULTS.
Unsupervised, filtered (forward-only) 3-state Gaussian HMM refit every January on all earlier days. Daily features known at 09:30: 5d and 20d close-to-close return (in HAR-sigma units), log HAR range, overnight gap, log VIX. States named by training mean 5d return: BEAR (lowest), CHOP (middle), BULL (highest); also a 2-state version.
Test 1: noise-engine trades (all 3069, 2014-26 evaluated 2018+ walk-forward) points per trade by (regime x side): does LONG pay in BULL and SHORT in BEAR (diagonal) vs off-diagonal?
Test 2: system filter "trade only with the regime" (long in BULL, short in BEAR, both in CHOP) vs unfiltered; pts/day, std, mean/std, both periods.
Test 3: same states on RA and V15a (the engines with a direction) points per trade by regime x side.
PASS: diagonal minus off-diagonal spread same sign, |t|>=2 in both periods."""
import sys, io, contextlib, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; exec(open(f'{SP}/ctxhead.py').read())
from hmmlearn.hmm import GaussianHMM
Dd = pd.DataFrame(index=ix); dc = np.log(Dn.DC.astype(float)); sg = np.log(R).rolling(22).mean().shift(1).pipe(np.exp)
Dd['r5'] = dc.diff(5).shift(1) / (np.sqrt(5) * 0.01); Dd['r20'] = dc.diff(20).shift(1) / (np.sqrt(20) * 0.01)
Dd['lr'] = np.log(R.rolling(5).mean().shift(1)); Dd['gap'] = Dn.gap.astype(float) / Dn.rng20.astype(float).shift(1); Dd['vix'] = np.log(vx.shift(1).reindex(ix, method='ffill')).values
Dd = Dd.replace([np.inf, -np.inf], np.nan).dropna(); Dd = Dd[Dd.index >= '2014-04-01']
def states(K):
    out = pd.Series(np.nan, index=Dd.index)
    for Y in range(2018, 2027):
        tr = Dd.index < f'{Y}-01-01'; te = Dd.index.year == Y
        if te.sum() == 0: continue
        mu, sd = Dd[tr].mean(), Dd[tr].std(); h = GaussianHMM(K, covariance_type='diag', n_iter=100, random_state=2).fit(((Dd[tr] - mu) / sd).values)
        order = np.argsort(h.means_[:, 0]); mp = {k: i for i, k in enumerate(order)}; Xte = ((Dd[te] - mu) / sd).values; alpha = h.startprob_.copy(); fp = []
        for k in range(len(Xte)):
            lik = np.exp(h._compute_log_likelihood(Xte[k:k + 1])[0]); alpha = (alpha @ h.transmat_ if k else alpha) * lik; alpha /= alpha.sum(); fp.append(mp[int(alpha.argmax())])
        out[te] = fp
    return out
G = pd.read_pickle(f'{SP}/G_noise.pkl'); G['day'] = pd.to_datetime(G.day).dt.normalize(); G = G[G.day >= '2018-02-01'].reset_index(drop=True)
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
for K, names in ((3, ['BEAR', 'CHOP', 'BULL']), (2, ['BEAR', 'BULL'])):
    st = states(K); G['st'] = G.day.map(st); g = G.dropna(subset=['st']); yr = g.day.dt.year.values
    print(f'\n##### {K}-state HMM. days per state: ' + ' '.join(f'{names[k]} {int((st == k).sum())}' for k in range(K)))
    print('  noise engine pts per trade by regime x side (18-21 | 22-26):')
    for k in range(K):
        row = []
        for sdir, sn in ((1, 'LONG'), (-1, 'SHORT')):
            for f in (yr < 2022, yr >= 2022):
                pass
            a = [g[(g.st == k) & (g.side == sdir) & (yr < 2022)].pts, g[(g.st == k) & (g.side == sdir) & (yr >= 2022)].pts]; row.append(f'{sn} {a[0].mean():+6.1f} (n{len(a[0])}) | {a[1].mean():+6.1f} (n{len(a[1])})')
        print(f'    {names[k]:<5} ' + '   '.join(row))
    # diagonal test
    bull, bear = K - 1, 0; res = []
    for f in (yr < 2022, yr >= 2022):
        diag = pd.concat([g[f & (g.st == bull) & (g.side == 1)].pts, g[f & (g.st == bear) & (g.side == -1)].pts]); off = pd.concat([g[f & (g.st == bull) & (g.side == -1)].pts, g[f & (g.st == bear) & (g.side == 1)].pts]); res.append((diag.mean() - off.mean(), tt(diag, off), len(diag), len(off)))
    ok = np.sign(res[0][0]) == np.sign(res[1][0]) and all(abs(r[1]) >= 2 for r in res)
    print('  with-regime minus against-regime: ' + ' | '.join(f'{r[0]:+.1f} pts (t {r[1]:+.2f}, n {r[2]}/{r[3]})' for r in res) + ('   <== PASS' if ok else ''))
    keep = (g.st == bull) & (g.side == 1) | (g.st == bear) & (g.side == -1) | ((g.st != bull) & (g.st != bear))
    for lab, kk in (('all trades', pd.Series(True, index=g.index)), ('with-regime only (chop = both)', keep), ('against-regime only', ~keep)):
        o = []
        for nm, f in (('18-21', yr < 2022), ('22-26', yr >= 2022)):
            s = g[f & kk.values].groupby('day').pts.sum().reindex(pd.DatetimeIndex(sorted(g.day[f].unique()))).fillna(0); o.append(f'{nm}: {s.mean():+5.2f}/d t {s.mean() / (s.std() / np.sqrt(len(s))):+5.2f} m/s {s.mean() / s.std():.3f}')
        print(f'    {lab:<32} ' + ' | '.join(o))
    OUT_, NM_ = pickle.load(open(f'{SP}/s6.pkl', 'rb'))
    for (t, u, y_), n in zip(OUT_['V2'], NM_):
        if 'dir' not in t.columns: continue
        t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); t['st'] = t.day.map(st); t = t.dropna(subset=['st']); t = t[t.sz > 0] if 'sz' in t else t; y2 = t.day.dt.year.values; line = []
        for k in range(K):
            for sdir, sn in ((1, 'L'), (-1, 'S')):
                a = [t[(t.st == k) & (t.dir == sdir) & (y2 < 2022)].pts, t[(t.st == k) & (t.dir == sdir) & (y2 >= 2022)].pts]; line.append(f'{names[k][:2]}{sn} {a[0].mean():+5.0f}|{a[1].mean():+5.0f} (n{len(a[0])}/{len(a[1])})')
        print(f'  {n}: ' + '  '.join(line))
