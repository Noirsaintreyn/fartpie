"""Nov 12 2026: do entries that go red at once explain the bad trades? DECLARED BEFORE RESULTS. Rey: most of the last 10 trades were wrong from the first bar.
Sets: K&E1 and K&E1&E4 on $1-2, vol>=1.5x flips (bands_T.pkl), 8-slot book, 40 bp, train < 2025-07-01 | test after.
Step 1: share of trades whose FIRST bar closes below the entry open, and mean final return by that first-bar sign (and by MAE in the first 3 bars).
Step 2 variants (all OHLCV, all causal): CONF-H: enter at the next open only if the entry bar closed above the flip bar's HIGH (follow-through) | CONF-G: only if the entry bar closed green (close > open) |
 FAIL k: exit at the open after bar k if bar k's close is still below the entry open (k = 1, 2, 3) | FAIL-3%: exit at the open after any of the first 3 bars closing >= 3% below entry. Everything else = baseline (exit at the vector's bear flip)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004
T0 = pd.read_pickle(SC + 'bands_T.pkl'); T0 = T0[(T0.sig >= 1) & (T0.sig < 2)]
src = open('xs/report_12_bands.py').read(); exec(src[src.index('def book(Z, cost'):src.index("rng = np.random")])
def feats(Z):
    rows = []
    for tk, g in Z.groupby('tk'):
        Y = M.closed_only(data[tk].dropna(subset=['open', 'close'])); o, h, l, c = Y.open.values, Y.high.values, Y.low.values, Y.close.values; ix = Y.index.values
        for r in g.to_dict('records'):
            ei = int(np.searchsorted(ix, np.datetime64(r['et']))); xi = int(np.searchsorted(ix, np.datetime64(r['xt']))); fi = ei - 1; e = o[ei]; d = dict(r); d['base'] = o[xi] / e - 1; d['dur'] = xi - ei
            d['first_red'] = c[ei] < e; d['mae3'] = min(l[ei:min(ei + 3, xi)].min() / e - 1, 0) if xi > ei else 0
            for k in (1, 2, 3):
                d[f'F{k}'] = (o[ei + k] / e - 1) if (ei + k < xi and c[ei + k - 1] < e) else d['base']; d[f'F{k}_t'] = bool(ei + k < xi and c[ei + k - 1] < e)
            ff = None
            for k in range(1, 4):
                if ei + k < xi and c[ei + k - 1] <= e * .97: ff = o[ei + k] / e - 1; break
            d['F3pct'] = ff if ff is not None else d['base']; d['F3pct_t'] = ff is not None
            d['ch'] = (o[xi] / o[ei + 1] - 1) if (xi > ei + 1 and c[ei] > h[fi]) else np.nan; d['cg'] = (o[xi] / o[ei + 1] - 1) if (xi > ei + 1 and c[ei] > o[ei]) else np.nan
            rows.append(d)
    return pd.DataFrame(rows)
rng = np.random.default_rng(9)
def rep(nm, Z, col, trig=None):
    Z = Z[Z[col].notna()].copy(); Z['ret'] = Z[col]
    if len(Z) < 12: print(f'  {nm:<24} {len(Z)} trades'); return
    r = Z.ret - COST; a, b = r[Z.et < '2025-07-01'], r[Z.et >= '2025-07-01']; s = book(Z, COST); yrs = (s.index[-1] - Z.et.min()).days / 365.25; cg = s.iloc[-1] ** (1 / max(yrs, .5)) - 1
    print(f'  {nm:<24} {len(Z):>4} tr | mean {r.mean()*100:+.2f}% med {r.median()*100:+.1f}% win {(r>0).mean():.0%} p10 {r.quantile(.1)*100:+.1f}% | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cg:+.0%}/yr DD {(s/s.cummax()-1).min():.0%}' + (f' | fired {trig:.0%}' if trig is not None else ''), flush=True)
for nm, m in (('K&E1', T0.kal & T0.a200), ('K&E1&E4', T0.kal & T0.a200 & T0.a4)):
    F = feats(T0[m]); print(f'== {nm}: {len(F)} trades')
    print(f'  first bar closes BELOW the entry open: {F.first_red.mean():.0%} of trades | mean final net: first bar red {(F[F.first_red].base-COST).mean()*100:+.2f}% (win {((F[F.first_red].base-COST)>0).mean():.0%}), first bar green {(F[~F.first_red].base-COST).mean()*100:+.2f}% (win {((F[~F.first_red].base-COST)>0).mean():.0%})')
    print(f'  median trade length {F.dur.median():.0f} bars; mean worst dip in first 3 bars {F.mae3.mean()*100:.1f}%; winners (net>0): worst dip {F[F.base-COST>0].mae3.mean()*100:.1f}% vs losers {F[F.base-COST<=0].mae3.mean()*100:.1f}%')
    rep('baseline', F, 'base'); rep('CONF-H (close > flip high)', F, 'ch'); rep('CONF-G (entry bar green)', F, 'cg')
    for k in (1, 2, 3): rep(f'FAIL {k} (exit if red after bar {k})', F, f'F{k}', F[f'F{k}_t'].mean())
    rep('FAIL -3% in first 3 bars', F, 'F3pct', F['F3pct_t'].mean())
