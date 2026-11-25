"""Report 468 (Nov 24 2026): Rey - exits are manual; what matters is catching the move, HOW LONG until the ~2/3 that reach +10% get there, how much capital the rest tie up, and whether this model is concentrated (by name / sector / month / year).
DECLARED BEFORE RESULTS. Group = top 10% of the base5 rank score (ATRP, F11, VOLX, LTURN, LMC; frozen train-ECDF cuts, report 466/467), px < $10 and px < $20, 2022+ (test) and, for context, 2019-21.
Trade timing is reconstructed from daily OHLCV: entry = next day's open after the flip close; exit = the open that reproduces the stored return r (+30bp cost) and whose pre-exit high reproduces PEAK (match within 0.2%); rows that do not reconcile are dropped and the share is reported.
Outputs: share of the group that has touched +10% within 1/2/3/5/10/20 trading days of entry (of ALL group flips, and of the ones that eventually reach it), median days to reach; hold length (trading days) of those that never reach +10% (median/p75/p90) and the share of all capital-days they use; average hold if you exit manually at +10% (else at the vector exit); concurrency: average and max simultaneous positions per year if every group flip is taken; concentration per year: distinct names, top-5 names' share of flips, top month's share, top SIC group's share."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_467_har_vol_features.py').read(); exec(src[:src.index("# ---- stage 2")].replace("print(f'\\n=== {tgt}", "(lambda *a, **k: None)(f'\\n=== {tgt}").replace("print(f'  {f:10s}", "(lambda *a, **k: None)(f'  {f:10s}"))
def score(feats):
    R = pd.DataFrame(index=O.index)
    for f, s in feats.items():
        x = O[f].astype(float); ref = np.sort(x[tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
O['s_'] = score({'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1})
def recon(r):
    df = W[r.tk]; df = df[~df.index.duplicated()]; ix = df.index; j = ix.searchsorted(pd.Timestamp(r.d))
    if j >= len(ix) or ix[j] != pd.Timestamp(r.d) or j + 2 >= len(ix): return None
    e = float(df.open.iloc[j + 1]); tgt = 1 + r.r + .003; ope = df.open.values[j + 1:j + 400] / e; hi = np.maximum.accumulate(df.high.values[j + 1:j + 400] / e)
    for k in range(1, len(ope)):
        if abs(ope[k] - tgt) < .002 and abs((hi[k - 1] - 1) - r.PEAK) < .002:
            h10 = np.argmax(df.high.values[j + 1:j + 1 + k] / e >= 1.10) if (df.high.values[j + 1:j + 1 + k] / e >= 1.10).any() else -1
            return k, h10, ix[j + 1], ix[j + 1 + k]
    return None
rows = []
for sub, nm in ((lambda D: D[D.px < 10], 'px<10'), (lambda D: D, 'px<20')):
    cut = np.quantile(sub(O[tr]).s_.dropna(), .9); G = sub(O[O.s_ >= cut]).copy(); out = [recon(r) for r in G.itertuples()]
    G['ok'] = [o is not None for o in out]; G['hold'] = [o[0] if o else np.nan for o in out]; G['t10'] = [o[1] if o else np.nan for o in out]; G['ent'] = [o[2] if o else pd.NaT for o in out]; G['ext'] = [o[3] if o else pd.NaT for o in out]
    for half, hl in ((False, '2022+ (test)'), (True, '2019-21 (train)')):
        H = G[(G.tr == half)]; K = H[H.ok]; print(f'\n##### {nm} {hl}: group {len(H)} flips, reconciled {len(K)} ({len(K) / max(len(H), 1):.0%})')
        if len(K) < 30: continue
        reach = K[K.t10 >= 0]; no = K[K.t10 < 0]; d = reach.t10 + 1
        print(f'  reached +10% at some point: {len(reach) / len(K):.0%} | by trading day (of ALL): ' + ' '.join(f'<={x}d {(d <= x).sum() / len(K):.0%}' for x in (1, 2, 3, 5, 10, 20)) + f' | median days to reach {d.median():.0f} (p75 {d.quantile(.75):.0f}, p90 {d.quantile(.9):.0f})')
        print(f'  NEVER reach +10% ({len(no) / len(K):.0%}): hold days median {no.hold.median():.0f} p75 {no.hold.quantile(.75):.0f} p90 {no.hold.quantile(.9):.0f}; their share of all capital-days (vector exit) {no.hold.sum() / K.hold.sum():.0%} | reachers hold at the vector exit: median {reach.hold.median():.0f}d (but only {d.median():.0f}d to the +10%)')
        man = np.where(K.t10 >= 0, K.t10 + 1, K.hold); print(f'  avg hold if you sell at +10% else vector exit: {man.mean():.1f}d vs {K.hold.mean():.1f}d vector-only; capital-days saved {1 - man.sum() / K.hold.sum():.0%}')
        if not half:
            for y, g in K.groupby(K.ent.dt.year):
                days = pd.date_range(g.ent.min(), g.ext.max(), freq='B'); conc = np.array([((g.ent <= t) & (g.ext > t)).sum() for t in days]); y_days = np.array([t.year == y for t in days]); cm = conc[y_days]
                top5 = g.tk.value_counts().head(5).sum() / len(g); tm = g.mon.value_counts().iloc[0] / len(g); sg = g.sec.value_counts(); print(f'  {y}: flips {len(g):3d} names {g.tk.nunique():3d} | top-5 names {top5:.0%} of flips, top month {tm:.0%}, top SIC {sg.index[0]} {sg.iloc[0] / len(g):.0%} | open positions: avg {cm.mean():.1f}, p90 {np.quantile(cm, .9):.0f}, max {cm.max()}')
