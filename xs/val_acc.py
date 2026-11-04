"""Nov 4 2026: is a flip VALID (accepting higher value) or noise? Entry unchanged (no delay); only a classifier known at entry. DECLARED BEFORE RESULTS.
Volume profile per RTH session (15m bars, typical price, 70% value area around the POC): VAL / POC / VAH.
 vsVAH    entry price vs the prior session's value area: above VAH / inside / below VAL
 mig      value migration: prior session POC vs the one before it, in ATR-ish (prior-day range) units: higher (> +0.25) / flat / lower (< -0.25)
 hi_ov    prior value area fully above the one before it (VAL_prev > VAH_prev2) = clearly higher value
 acc      share of the last 26 bars' volume (about one RTH day) that traded with close above the prior VAH  (time/volume accepted above value) tercile
 flipacc  combo: entry above prior VAH AND value migrating higher
Cells reported 19-21 || 22-26 like ctx_rank; a cell holds if on the same side of the period mean by >= .25pt in both with n >= 120.  Then book test: drop non-accepted entries (matched runs vs live)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
def va(g):
    tp = ((g.high + g.low + g.close) / 3).values; v = g.volume.values.astype(float); lo, hi = tp.min(), tp.max()
    if hi <= lo or v.sum() <= 0: return None
    bins = np.linspace(lo, hi, 25); idx = np.clip(np.digitize(tp, bins) - 1, 0, 23); prof = np.bincount(idx, weights=v, minlength=24)
    p = int(prof.argmax()); a = b = p; tot = prof[p]
    while tot < .7 * prof.sum() and (a > 0 or b < 23):
        l = prof[a - 1] if a > 0 else -1; r = prof[b + 1] if b < 23 else -1
        if l >= r: a -= 1; tot += prof[a]
        else: b += 1; tot += prof[b]
    c = (bins[:-1] + bins[1:]) / 2; return c[a], c[p], c[b], hi - lo
rows = []
for tk, G in XL.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]).reset_index(drop=True); H['d'] = H.dt.dt.normalize(); R = H[(H.m >= 570) & (H.m < 960)]
    S = {d: va(g) for d, g in R.groupby('d')}; days = sorted(k for k, v in S.items() if v is not None)
    dix = pd.Series(range(len(days)), index=days); cl = H.close.values; vo = H.volume.values.astype(float); pos = pd.Series(np.arange(len(H)), index=H.dt.values)
    for i, r in G.iterrows():
        j = pos.get(r.entry)
        if j is None or np.ndim(j): continue
        k = np.searchsorted(days, r.entry.normalize(), side='left') - 1   # last session strictly before the entry day
        if k < 1: continue
        d1, d2 = S[days[k]], S[days[k - 1]]; px = H.open.values[j]; w = slice(max(0, j - 26), j)
        rows.append(dict(i=i, vsVAH='above VAH' if px > d1[2] else ('below VAL' if px < d1[0] else 'inside VA'), mig_v=(d1[1] - d2[1]) / max(d1[3], 1e-9),
                         hi_ov=d1[0] > d2[2], lo_ov=d1[2] < d2[0], acc=(vo[w] * (cl[w] > d1[2])).sum() / max(vo[w].sum(), 1)))
F = pd.DataFrame(rows).set_index('i'); Z = XL.join(F, how='inner')
Z['mig'] = pd.cut(Z.mig_v, [-9, -.25, .25, 9], labels=['value lower', 'flat', 'value higher']); Z['acct'] = pd.qcut(Z.acc, 3, labels=['low acc', 'mid', 'high acc'], duplicates='drop')
Z['flipacc'] = np.where((Z.vsVAH == 'above VAH') & (Z.mig == 'value higher'), 'above VAH + value higher', 'other'); Z['hiov'] = np.where(Z.hi_ov, 'value clearly higher', np.where(Z.lo_ov, 'value clearly lower', 'overlapping'))
Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26'); Z['big'] = Z.ret >= .15; mu = {p: Z[Z.per == p].ret.mean() for p in ('19-21', '22-26')}
print('trades', len(Z), 'of', len(XL), {p: f'{v*100:+.3f}%' for p, v in mu.items()}); HOLD = []
for col in ('vsVAH', 'mig', 'hiov', 'acct', 'flipacc'):
    print(f'\n{col}   (avg trade | win | >=15% share | n)   19-21 || 22-26   * holds')
    for v, g in Z.groupby(col, observed=True):
        out, ds = [], []
        for p in ('19-21', '22-26'):
            z = g[g.per == p]; out.append(f'{z.ret.mean()*100:+.2f}% | {z.ret.gt(0).mean():.0%} | {z.big.mean():.1%} | n{len(z)}'); ds.append((z.ret.mean() - mu[p]) * 100 if len(z) >= 120 else np.nan)
        ok = all(not np.isnan(d) and abs(d) >= .25 for d in ds) and np.sign(ds[0]) == np.sign(ds[1]); HOLD.append((col, v, ds)) if ok else None
        print(f'  {str(v):<26}{out[0]:<32}||  {out[1]:<32}{"*" if ok else ""}')
print('\nHOLD:', [(c, str(v), [round(x, 2) for x in d]) for c, v, d in HOLD])
Z.to_parquet('/tmp/claude-0/ml/val_acc_Z.parquet')
B0 = ev(XL); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
def drop(nm, bad):
    Y = XL[~XL.index.isin(Z[bad(Z)].index)].reset_index(drop=True); line(nm, ev(Y), B0); print(f'     kept {len(Y)}/{len(XL)}; big kept {(Y.ret >= .15).sum()}/{(XL.ret >= .15).sum()}')
drop('drop below prior VA', lambda z: z.vsVAH == 'below VAL')
drop('drop value lower', lambda z: z.mig == 'value lower')
drop('drop not (above VAH or inside+higher)', lambda z: (z.vsVAH != 'above VAH') & (z.mig != 'value higher'))
drop('keep only above VAH + value higher', lambda z: z.flipacc == 'other')
