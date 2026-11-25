"""Report 478 (Nov 25 2026): Rey - 'a lot of these weren't bad if you closed at a close below the vector'. On the DAILY high-ATRP group (report 477 daily setup, test 2022+), compare the exit fill: (A) next open after the daily bar that flips the vector bear (the backtest's rule, what the alert says), vs (B) the CLOSE of that bar (a market-on-close order when price is below the vector level near the close). Same entries (next open), cost 30bp. DECLARED BEFORE RESULTS. B is optimistic (the flip is only known at the close; fill assumed AT the close) - the gap between A and B is the most it can be worth. Also 'sell +10% else exit B'."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_467_har_vol_features.py').read(); exec(src[:src.index("# ---- stage 2")].replace("print(f'\\n=== {tgt}", "(lambda *a, **k: None)(f'\\n=== {tgt}").replace("print(f'  {f:10s}", "(lambda *a, **k: None)(f'  {f:10s}"))
def score(feats):
    R = pd.DataFrame(index=O.index)
    for f, s in feats.items():
        x = O[f].astype(float); ref = np.sort(x[tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
O['s_'] = score({'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}); CD = .003
def recon(r):
    df = W[r.tk]; df = df[~df.index.duplicated()]; ix = df.index; j = ix.searchsorted(pd.Timestamp(r.d))
    if j >= len(ix) or ix[j] != pd.Timestamp(r.d) or j + 2 >= len(ix): return None
    e = float(df.open.iloc[j + 1]); tgt = 1 + r.r + .003; ope = df.open.values[j + 1:j + 400] / e; cl = df.close.values[j + 1:j + 400] / e; hi = np.maximum.accumulate(df.high.values[j + 1:j + 400] / e)
    for k in range(1, len(ope)):
        if abs(ope[k] - tgt) < .002 and abs((hi[k - 1] - 1) - r.PEAK) < .002:
            hv = df.high.values[j + 1:j + 1 + k] / e; h10 = np.argmax(hv >= 1.10) if (hv >= 1.10).any() else -1
            return k, h10, cl[k - 1] - 1
    return None
rows = []
for cap in (20, 10, 5):
    for pct, nm in ((.7, 'top30%'), (.9, 'top10%')):
        S = O[(O.px < cap) & (~O.tr)]; cut = np.quantile(O[(O.px < cap) & O.tr].s_.dropna(), pct); G = S[S.s_ >= cut].copy(); out = [recon(r) for r in G.itertuples()]; G['ok'] = [o is not None for o in out]; K_ = G[G.ok].copy()
        K_['hold'] = [o[0] for o in [x for x in out if x is not None]]; K_['t10'] = [o[1] for o in [x for x in out if x is not None]]; K_['rc'] = [o[2] - CD for o in [x for x in out if x is not None]]
        reach = K_.t10 >= 0; alt = np.where(reach, .10 - CD, K_.rc)
        rows.append((f'sub-${cap}', nm, len(K_), (K_.r > 0).mean(), K_.r.mean(), K_.r.median(), (K_.rc > 0).mean(), K_.rc.mean(), K_.rc.median(), K_.r.quantile(.1), K_.rc.quantile(.1), (alt > 0).mean(), alt.mean()))
R = pd.DataFrame(rows, columns=['px', 'group', 'n', 'winA', 'meanA', 'medA', 'winB', 'meanB', 'medB', 'p10_A', 'p10_B', 'win_10B', 'mean_10B']); pd.options.display.width = 250
fm = {c: '{:.0%}'.format for c in ('winA', 'winB', 'win_10B')}; fm.update({c: '{:+.1%}'.format for c in ('meanA', 'medA', 'meanB', 'medB', 'p10_A', 'p10_B', 'mean_10B')}); print(R.to_string(index=False, formatters=fm))
