"""Participant-proxy features for vector flips (all causal at the flip-bar close; entry = next bar open). W = bars per trading day, LT = long-term-extreme window in bars.
Participant story each feature encodes (hypotheses fixed before looking at results):
 ABSORPTION (large orders being worked: heavy volume, little price impact): BV, BV3, ABS, CLV
 ACCUMULATION BEFORE THE BREAK (one-sided informed flow): UVS (direction-share of volume, last W bars), VT (volume building into the flip)
 PERSISTENCE STATE (are order-splitters in control): ER, AC1, VR
 DISPLACEMENT / WHO IS TRAPPED: DH (distance from the long-term extreme), PL_SIGN, PL_DUR (previous leg), FLIPS (chop = both sides active)
 REGIME: LV (prior-day range), RISK (ATR%), MKT/DISP (market 20d return, cross-sectional dispersion), TOD."""
import numpy as np, pandas as pd
from vector_pine import run_state_machine
FEATS = ['BV', 'BV3', 'VT', 'IMP', 'CLV', 'ABS', 'UVS', 'ER', 'AC1', 'VR', 'DH', 'PL_SIGN', 'PL_DUR', 'FLIPS', 'LV', 'RISK', 'TOD']
def flips_for(B, W, LT, tk=''):
    B = B.dropna(subset=['open', 'high', 'low', 'close']); B = B[~B.index.duplicated()]
    if len(B) < 8 * W * 20: return []
    reg = run_state_machine(B.assign(datetime=B.index))[0]; n = len(B); o = B.open.values; h = B.high.values; l = B.low.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; day = idx.normalize()
    pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); atr = pd.Series(tr).rolling(W).mean().values; ret1 = np.r_[0, np.diff(np.log(c))]
    dd = pd.DataFrame({'h': B.high.groupby(day).max(), 'l': B.low.groupby(day).min(), 'c': B.close.groupby(day).last()}); rg = (dd.h - dd.l) / dd.c; lv = (rg.shift(1) / rg.shift(1).rolling(20).mean()); LVs = lv.reindex(day).values
    hx = pd.Series(h).rolling(LT, min_periods=LT // 5).max().values; lx = pd.Series(l).rolling(LT, min_periods=LT // 5).min().values
    up = (c > o).astype(float); mins = idx.hour * 60 + idx.minute; ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; chA = np.array(ch); rows = []
    for a in range(1, len(ch) - 1):
        i, j = ch[a], ch[a + 1]; st, en = i + 1, j + 1; d = int(reg[i]); p = ch[a - 1]
        if en >= n or i < 3 * W or np.isnan(LVs[i]) or np.isnan(hx[i]) or np.isnan(atr[i]) or atr[i] <= 0 or i < W * 2: continue
        base = v[i - W:i].mean()
        if base <= 0: continue
        w = slice(i - W + 1, i + 1); vv = v[w]; upv = (vv * (up[w] if d > 0 else 1 - up[w])).sum() / max(vv.sum(), 1); net = abs(c[i] - c[i - W]); path = np.abs(np.diff(c[i - W:i + 1])).sum()
        r = ret1[i - 2 * W + 1:i + 1]; ac1 = np.corrcoef(r[:-1], r[1:])[0, 1] if r.std() > 0 else 0.0
        r4 = np.log(c[i - 2 * W + 1:i + 1][4::4] / c[i - 2 * W + 1:i + 1][:-4][::4]) if False else None
        rr = np.log(c[i - 4 * W + 1:i + 1]); vr1 = np.var(np.diff(rr)); vr = np.var(rr[4:] - rr[:-4]) / (4 * vr1) if vr1 > 0 else 1.0
        clv = (c[i] - l[i]) / max(h[i] - l[i], 1e-12); clv = clv if d > 0 else 1 - clv; rngatr = (h[i] - l[i]) / atr[i]
        rows.append(dict(tk=tk, t=idx[st], d=d, ret=d * (o[en] / o[st] - 1), BV=v[i] / base, BV3=v[i - 2:i + 1].mean() / base, VT=v[i - 5:i].mean() / max(v[i - W:i - 5].mean(), 1), IMP=abs(c[i] - o[i]) / atr[i], CLV=clv,
            ABS=np.log(max(v[i] / base, 1e-3)) - np.log(rngatr + 0.1), UVS=upv, ER=net / path if path > 0 else 0, AC1=ac1, VR=vr, DH=(1 - c[i] / hx[i]) if d > 0 else (c[i] / lx[i] - 1), PL_SIGN=d * (c[i] / c[p] - 1), PL_DUR=np.log1p(i - p),
            FLIPS=((chA > i - 4 * W) & (chA <= i)).sum(), LV=LVs[i], RISK=np.log(atr[i] / c[i]), TOD=mins[st], day=day[st]))
    return rows
