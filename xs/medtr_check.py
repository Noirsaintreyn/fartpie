"""Nov 4 2026: neighbour check for P3 (jump-robust ATR: causal median of 3 adjacent true ranges, Wilder-smoothed, rescaled to the TR level on the first 5000 bars). DECLARED BEFORE RESULTS.
P3 again (15m vector only) | P5 median of 5 | P3raw no rescale (k = 1) | P3both median-of-3 ATR in the 15m AND 1h vectors. Paper-track candidate only if P3 >= 70% in both halves
and the neighbours >= 60% (the repo's neighbour rule)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h.py').read(); exec(src[:src.index("if __name__ == '__main__':")])
from vector_core import wilder_rma
_ATR0 = compute_atr_series
def tr_(h, l, c): pc = np.r_[c[0], c[:-1]]; return np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
def med_atr(w, rescale=True):
    def f(hh, ll, cc, L):
        tr = tr_(hh, ll, cc); md = pd.Series(tr).rolling(w, min_periods=1).median().values
        k = (np.nanmean(tr[:5000]) / max(np.nanmean(md[:5000]), 1e-12)) if rescale else 1.0
        return wilder_rma(md * k, L)
    return f
if __name__ == '__main__':
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); tks = sorted(Xo.tk.unique()); rows = []
    for tk in tks:
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0]); reg0, s10 = states(H, vec); X5 = H[['dt', 'open', 'high', 'low', 'close']]
        for nm, fn in (('P3', med_atr(3)), ('P5', med_atr(5)), ('P3raw', med_atr(3, False))):
            globals()['compute_atr_series'] = fn; rg = np.asarray(vec(X5)); globals()['compute_atr_series'] = _ATR0
            rows += [(tk, nm, False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run2(H, rg, s10, 'L')]
        globals()['compute_atr_series'] = med_atr(3); rgb, s1b = states(H, vec); globals()['compute_atr_series'] = _ATR0
        rows += [(tk, 'P3both', False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run2(H, rgb, s1b, 'L')]
        rows += [(tk, 'P3bothH1', True, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run(H, rgb, s1b, True)[0]]
        rows += [(tk, 'L', False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run2(H, reg0, s10, 'L')]
        for a, b, r in run(H, reg0, s10, True)[0]: rows.append((tk, 'H1', True, pd.Timestamp(a), pd.Timestamp(b), r))
        print(tk, flush=True)
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet('/tmp/claude-0/ml/medtr_V.parquet')
    e = open('xs/stop_1h_ev.py').read().replace("/tmp/claude-0/ml/stop1h_V.parquet", "/tmp/claude-0/ml/medtr_V.parquet"); exec(e[:e.index("for nm in ('A3'")])
    for nm in ('P3', 'P5', 'P3raw'):
        Xv = build(nm); print(f'  {nm}: trades {len(Xv)}, runners {(Xv[~Xv.h1].ret >= .15).sum()}', flush=True); line(nm, ev(Xv), B0)
    S = pd.concat([V[V['var'] == 'P3both'], V[V['var'] == 'P3bothH1']]); S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].copy(); S['d'] = S.entry.dt.normalize()
    S = pd.merge_asof(S.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)].sort_values('entry').reset_index(drop=True)
    print(f'  P3both: trades {len(S)}', flush=True); line('P3both (15m + 1h vectors)', ev(S), B0)
