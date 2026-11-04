"""Nov 4 2026: (A) other vectors as a cross / confirmation / ranker on the live trades, (B) ATR variants from the literature. DECLARED BEFORE RESULTS. 54 names, 40 matched runs, 2019-21 / 2022-26.
A. At each live entry, the state (bull/bear at the last completed bar) of six other vectors, same settings, different bars:
   VB volume bars (K=48, built from 15m) | 30m | 4h | daily | PM premarket-only bars (state at 09:30) | AM 09:30-12:00 bars only (NY morning)
   rank: K = ratio x (1 + w x n_bull), w = .1 / .25 (nothing dropped)    filters: VB bull | daily bull | PM bull | AM bull | n_bull >= 4 | VB cross (VB flipped bull in the last 8 bars)
B. ATR feeding the 15m vector (1h track and everything else unchanged; trade set rebuilt):
   P1 periodicity-filtered ATR (Andersen-Bollerslev / Boudt et al.): TR / f(time-of-day), Wilder-smoothed, times f(now); f = 20-session median TR per 15m slot, normalised to mean 1
   P2 same filter, no re-scaling (one day-average ATR for every bar)       P3 jump-robust: causal median of 3 adjacent TRs (MedRV idea), Wilder-smoothed, rescaled to TR level
PASS = beats live ret/DD in >= 70% of runs in both halves."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h.py').read(); exec(src[:src.index("if __name__ == '__main__':")])
vb = open('xs/vbar_test.py').read(); exec(vb[vb.index('def vbars('):vb.rindex("if __name__ == '__main__':")])
from vector_core import wilder_rma
_ATR0 = compute_atr_series
def tr_(h, l, c): pc = np.r_[c[0], c[:-1]]; return np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
def by_index(n, ends, st):
    out = np.zeros(n); k = -1
    for j in range(n):
        while k + 1 < len(ends) and ends[k + 1] <= j: k += 1
        out[j] = st[k] if k >= 0 else 0
    return out
def sess_vec(H, mask):
    idx = np.where(mask)[0]; S = H.iloc[idx][['open', 'high', 'low', 'close']].reset_index(drop=True)
    return by_index(len(H), idx, np.asarray(vec(S))) if len(idx) > 100 else np.zeros(len(H))
def periodicity(H):
    m = H.m.values; d = H.dt.dt.normalize().values; tr = tr_(H.high.values, H.low.values, H.close.values)
    P = pd.DataFrame({'d': d, 'm': m, 'tr': tr}).pivot_table(index='d', columns='m', values='tr', aggfunc='last')
    F = P.rolling(20, min_periods=10).median().shift(1); F = F.div(F.mean(axis=1), axis=0)
    f = F.stack().rename('f').reset_index(); f.columns = ['d', 'm', 'f']
    return pd.DataFrame({'d': d, 'm': m}).merge(f, on=['d', 'm'], how='left').f.fillna(1.0).clip(.05, 20).values, tr
if __name__ == '__main__':
    e = open('xs/stop_1h_ev.py').read(); exec(e[:e.index("B0 = ev(XL)")])
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); tks = sorted(Xo.tk.unique()); rows = []; feat = []
    for tk in tks:
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0]); reg0, s10 = states(H, vec); n = len(H); m = H.m.values
        # ---- A: other vectors
        V = {'VB': vreg(H, 48)[0]}
        for nm, fr, off in (('30m', '30min', '4h'), ('4h', '4h', '4h')): r = agg(H, fr, off); V[nm] = known(H, r, np.asarray(vec(r)), fr)
        _, D, s = daily_track(H, vec); dd = H.dt.dt.normalize().values; ds = pd.Series(s, index=pd.DatetimeIndex(D.dt).normalize()).shift(1)
        V['daily'] = ds.reindex(pd.DatetimeIndex(dd)).fillna(0).values
        V['PM'] = sess_vec(H, m < 570); V['AM'] = sess_vec(H, (m >= 570) & (m < 720))
        pos = pd.Series(np.arange(n), index=H.dt.values)
        for i, r in XL[XL.tk == tk].iterrows():
            j = pos.get(r.entry)
            if j is None or np.ndim(j) or j < 10: continue
            j = int(j); rec = {k: v[j - 1] for k, v in V.items()}; rec['VBx'] = float(V['VB'][j - 1] == 1 and (V['VB'][j - 9:j - 1] == -1).any()); rec['i'] = i; feat.append(rec)
        # ---- B: ATR variants on the 15m vector
        fp, tr = periodicity(H); h, l, c = H.high.values, H.low.values, H.close.values
        med = pd.Series(tr).rolling(3, min_periods=1).median().values; k3 = np.nanmean(tr[:5000]) / max(np.nanmean(med[:5000]), 1e-12)
        ATRS = {'P1': wilder_rma(tr / fp, 14) * fp, 'P2': wilder_rma(tr / fp, 14), 'P3': wilder_rma(med * k3, 14)}
        for nm, a in ATRS.items():
            globals()['compute_atr_series'] = (lambda a_: (lambda hh, ll, cc, L: a_ if len(cc) == n else _ATR0(hh, ll, cc, L)))(a)
            rg = np.asarray(vec(H[['dt', 'open', 'high', 'low', 'close']])); globals()['compute_atr_series'] = _ATR0
            rows += [(tk, nm, False, pd.Timestamp(a_), pd.Timestamp(b_), r_) for a_, b_, r_ in run2(H, rg, s10, 'L')]
        rows += [(tk, 'L', False, pd.Timestamp(a_), pd.Timestamp(b_), r_) for a_, b_, r_ in run2(H, reg0, s10, 'L')]
        for a_, b_, r_ in run(H, reg0, s10, True)[0]: rows.append((tk, 'H1', True, pd.Timestamp(a_), pd.Timestamp(b_), r_))
        print(tk, flush=True)
    Fz = pd.DataFrame(feat).set_index('i'); Z = XL.join(Fz); Z.to_parquet('/tmp/claude-0/ml/multi_vec_Z.parquet')
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet('/tmp/claude-0/ml/multi_vec_V.parquet')
    ks = ['VB', '30m', '4h', 'daily', 'PM', 'AM']; Z['nb'] = sum((Z[k] == 1).astype(int) for k in ks)
    Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26'); Z['run'] = Z.ret >= .15
    print('\nA. per trade by state at entry (avg / win / runner rate), coverage', f'{Z.VB.notna().mean():.0%}', flush=True)
    for k in ks + ['VBx']:
        out = f'  {k:<6}'
        for p in ('19-21', '22-26'):
            for lab, msk in (('bull', Z[k] == 1), ('bear', Z[k] != 1)):
                z = Z[msk & (Z.per == p) & Z[k].notna()]; out += f' | {p} {lab} n{len(z):>5} {z.ret.mean()*100:+.2f}% w{z.ret.gt(0).mean():.0%} r{z.run.mean()*100:.1f}%'
        print(out, flush=True)
    for p in ('19-21', '22-26'):
        z = Z[Z.per == p]; print(f'  n_bull {p}:', ' '.join(f"{k}:{z[z.nb == k].ret.mean()*100:+.2f}%(n{(z.nb == k).sum()})" for k in range(7)), flush=True)
    B0 = ev(XL); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
    def K(nm, k): Y = XL.copy(); Y['ratio'] = k; line(nm, ev(Y), B0)
    def F(nm, keep): Y = XL[keep.fillna(True).values].reset_index(drop=True); line(nm + f' (drops {(~keep.fillna(True)).mean():.0%})', ev(Y), B0)
    K('rank ratio x (1 + .1 n_bull)', XL.ratio * (1 + .1 * Z.nb.values)); K('rank ratio x (1 + .25 n_bull)', XL.ratio * (1 + .25 * Z.nb.values))
    for k in ('VB', 'daily', 'PM', 'AM'): F(f'filter {k} bull', Z[k] == 1)
    F('filter n_bull >= 4', Z.nb >= 4); F('filter VB cross (flipped bull <= 8 bars ago)', Z.VBx == 1)
    e2 = open('xs/stop_1h_ev.py').read().replace("/tmp/claude-0/ml/stop1h_V.parquet", "/tmp/claude-0/ml/multi_vec_V.parquet"); exec(e2[e2.index("V = pd.read_parquet"):e2.index("def stat2")])
    print('\nB. ATR variants (15m vector)', flush=True)
    for nm in ('P1', 'P2', 'P3'):
        Xv = build(nm); h_ = Xv[~Xv.h1]; print(f'  {nm}: trades {len(Xv)}, 15m avg {h_.ret.mean():+.3%} (live {XL[~XL.h1].ret.mean():+.3%}), runners {(h_.ret >= .15).sum()} (live {(XL[~XL.h1].ret >= .15).sum()})', flush=True)
        line(nm, ev(Xv), B0)
