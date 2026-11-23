"""Nov 23 2026 report 395. Rey: no time-based rules; identify RANGES structurally, find a sweep of one side with a close back inside, and if the vector flips after that we should be more confident.
Range = the last confirmed swing low L0 and swing high H0 (causal).  Two scales, pre-registered: (A) zig-zag / persistence pivots confirmed at a 2 x ATR retracement (scale-free), (B) the vector's own 8-bar pivots (pivothigh/low len 8, confirmed 8 bars late) - the same pivots the vector ratchets on.  A range needs H0 - L0 >= 2 ATR.
SWEEP-AND-RECLAIM of the low side: a bar trades below L0 (low < L0) and a close back above L0 occurs on that bar or within the next 2 bars; the event bar is the reclaim bar; depth = (L0 - lowest low)/ATR.  L0 must have been confirmed before the sweep.  The test is whether a 15m bull flip whose flip bar comes within W = 20 (primary) or 40 bars after such an event is better (mechanism: the sweep trapped late sellers below the range low, so the flip that follows is confirmation of a failed break-down, with stops under the sweep low as supply).  Would stop working when sweeps are routine noise (high share of flips qualify so the flag carries no information).
Features per flip: SW20, SW40 (event within W bars), depth of the sweep (terciles among SW20), and IN = the flip bar closes inside (L0, H0) vs above H0 (already broke the range).  No clock rules anywhere.
Tests: yes-minus-no mean flip return, week-clustered OLS t, train 2019-21 | test 2022-26; admit = same sign both halves, |t|>=2 both.  Then slot priority: sweep-first order in the live book sim (only the pick order changes)."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
import vector_pine as vp, report_295_flip_quality as q
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b = b[b.entry >= '2019-01-01'].reset_index(drop=True)
def swings_zz(hi, lo, thr):
    n = len(hi); L0 = np.full(n, np.nan); H0 = L0.copy(); mh, ml = hi[0], lo[0]; cl, ch = np.nan, np.nan
    for i in range(1, n):
        t = thr[i]
        if hi[i] > mh: mh = hi[i]
        elif not np.isnan(t) and mh - hi[i] >= t: ch = mh; mh = hi[i]
        if lo[i] < ml: ml = lo[i]
        elif not np.isnan(t) and lo[i] - ml >= t: cl = ml; ml = lo[i]
        L0[i], H0[i] = cl, ch
    return L0, H0
def swings_vec(hi, lo):
    pl = vp._piv(lo, 8, False); ph = vp._piv(hi, 8, True); n = len(hi); L0 = np.full(n, np.nan); H0 = L0.copy(); cl, ch = np.nan, np.nan
    for i in range(n):
        if not np.isnan(pl[i]): cl = pl[i]
        if not np.isnan(ph[i]): ch = ph[i]
        L0[i], H0[i] = cl, ch
    return L0, H0
def events(L0, H0, lo, c, a):
    n = len(c); ev = np.zeros(n); depth = np.zeros(n); pend = None
    for i in range(1, n):
        L = L0[i - 1]; Hh = H0[i - 1]
        if np.isnan(L) or np.isnan(Hh) or np.isnan(a[i]) or Hh - L < 2 * a[i]: pend = None; continue
        if pend is None:
            if lo[i] < L:
                if c[i] > L: ev[i] = 1; depth[i] = (L - lo[i]) / a[i]
                else: pend = [i, L, lo[i]]
        else:
            pend[2] = min(pend[2], lo[i])
            if c[i] > pend[1]: ev[i] = 1; depth[i] = (pend[1] - pend[2]) / a[i]; pend = None
            elif i - pend[0] >= 2: pend = None
    return ev, depth
rows = []; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sorted(b.tk.unique()):
    H = q.build(fs[tk]); r_, l_, a_, fl = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); r_ = np.asarray(r_); a_ = np.asarray(a_); hi, lo, c = H.high.values, H.low.values, H.close.values; pos = {t: i for i, t in enumerate(H.dt.values)}
    sc = {'A': swings_zz(hi, lo, a_ * 2.0), 'B': swings_vec(hi, lo)}; E = {s: events(sc[s][0], sc[s][1], lo, c, a_) for s in sc}
    for _, e in b[b.tk == tk].iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None: continue
        k = j - 1; n = 0
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None or k < 60: continue
        d = dict(tk=tk, entry=e.entry, ret=e.ret)
        for s in ('A', 'B'):
            ev, dp = E[s]; L0, H0 = sc[s]
            for W in (20, 40):
                idx = np.nonzero(ev[k - W:k + 1])[0]; d[f'{s}{W}'] = int(len(idx) > 0); d[f'{s}{W}d'] = dp[k - W + idx[-1]] if len(idx) else np.nan
            d[f'{s}in'] = float(c[k] <= H0[k]) if not np.isnan(H0[k]) else np.nan
        rows.append(d)
    print(tk, len(rows), flush=True)
F = pd.DataFrame(rows); F['tr'] = F.entry < '2022-01-01'; F['wk'] = F.entry.dt.to_period('W').astype(str); F.to_parquet(SP + 'sweep395.parquet'); print('rows', len(F), 'of', len(b))
def diff(d, lab, a, bb):
    out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[a(D) | bb(D)].copy(); D['top'] = a(D).astype(int)
        if D.top.nunique() < 2 or len(D) < 30: out.append(h + ': n/a'); continue
        m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); out.append('%s: yes %+.2f%% (n %d) no %+.2f%% (n %d) diff %+.2fpp t %+.1f' % (h, D[D.top == 1].ret.mean() * 100, D.top.sum(), D[D.top == 0].ret.mean() * 100, (1 - D.top).sum(), m.params['top'] * 100, m.tvalues['top']))
    print('%-40s %s | %s' % (lab, out[0], out[1]), flush=True)
for s, nm in (('A', 'zigzag 2ATR'), ('B', 'vector 8-bar pivots')):
    print('\n== scale %s (%s): share of flips with a sweep-and-reclaim in last 20 bars %.0f%%, 40 bars %.0f%%' % (s, nm, F[s + '20'].mean() * 100, F[s + '40'].mean() * 100))
    for W in (20, 40): diff(F, '%s SWEEP within %d bars (+)' % (s, W), lambda D, c=f'{s}{W}': D[c] == 1, lambda D, c=f'{s}{W}': D[c] == 0)
    d = F[F[s + '20'] == 1].copy(); q1, q2 = d[d.tr][s + '20d'].quantile([1 / 3, 2 / 3]); diff(d, '%s deep vs shallow sweep (W20)' % s, lambda D, c=s + '20d': D[c] >= q2, lambda D, c=s + '20d': D[c] <= q1)
    diff(F.dropna(subset=[s + 'in']), '%s flip inside range vs above H0' % s, lambda D, c=s + 'in': D[c] == 1, lambda D, c=s + 'in': D[c] == 0)
