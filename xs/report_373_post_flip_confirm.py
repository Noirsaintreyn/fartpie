"""Nov 22 2026 report 373. Rey's Pine 'Vector - Post-Flip Confirmation': after a flip, wait (2-10 bars) for 2 closes above flip-bar high+0.05ATR, progress >=0.5 ATR from flip close, efficiency (net/path) >=0.55, no adverse move >0.5 ATR, vector level not lost; then enter.
HYPOTHESIS (mechanism: flips that fail to follow through immediately are the ones that get stopped; waiting filters them at the cost of a worse entry). Declared: (a) confirmed flips have better ORIGINAL-entry return than unconfirmed (selection value); (b) entering at the confirmation (next open) beats the flip-entry on the same trades (net of the price given up).  Bull flips only, 15m, 54 names, exact Pine parameters, exit = open after the next bear flip, 4bp round trip.  Train 2019-21 | test 2022+, week-clustered t.  Pass: same sign both halves, |t|>=2, and (b) mean confirmed-entry return > original-entry return on all flips."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; COST = .0004
b = pd.read_parquet(SP + 'b302.parquet'); rows = []
MINW, MAXW, HOLD, BUF, PROG, EFF, ADV = 2, 10, 2, .05, .5, .55, .5
for tk in sorted(b.tk.unique()):
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    m = H.index.hour * 60 + H.index.minute; H = H[(m >= 570) & (m < 960)]
    regs, lv, atr, fl = run_state_machine(H); O, Hi, L, C = H.open.values, H.high.values, H.low.values, H.close.values; last = len(H) - 1
    for i, d in fl:
        if d != 1 or i + 1 > last: continue
        nxt = [j for j, dd in fl if dd == -1 and j > i]
        if not (nxt and nxt[0] + 1 <= last): continue
        j = nxt[0]; ex = O[j + 1]; ent0 = O[i + 1]; fh, fp, fa = Hi[i], C[i], atr[i]; path = 0; hc = 0; conf = None
        if not fa > 0: continue
        for k in range(i + 1, min(i + MAXW, j) + 1):
            path += abs(C[k] - C[k - 1]); hc = hc + 1 if C[k] > fh + fa * BUF else 0; prog = (C[k] - fp) / fa; eff = (C[k] - fp) / path if path > 0 else 0
            if not (C[k] > lv[k]) or (fp - L[k]) > fa * ADV: break
            if k - i >= MINW and prog >= PROG and eff >= EFF and hc >= HOLD: conf = k; break
        r0 = ex / ent0 - 1 - COST; r1 = (ex / O[conf + 1] - 1 - COST) if conf is not None and conf + 1 < j + 1 else np.nan
        rows.append((tk, H.index[i], r0, r1, 0 if conf is None else conf - i))
    print(tk, len(rows), end=' | ', flush=True)
A = pd.DataFrame(rows, columns=['tk', 'flip', 'r0', 'r1', 'lag']); A['conf'] = A.r1.notna(); A['wk'] = A.flip.dt.to_period('W').astype(str); A['half'] = np.where(A.flip < '2022-01-01', 'train', 'test'); A.to_parquet(SP + 'A373.parquet')
print('\nflips %d, confirmed %.0f%%, median lag %d bars' % (len(A), A.conf.mean() * 100, A[A.conf].lag.median()))
def clus(T, R):
    Z = pd.concat([T.assign(x=1.0), R.assign(x=0.0)]); mm = sm.OLS(Z.y.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); return mm.params[1] * 100, mm.tvalues[1]
for h in ('train', 'test', 'all'):
    S = A if h == 'all' else A[A.half == h]; c = S[S.conf]; u = S[~S.conf]
    a = clus(c.assign(y=c.r0), u.assign(y=u.r0)) if h != 'all' else (np.nan, np.nan)
    print('%-5s all flips: n %d mean %+.2f%% win %.0f%% | confirmed n %d: orig-entry %+.2f%%, confirm-entry %+.2f%% (win %.0f%%) | unconfirmed n %d: orig %+.2f%% | (a) conf-vs-unconf orig diff %+.2f pts t %+.1f' % (h, len(S), S.r0.mean() * 100, (S.r0 > 0).mean() * 100, len(c), c.r0.mean() * 100, c.r1.mean() * 100, (c.r1 > 0).mean() * 100, len(u), u.r0.mean() * 100, a[0], a[1]))
c = A[A.conf]; print('Per-name-trade totals: sum of returns all flips %.1f | confirmed at confirm-entry %.1f | confirmed at flip-entry %.1f | (compare capital use: confirmed only %.0f%% of trades)' % (A.r0.sum(), c.r1.sum(), c.r0.sum(), A.conf.mean() * 100))
