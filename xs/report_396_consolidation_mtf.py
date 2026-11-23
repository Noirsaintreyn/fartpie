"""Nov 23 2026 report 396. Rey: 'specifically consolidation or accumulation in ANY timeframe ... look at how they define it'.
Definition taken from the published Wyckoff trading-range / LuxAlgo 'trading range' procedure and the AMD description (sources: luxalgo.com/library/concept/trading-range, .../accumulation-manipulation-distribution; no fixed numbers are published, so parameters are fixed here BEFORE the run):
 VALID RANGE at TF bar t = the last n=20 TF bars satisfy ALL of: (1) width (max high - min low)/ATR14(TF) <= k = 4.0; (2) >= 2 separated touches (>= 3 bars apart) of the high zone (high >= RH - 0.5 ATR) AND >= 2 of the low zone (low <= RL + 0.5 ATR) [swing highs/lows clustered in horizontal zones]; (3) sideways confirmation: Kaufman efficiency ratio of the 20 closes <= 0.30 [overlapping candles, no net drift].
 CONS_tf = a valid range existed at any of the last 5 completed TF bars (the flip came out of a consolidation).
 SEQ_tf = AMD sequence: within the last 10 completed TF bars a MANIPULATION event happened - a bar whose low is below the prior valid range low RL and whose close is back above RL (a wick through the edge that snaps back = failed attempt) - and since then price has closed above the range midpoint (DISTRIBUTION confirmation, 'reclaim through the open' analogue) before/at the flip.
Timeframes: 15m, 1H, 4H, daily (built from the 15m cache; a TF bar is used only after it closes).  Anchors/clock times are NOT used.
Tests on the 12,360 live flips: yes-minus-no mean flip return per TF for CONS and SEQ; ANY-TF versions; number of TFs in consolidation (0 / 1 / 2+); tightness; week-clustered OLS t; train 2019-21 | test 2022-26; admit = same sign both halves, |t|>=2 both (16+ tests: treat a lone winner as a lead)."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
import report_295_flip_quality as q
from vector_pine import run_state_machine
from vector_core import compute_atr_series, ATR_LEN
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; N = 20; KW = 4.0
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b = b[b.entry >= '2019-01-01'].reset_index(drop=True)
def tf_bars(H, rule):
    x = H[H.m < 960].set_index('dt')[['open', 'high', 'low', 'close']]
    if rule == '15m': r = x.copy(); r['end'] = r.index + pd.Timedelta('15min'); return r.reset_index()
    if rule == 'D': g = x.groupby(x.index.normalize()); r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')); r.index = r.index.rename('dt'); r['end'] = r.index + pd.Timedelta('16h'); return r.reset_index()
    g = x.groupby([x.index.normalize(), pd.Grouper(freq=rule, origin='start_day', offset='9h30min')]); r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); idx = r.index.get_level_values(1); r = r.reset_index(drop=True); r.insert(0, 'dt', idx); r['end'] = r.dt + pd.Timedelta(rule); return r
def analyse(r):
    hi, lo, c = r.high.values, r.low.values, r.close.values; n = len(r); a = compute_atr_series(hi, lo, c, ATR_LEN); valid = np.zeros(n, bool); RL = np.full(n, np.nan); RH = RL.copy(); tight = np.full(n, np.nan)
    for t in range(N - 1, n):
        if np.isnan(a[t]) or a[t] <= 0: continue
        h, l, cc = hi[t - N + 1:t + 1], lo[t - N + 1:t + 1], c[t - N + 1:t + 1]; rh, rl = h.max(), l.min(); w = (rh - rl) / a[t]
        if w > KW: continue
        def touches(mask):
            idx = np.nonzero(mask)[0]; cnt = 0; last = -99
            for i in idx:
                if i - last >= 3: cnt += 1; last = i
            return cnt
        if touches(h >= rh - .5 * a[t]) < 2 or touches(l <= rl + .5 * a[t]) < 2: continue
        path = np.abs(np.diff(cc)).sum(); er = abs(cc[-1] - cc[0]) / path if path > 0 else 1
        if er > .30: continue
        valid[t] = True; RL[t] = rl; RH[t] = rh; tight[t] = w
    # sequence
    seq = np.zeros(n, bool); last_ev = -99; mid = np.nan; seen_mid = False
    for t in range(1, n):
        if valid[t - 1] and lo[t] < RL[t - 1] and c[t] > RL[t - 1]: last_ev = t; mid = (RH[t - 1] + RL[t - 1]) / 2; seen_mid = False
        if last_ev > 0 and t - last_ev <= 10:
            if t > last_ev and c[t] > mid: seen_mid = True
            seq[t] = seen_mid
    cons = np.array([valid[max(0, t - 4):t + 1].any() for t in range(n)])
    return pd.DataFrame({'known': r.end.values, 'cons': cons, 'seq': seq, 'tight': tight})
rows = []; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sorted(b.tk.unique()):
    H = q.build(fs[tk]); r_ = np.asarray(run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt))[0]); pos = {t: i for i, t in enumerate(H.dt.values)}; TFS = {}
    for rule in ('15m', '1h', '4h', 'D'):
        S = analyse(tf_bars(H, rule)).sort_values('known'); TFS[rule] = S
    ks = []; meta = []
    for _, e in b[b.tk == tk].iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None: continue
        k = j - 1; n = 0
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None: continue
        ks.append(H.dt.values[k] + np.timedelta64(15, 'm')); meta.append((e.entry, e.ret))
    Q = pd.DataFrame({'t': ks, 'entry': [m[0] for m in meta], 'ret': [m[1] for m in meta]}).sort_values('t')
    for rule, S in TFS.items():
        m = pd.merge_asof(Q[['t']], S, left_on='t', right_on='known', direction='backward'); Q[f'C_{rule}'] = m.cons.values.astype(float); Q[f'S_{rule}'] = m.seq.values.astype(float); Q[f'W_{rule}'] = m.tight.values
    Q['tk'] = tk; rows.append(Q.drop(columns='t')); print(tk, sum(len(x) for x in rows), flush=True)
F = pd.concat(rows, ignore_index=True); F['tr'] = F.entry < '2022-01-01'; F['wk'] = F.entry.dt.to_period('W').astype(str); F.to_parquet(SP + 'cons396.parquet'); print('rows', len(F))
tfs = ('15m', '1h', '4h', 'D'); F['C_any'] = F[[f'C_{t}' for t in tfs]].max(axis=1); F['S_any'] = F[[f'S_{t}' for t in tfs]].max(axis=1); F['nC'] = F[[f'C_{t}' for t in tfs]].sum(axis=1)
def diff(d, lab, a, bb):
    out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[a(D) | bb(D)].copy(); D['top'] = a(D).astype(int)
        if D.top.nunique() < 2 or D.top.sum() < 20 or (1 - D.top).sum() < 20: out.append(h + ': n/a (n %d)' % D.top.sum()); continue
        m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); out.append('%s: yes %+.2f%% (n %d) no %+.2f%% (n %d) diff %+.2fpp t %+.1f' % (h, D[D.top == 1].ret.mean() * 100, D.top.sum(), D[D.top == 0].ret.mean() * 100, (1 - D.top).sum(), m.params['top'] * 100, m.tvalues['top']))
    print('%-34s %s | %s' % (lab, out[0], out[1]), flush=True)
print('\nshares of flips: ' + ', '.join('CONS_%s %.0f%% SEQ_%s %.1f%%' % (t, F[f'C_{t}'].mean() * 100, t, F[f'S_{t}'].mean() * 100) for t in tfs) + ', CONS_any %.0f%%, SEQ_any %.1f%%' % (F.C_any.mean() * 100, F.S_any.mean() * 100))
for t in tfs: diff(F, 'CONS_%s (came out of range)' % t, lambda D, c=f'C_{t}': D[c] == 1, lambda D, c=f'C_{t}': D[c] == 0); diff(F, 'SEQ_%s (sweep + reclaim + mid)' % t, lambda D, c=f'S_{t}': D[c] == 1, lambda D, c=f'S_{t}': D[c] == 0)
diff(F, 'CONS_any', lambda D: D.C_any == 1, lambda D: D.C_any == 0); diff(F, 'SEQ_any', lambda D: D.S_any == 1, lambda D: D.S_any == 0); diff(F, '2+ TFs in consolidation vs none', lambda D: D.nC >= 2, lambda D: D.nC == 0)
