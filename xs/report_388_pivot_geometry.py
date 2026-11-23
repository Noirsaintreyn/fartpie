"""Nov 23 2026 report 388. Rey: 'run pivot geometry, try lower and higher time frame; better trades come when the market dumps and the higher-TF vector is likely NOT flipped'.
Hypothesis (his): a 15m bull flip that fires while the HIGHER-timeframe vector is still bearish (not flipped) is a reversal out of a dump and follows through better; the vector's own geometry (how far the flip level sits from price, how far price broke the old level, how long/ratcheted the prior bear run was) also carries follow-through.
Features at the flip bar (all known before entry), on the 54 b302 flips (matched by tk, entry):
 H1 = 1H vector regime (b302 s1 is 0/1 only: 'not bullish' vs 'bullish'); H4 = 4H vector regime, D1 = daily vector regime (state machines on bars built from the 15m RTH data, state used only after the higher-TF bar closes); H4/D1 are -1/+1.
 G1 = (close - new ratchet level)/ATR at the flip bar (room in ATR); G2 = (close - old bear level)/ATR (break strength); G3 = number of ratchets of the prior bear run; G4 = prior bear-run length (bars; b302 bearlen).
Lower timeframe: no 5m cache exists for the names (QQQ only), so 5m is NOT tested.
Test: tercile (or sign) top-minus-bottom mean flip return, week-clustered OLS t, terciles cut on train 2019-21, train | test; admit = same sign both halves, |t| >= 2 both.  Then 'not flipped' (H4 = -1 or D1 = -1) vs flipped, by half."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from vector_pine import run_state_machine
import report_295_flip_quality as q
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b = b[b.entry >= '2019-01-01'].reset_index(drop=True)
def htf(H, rule):
    x = H[H.m < 960].set_index('dt')[['open', 'high', 'low', 'close']]
    if rule == '4H': g = x.groupby([x.index.normalize(), pd.Grouper(freq='4h', origin='start_day', offset='9h30min')])
    else: g = x.groupby(x.index.normalize())
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); idx = r.index.get_level_values(-1) if rule == '4H' else r.index; r = r.reset_index(drop=True); r.insert(0, 'dt', idx)
    reg = np.asarray(run_state_machine(r[['open', 'high', 'low', 'close']].assign(datetime=r.dt))[0]); known = r.dt + (pd.Timedelta('4h') if rule == '4H' else pd.Timedelta('1D'))
    s = pd.DataFrame({'known': known, 'r': reg}).sort_values('known'); return pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), s, left_on='t', right_on='known', direction='backward').r.fillna(0).values
rows = []; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sorted(b.tk.unique()):
    H = q.build(fs[tk]); H['h4'] = htf(H, '4H'); H['d1'] = htf(H, '1D'); r_, l_, a_, fl = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); r_ = np.asarray(r_); l_ = np.asarray(l_); a_ = np.asarray(a_)
    close = H.close.values; dts = H.dt.values; pos = {t: i for i, t in enumerate(dts)}; ent = b[b.tk == tk]
    # ratchet count per regime run, and old level at the flip
    runstart = np.zeros(len(H), int); nrat = np.zeros(len(H)); s = 0; c = 0
    for i in range(len(H)):
        if i > 0 and r_[i] != r_[i - 1]: s = i; c = 0
        elif i > 0 and l_[i] != l_[i - 1]: c += 1
        runstart[i] = s; nrat[i] = c
    for _, e in ent.iterrows():
        j = pos.get(np.datetime64(e.entry));
        if j is None: continue
        # flip bar = last bar before entry at which regime turned bullish
        k = j - 1
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1): 
            k -= 1
            if j - k > 40: k = None; break
        if k is None or k < 2: continue
        a = a_[k]; old = l_[k - 1]
        rows.append(dict(tk=tk, entry=e.entry, ret=e.ret, H1=e.s1, H4=H.h4[k], D1=H.d1[k], G1=(close[k] - l_[k]) / a, G2=(close[k] - old) / a, G3=nrat[k - 1], G4=e.bearlen))
    print(tk, len(rows), flush=True)
F = pd.DataFrame(rows); F['tr'] = F.entry < '2022-01-01'; F['wk'] = F.entry.dt.to_period('W').astype(str); F.to_parquet(SP + 'geom388.parquet'); print('matched', len(F), 'of', len(b))
def cut(d, col):
    q1, q2 = d[d.tr][col].quantile([1 / 3, 2 / 3]); return np.where(d[col] <= q1, 'lo', np.where(d[col] >= q2, 'hi', 'mid'))
def diff(d, lab, hi, lo):
    out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[hi(D) | lo(D)].copy(); D['top'] = hi(D).astype(int)
        if D.top.nunique() < 2 or len(D) < 30: out.append(h + ': n/a'); continue
        m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); out.append('%s: A %+.2f%% (n %d) B %+.2f%% (n %d) diff %+.2fpp t %+.1f' % (h, D[D.top == 1].ret.mean() * 100, D.top.sum(), D[D.top == 0].ret.mean() * 100, (1 - D.top).sum(), m.params['top'] * 100, m.tvalues['top']))
    print('%-46s %s | %s' % (lab, out[0], out[1]), flush=True)
print('\nA = high / bullish group, B = low / bearish group')
for col in ('G1', 'G2', 'G3', 'G4'):
    d = F.dropna(subset=[col]).copy(); d['T'] = cut(d, col); diff(d, col + ' top third vs bottom third', lambda D: D['T'] == 'hi', lambda D: D['T'] == 'lo')
diff(F, 'H1 bullish (1) vs not (0)', lambda D: D.H1 == 1, lambda D: D.H1 == 0); diff(F, 'H4 bull(+1) vs bear(-1)', lambda D: D.H4 == 1, lambda D: D.H4 == -1); diff(F, 'D1 bull(+1) vs bear(-1)', lambda D: D.D1 == 1, lambda D: D.D1 == -1)
print('\nshares: H4 bear %.0f%%, D1 bear %.0f%% of flips' % ((F.H4 == -1).mean() * 100, (F.D1 == -1).mean() * 100))
F['nf'] = ((F.H4 == -1) | (F.D1 == -1)).astype(int); diff(F, 'any HTF (4H or D1) NOT flipped vs all bullish', lambda D: D.nf == 1, lambda D: D.nf == 0)
