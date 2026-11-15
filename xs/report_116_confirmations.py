"""Nov 15 2026 CONFIRMATIONS BESIDES FLIP-BAR VOLUME (Rey). Declared before results. Daily flips (report_90 file, 30,323, 2019-Mar 2026, entry next open, hold to bear flip, 30bp, survivorship-biased), every feature known at the flip-bar CLOSE (no look-ahead). Declared sign in brackets, each from a mechanism: F1 close location in the flip bar's range (c-l)/(h-l) [+ strong close = buyers held]; F2 flip-bar range / ATR14 of the prior bar [+ expansion = real break]; F3 gap = open / prior close - 1 [- gap-ups are the late chase]; F4 close / 252-bar high [+ near highs = no overhead supply]; F5 pre-flip volume trend = mean volume of the prior 5 bars / prior 60 bars [- a quiet base then a spike = fresh]; F6 close above its 200-day SMA [+]; F7 20-day relative strength = own 20-day return minus the equal-weight universe 20-day return [+ leaders]; F8 chop = count of bull flips of the same name in the prior 60 bars [- whipsaw]; F9 base length = bars since the name last was in bull state [+ long base before the flip]; F10 breadth = share of the universe in bull state at the prior close [+]; F11 distance of the flip bar close above the vector level = (c - level)/c [- tight ratchet = small stop; mechanical, reported only]. Universes: (A) all flips, (B) price < $5, (C) price < $5 and frozen score top 20% (the report_115b shortlist). Metrics per feature and era (TRAIN 2019-21, TEST 2022-26): Spearman IC to net return and to the >= +10% flag, top-tercile minus bottom-tercile mean net (terciles cut on TRAIN), month-clustered t of that spread; a feature PASSES if the declared sign holds for IC and spread on BOTH eras and TEST month-t >= 2 in universe A or B. If anything passes, PART 2 combines the passing features as a count filter inside universe C and judges it on TEST only. Universe C is small (254 train / 704 test), so it is reported for sign agreement, not significance."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); COST = .003
regs = {}; closes = {}; info = {}
for tk, g in F.groupby('tk'):
    q = D.get(tk)
    if q is None or len(q) < 300: continue
    q = q[['open', 'high', 'low', 'close', 'volume']].copy(); q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize(); reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); regs[tk] = pd.Series(reg, index=q.index); closes[tk] = q.close
    o = q.open.values; h = q.high.values; l = q.low.values; c = q.close.values; v = q.volume.values.astype(float); n = len(q); pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); a14 = pd.Series(tr).rolling(14).mean().values; hi252 = pd.Series(h).rolling(252, min_periods=120).max().values; sma200 = pd.Series(c).rolling(200).mean().values; v5 = pd.Series(v).rolling(5).mean().shift(1).values; v60 = pd.Series(v).rolling(60).mean().shift(1).values; isf = np.r_[False, (reg[1:] == 1) & (reg[:-1] != 1)]; cf = np.cumsum(isf); pos = {d: k for k, d in enumerate(q.index)}
    for _, r in g.iterrows():
        i = pos.get(r.d)
        if i is None or i < 61: continue
        k = i - 1
        while k > 0 and reg[k] == 1: k -= 1
        bl = 0
        while k > 0 and reg[k] != 1: bl += 1; k -= 1
        info[(tk, r.d)] = dict(F1=(c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else .5, F2=(h[i] - l[i]) / a14[i - 1] if a14[i - 1] > 0 else np.nan, F3=o[i] / c[i - 1] - 1, F4=c[i] / hi252[i] if hi252[i] > 0 else np.nan, F5=v5[i] / v60[i] if v60[i] > 0 else np.nan, F6=float(c[i] > sma200[i]) if not np.isnan(sma200[i]) else np.nan, r20=c[i] / c[i - 20] - 1, F8=float(cf[i] - cf[max(0, i - 60)] - 1), F9=float(bl), F11=(c[i] - lev[i]) / c[i] if lev[i] == lev[i] else np.nan)
I = pd.DataFrame.from_dict(info, orient='index'); I.index = pd.MultiIndex.from_tuples(I.index, names=['tk', 'd']); A = F.merge(I.reset_index(), on=['tk', 'd'])
R = pd.DataFrame(regs); C = pd.DataFrame(closes); ew20 = (C / C.shift(20) - 1).mean(axis=1); bull = ((R == 1).sum(axis=1) / R.notna().sum(axis=1)).shift(1)
A['F7'] = A.r20 - A.d.map(ew20); A['F10'] = A.d.map(bull); A['n'] = A.r - COST; A['mo'] = A.d.dt.to_period('M').astype(str); A['y'] = A.d.dt.year; A = A.dropna(subset=['F1', 'F2', 'F4', 'F5', 'F6', 'F7', 'F10']); print('flips with features', len(A))
cut80 = F[~F.test].lr.quantile(.8); A['top'] = A.lr >= cut80; sign = dict(F1='+', F2='+', F3='-', F4='+', F5='-', F6='+', F7='+', F8='-', F9='+', F10='+', F11='-'); UN = {'A all': A, 'B px<5': A[A.px < 5], 'C px<5 & top20': A[(A.px < 5) & A.top]}; passed = set()
for un, U in UN.items():
    print(f'\n######## {un}: train n {(~U.test).sum()} test n {U.test.sum()} ########')
    for f, sg in sign.items():
        out = []; ok = True
        tr = U[~U.test]; q1, q2 = tr[f].quantile([1 / 3, 2 / 3]).values
        for era, z in (('TR', U[~U.test]), ('TE', U[U.test])):
            z = z.dropna(subset=[f]); ic = spearmanr(z[f], z.n)[0]; ic10 = spearmanr(z[f], z.n >= .1)[0]; t_ = z[z[f] >= q2]; b_ = z[z[f] <= q1]; mt = t_.groupby('mo').n.mean(); mb = b_.groupby('mo').n.mean(); d = (mt - mb).dropna(); tt = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 and d.std() > 0 else np.nan
            out.append(f'{era}: IC {ic:+.3f} IC10 {ic10:+.3f} top {t_.n.mean()*100:+6.2f}% bot {b_.n.mean()*100:+6.2f}% spread {(t_.n.mean()-b_.n.mean())*100:+6.2f}pp t {tt:+.2f}')
            ok &= ((ic > 0) == (sg == '+')) and (((t_.n.mean() - b_.n.mean()) > 0) == (sg == '+')) and (era == 'TR' or (tt == tt and abs(tt) >= 2 and (tt > 0) == (sg == '+')))
        print(f'  {f:3s} [{sg}] ' + ' | '.join(out) + ('   <== PASS' if ok else '')); 
        if ok and un in ('A all', 'B px<5'): passed.add(f)
print('\nPASSED (A or B):', sorted(passed))
A.to_pickle(X + 'confirm_features.pkl')
