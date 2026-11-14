"""Nov 14 2026 CUTTING LOSERS ON THE EXIT SIDE (declared before results). report_75/75b: the worst-looking flips cannot be removed at the entry without removing 20-44% of the HUGE winners (the cut group has median -12% but mean +2%). So test cutting the loss AFTER entry instead: a hard stop from the entry open, filled at the stop price intraday (at the open if it gaps below), five fixed rules only: -10%, -15%, -20% and 3x / 5x the flip's ATR% below the entry; otherwise the vector's bear-flip exit as now. No re-entry until the next flip. Base = screener rule (flip-bar volume >=1.5x), also all flips. Per era (2019-21 / 2022-26Q1): mean and median net, share stopped, HUGE winners lost (a winner counts as lost if its original return >= +100% and the stop cut it below +100%), total return per original flip, and the stopped trades' average eventual outcome had they been held (the 'recovery' cost). Rey's standing rule for flip strategies is NO other stop: this measures what that rule costs/saves."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl')
RULES = [('hard -10%', lambda r: .10), ('hard -15%', lambda r: .15), ('hard -20%', lambda r: .20), ('3x ATR%', lambda r: 3 * r.ATRP), ('5x ATR%', lambda r: 5 * r.ATRP)]
out = {k: np.full(len(P), np.nan) for k, _ in RULES}; stopped = {k: np.zeros(len(P), bool) for k, _ in RULES}; pos = {}
for tk, g in P.groupby('tk'):
    B = D[tk]; B = B[~B.index.duplicated()]; o = B.open.values; l = B.low.values; ix = B.index
    for i, r in zip(g.index, g.itertuples()):
        e = ix.get_indexer([r.t])[0]
        if e < 0 or e + r.hold >= len(o): continue
        x = e + int(r.hold); ent = o[e]
        for k, f in RULES:
            S = ent * (1 - min(f(r), .95)); ex = o[x]; st = False
            for d in range(e, x):
                if d > e and o[d] <= S: ex = o[d]; st = True; break
                if l[d] <= S: ex = S; st = True; break
            out[k][i] = ex / ent - 1 - 0.003; stopped[k][i] = st
for k, _ in RULES: P['s_' + k] = out[k]; P['st_' + k] = stopped[k]
P = P.dropna(subset=['s_hard -10%'])
for lab, S0 in (('BASE = BV>=1.5 flips (screener rule)', P[P.BV >= 1.5]), ('BASE = all flips', P)):
    print(f'\n=== {lab}')
    for en, z in (('2019-21', S0[S0.t < '2022']), ('2022-26Q1', S0[S0.t >= '2022'])):
        print(f' {en}: n {len(z)} | NO STOP mean {z.rn.mean()*100:+.2f}% med {z.rn.median()*100:+.2f} HUGE kept {int(z.HUGE.sum())} | worst-decile starts {z.rn.quantile(.1)*100:+.1f}%')
        for k, _ in RULES:
            s = z['s_' + k]; st = z['st_' + k]; lost = int(((z.ret >= 1) & ((s + .003) < 1)).sum()); rec = z.rn[st].mean() * 100
            print(f'     {k:10s} mean {s.mean()*100:+6.2f}% med {s.median()*100:+6.2f} | stopped {st.mean()*100:4.1f}% (their no-stop mean {rec:+6.1f}%) | HUGE lost {lost:3d} of {int(z.HUGE.sum())} | worst decile starts {s.quantile(.1)*100:+.1f}% | delta mean {(s.mean()-z.rn.mean())*100:+.2f}pp')
