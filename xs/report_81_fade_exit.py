"""Nov 14 2026 THE EXIT HALF OF REY'S POINT: 'flip after the move fades'. report_80: on huge legs the vector exits about 24-28 days after the peak and gives back 22-39% of the move (median F 30-39%). Test trailing exits that act on the fade, declared before results (8 rules, nothing else): a trail T below the RUNNING HIGH (highs through the previous bar; stop checked against the current bar's low, filled at the stop or at the open if it gaps below), either active from entry (T = 30%, 40%) or ACTIVATED only after the unrealised gain from the entry open reaches A (A = 50% or 100%; T = 20%, 30%, 40%). Otherwise the vector's bear-flip exit as now. Per era (2019-21 / 2022-26Q1) on the screener rule (BV>=1.5) and on all flips: mean and median net (30bp), HUGE (>= +100% net) count kept vs no-trail, mean of the HUGE legs themselves (what the trail does to winners), and the delta in total return per flip. Rey's no-extra-stop rule is the thing being measured."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl')
RULES = [('trail30 from entry', 0, .30), ('trail40 from entry', 0, .40), ('act+50% trail20', .5, .20), ('act+50% trail30', .5, .30), ('act+50% trail40', .5, .40), ('act+100% trail20', 1., .20), ('act+100% trail30', 1., .30), ('act+100% trail40', 1., .40)]
out = {k: np.full(len(P), np.nan) for k, _, _ in RULES}; pos = {i: n for n, i in enumerate(P.index)}
for tk, g in P.groupby('tk'):
    B = D[tk]; B = B[~B.index.duplicated()]; o = B.open.values; h = B.high.values; l = B.low.values; ix = B.index
    for i, r in zip(g.index, g.itertuples()):
        e = ix.get_indexer([r.t])[0]
        if e < 0 or e + r.hold >= len(o): continue
        x = e + int(r.hold); ent = o[e]
        for k, A, T in RULES:
            run = ent; act = (A == 0); ex = o[x]
            for d in range(e, x):
                if d > e: run = max(run, h[d - 1])
                else: run = max(run, ent)
                if not act and run >= ent * (1 + A): act = True
                if act:
                    S = run * (1 - T)
                    if d > e and o[d] <= S: ex = o[d]; break
                    if l[d] <= S and S > 0: ex = S; break
            out[k][pos[i]] = ex / ent - 1 - 0.003
for k, _, _ in RULES: P['t_' + k] = out[k]
P = P.dropna(subset=['t_trail30 from entry'])
for lab, S0 in (('SCREENER RULE (BV>=1.5)', P[P.BV >= 1.5]), ('ALL FLIPS', P)):
    print(f'\n=== {lab}')
    for en, z in (('2019-21', S0[S0.t < '2022']), ('2022-26Q1', S0[S0.t >= '2022'])):
        hg = z.rn >= 1; print(f' {en}: n {len(z)} | VECTOR EXIT mean {z.rn.mean()*100:+.2f}% med {z.rn.median()*100:+.2f} | HUGE(net>=+100%) {int(hg.sum())}, their mean {z.rn[hg].mean()*100:+.0f}%')
        for k, _, _ in RULES:
            s = z['t_' + k]; print(f'     {k:20s} mean {s.mean()*100:+6.2f}% ({(s.mean()-z.rn.mean())*100:+.2f}pp) med {s.median()*100:+6.2f} | HUGE kept {int((s>=1).sum()):3d} of {int(hg.sum())} | those winners now average {s[hg].mean()*100:+5.0f}% (vs {z.rn[hg].mean()*100:+.0f}%) | trail fired on {int((s.round(6)!=z.rn.round(6)).mean()*100)}% of flips')
