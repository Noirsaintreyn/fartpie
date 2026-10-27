import numpy as np, pandas as pd, glob
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec']
F = pd.read_parquet('xs/ext/px/feat.parquet'); L = L.merge(F, on=['tk', 'h1', 'entry'], how='left')
L['half'] = np.where(L.d.dt.year >= 2022, '22-26', '19-21'); print('trades', len(L), 'peer data', L.peer_bull.notna().sum(), 'options-vector data', L.opt_vol_trend.notna().sum())
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'    {nm:<60}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)), flush=True)
def tab(lab, q, Z):
    g = Z.groupby([q, 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100; print(f'\n{lab}'); print(g.round(3).to_string())
print(f'\n{"":<64}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
Z = L[L.peer_bull.notna()]
tab('share of industry peers in a daily bull trend', pd.cut(Z.peer_bull, [-.01, .25, .5, .75, 1.01]), Z)
tab('peers that flipped bull in the last 3 days', pd.cut(Z.peer_new, [-1, 0, 1, 3, 99], labels=['0', '1', '2-3', '4+']), Z)
Zl = Z[Z.lead.notna()]; tab('leader (1 = all bull peers flipped after it) vs laggard (0)', pd.cut(Zl.lead, [-.01, .25, .75, 1.01], labels=['laggard', 'middle', 'leader']), Zl)
show('skip when < 25% of peers are bull', L[~(L.peer_bull < .25)]); show('skip when > 75% of peers are bull (crowded)', L[~(L.peer_bull > .75)])
show('only when >= 1 peer flipped in the last 3 days', L[~(L.peer_new == 0)]); show('skip when >= 4 peers flipped in 3 days', L[~(L.peer_new >= 4)])
show('skip laggards', L[~(L.lead <= .25)]); show('skip leaders', L[~(L.lead > .75)])
for c in ('peer_bull', 'lead'):
    for sgn, nm in ((1, 'high'), (-1, 'low')):
        z = L.copy(); z['ratio'] = z.ratio + (z[c].rank(pct=True) if sgn == 1 else 1 - z[c].rank(pct=True)).fillna(.5) * .5; show(f'tie-break toward {nm} {c}', z)
Z = L[L.opt_vol_trend.notna()]
for c, lab in (('opt_vol_trend', 'vector on option volume (1 = building)'), ('call_share_trend', 'vector on call share'), ('iv_trend', 'vector on ATM implied vol')):
    tab(lab, Z[c], Z)
    show(f'skip covered trades with {c} = bull', L[~(L[c] == 1)]); show(f'skip covered trades with {c} = bear', L[~(L[c] == -1)])
