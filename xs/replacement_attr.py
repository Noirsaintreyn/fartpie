"""Oct 26 2026: replacement-effect attribution for flagged entries (as suggested in Rey's review note). For a flag on live trades:
 A  reserve   the flagged trade still takes its slot until its original exit but earns 0 (cash): was excluding it good, with no replacements?
 B  rerun     drop flagged trades and rerun the 6-slot scheduler (the deployable rule; other signals can take the slot)
 C  size      keep every admission, flagged trades at half size, the other half in cash
Flags: 2+ changepoints in ~5 sessions (cp5), 1h uptrend 40+ bars old (age1h), and both."""
import numpy as np, pandas as pd, glob
exec(open('xs/book_mtm.py').read().split("if __name__")[0].replace('/tmp/claude-0/ml/stopbook.py', 'xs/stop_book.py') if False else open('/tmp/claude-0/ml/vt/mtm.py').read().split("if __name__")[0])
src = open('/tmp/claude-0/ml/stopbook.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('/tmp/claude-0/ml/tm/*.parquet')); L = L[L['var'] == 'vec']
F = pd.concat(pd.read_parquet(f) for f in glob.glob('/tmp/claude-0/ml/fx/*.parquet')).drop(columns=['exit', 'ret']); L = L.merge(F, on=['tk', 'h1', 'entry'], how='left')
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)), flush=True)
print(f'{"":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
T = taken(L); T['k'] = list(zip(T.tk, T.entry)); L['k'] = list(zip(L.tk, L.entry)); inbook = L.k.isin(set(T.k))
for nm, fl in (('2+ changepoints', L.cp5 >= 2), ('1h trend 40+ bars old', L.age1h >= 40), ('either', (L.cp5 >= 2) | (L.age1h >= 40))):
    f = fl & inbook; print(f'\n{nm}: {fl.sum()} flagged signals, {f.sum()} of them actually taken by the live book; their avg return {L[f].ret.mean()*100:+.2f}% (book avg {L[inbook].ret.mean()*100:+.2f}%)')
    z = L.copy(); z.loc[fl, 'ret'] = 0.0; show('A reserve: flagged keep the slot, earn 0', z)
    show('B rerun: drop flagged, scheduler refills', L[~fl])
    z = L.copy(); z.loc[fl, 'w'] = .5; show('C size: flagged at half size', z)
