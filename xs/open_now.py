"""Nov 4 2026: open positions at data end (2026-10-21 09:30 mark) under live vs union-10: which trades hold slots, unrealized return, contribution = ret/N per slot. Breaker ignored (inactive)."""
import sys
sys.argv = sys.argv[:1] + ['live']
_s = open('xs/uni_slots_ratio.py').read(); exec(_s[:_s.index("base = evN(")])
END = W.exit.max()
def accepted(Y, N):
    o = np.lexsort((Y.tk.values, -Y.ratio.values.astype(float), Y.entry.values)); Z = Y.iloc[o]; op = []; keep = []
    for k, (e, x) in enumerate(zip(Z.entry.values, Z.exit.values)):
        op = [q for q in op if q > e]
        if len(op) >= N: continue
        op.append(x); keep.append(k)
    return Z.iloc[keep]
for nm, Y, N in (('LIVE S&P 6 slots', book('live', 1.5), 6), ('UNION 10 slots', book('union', 1.5), 10)):
    A = accepted(Y, N); op = A[A.exit >= END].sort_values('entry')
    print(f'\n{nm}: open at {END}: {len(op)} positions, slots used {len(op)}/{N}, unrealized book return {op.ret.sum()/N:+.2%} (avg position {op.ret.mean():+.2%})')
    print(op[['tk', 'entry', 'ratio', 'h1', 'ret']].assign(ret=lambda d: (d.ret * 100).round(2)).to_string(index=False))
