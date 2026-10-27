import numpy as np, pandas as pd, glob
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
S = prep(glob.glob('xs/ext/adv/tr/*.parquet')); L = S[S['var'] == 'live']
F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/adv/ft/*.parquet')).drop(columns=['exit', 'ret']); L = L.merge(F, on=['tk', 'h1', 'entry'], how='left')
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
Y = [(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01')) for y in range(2019, 2027)]; base = [sim(L, a, b)[0] for a, b in Y]
def show(nm, Z):
    yr = [sim(Z, a, b)[0] for a, b in Y]
    print(f'  {nm:<46}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)) + f'  beats live {sum(x > y + 1e-9 for x, y in zip(yr, base))}/8 ' + ' '.join(f'{x - y:+.0%}' for x, y in zip(yr, base)), flush=True)
T = taken(L); key = set(zip(T.tk, T.entry)); L['inbook'] = [k in key for k in zip(L.tk, L.entry)]
for v in (0, 1, 2, 3, 4):
    z = L[L.nsw == v]; b = z[z.inbook]; print(f'nsw={v}: signals {len(z)} avg {z.ret.mean()*100:+.2f}% | taken by book {len(b)} avg {b.ret.mean()*100:+.2f}%  by year (taken): ' + ' '.join(f'{y}:{b[b.d.dt.year == y].ret.mean()*100:+.1f}' for y in range(2019, 2027)))
print(f'{"":<48}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', L); show('skip nsw == 2', L[~(L.nsw == 2)]); show('skip nsw in 2..3', L[~L.nsw.between(2, 3)]); show('skip nsw in 2..4', L[~L.nsw.between(2, 4)]); show('skip nsw >= 2', L[~(L.nsw >= 2)])
z = L.copy(); z.loc[z.nsw == 2, 'ret'] = 0.0; show('nsw==2 slot held as cash (attribution A)', z)
