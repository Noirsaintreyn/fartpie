"""Risk-matched (levered) version of the live model: scale the daily mark-to-market book to SPY's / QQQ's volatility and to their max drawdown.
Financing: borrowed part pays the approx. T-bill rate + 0.5%/yr (rates approximate by year)."""
import numpy as np, pandas as pd
exec(open('xs/alpha_test.py').read().split('ret = CL.pct_change()')[0]); ret = CL.pct_change()
RF = {2019: .021, 2020: .004, 2021: .0005, 2022: .02, 2023: .05, 2024: .05, 2025: .042, 2026: .038}
def run(b, L):
    rf = pd.Series([RF[d.year] + .005 for d in b.index], index=b.index) / 252
    return L * b - max(L - 1, 0) * rf
def st(r): e = (1 + r).cumprod(); y = len(r) / 252; return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min(), r.std() * np.sqrt(252)
B = book(taken(T[T.ratio >= 2.5])); X = pd.DataFrame({'b': B, 'SPY': ret.SPY.reindex(D), 'QQQ': ret.QQQ.reindex(D)}).dropna()
PER = (('2019-21', '2019', '2022'), ('2022-26', '2022', '2027'), ('2019-26', '2019', '2027'))
full = X; vb, vs, vq = full.b.std(), full.SPY.std(), full.QQQ.std()
_, ddb, _ = st(full.b); _, dds, _ = st(full.SPY); _, ddq, _ = st(full.QQQ)
rows = [('SPY buy & hold', None, 'SPY'), ('QQQ buy & hold', None, 'QQQ'), ('Model 2.5, 1x', 1.0, 'b'),
        (f'Model, same vol as SPY ({vs/vb:.2f}x)', vs / vb, 'b'), (f'Model, same vol as QQQ ({vq/vb:.2f}x)', vq / vb, 'b')]
# drawdown-matched: search leverage so 2019-26 max DD equals SPY's / QQQ's
for nm, target in (('SPY', dds), ('QQQ', ddq)):
    Ls = np.arange(1, 4.01, .05); L = min(Ls, key=lambda l: abs(st(run(full.b, l))[1] - target)); rows.append((f'Model, same max DD as {nm} ({L:.2f}x)', L, 'b'))
print(f'2.5 cutoff, {len(T[T.ratio>=2.5].tk.unique())} names. Vol 2019-26: model {vb*np.sqrt(252):.1%}, SPY {vs*np.sqrt(252):.1%}, QQQ {vq*np.sqrt(252):.1%}\n')
print(f'{"":40}' + ''.join(f'{p:>24}' for p, _, _ in PER) + '\n' + f'{"":40}' + '        CAGR / maxDD / vol' * 3)
for nm, L, col in rows:
    out = ''
    for _, a, z in PER:
        x = X[(X.index >= a) & (X.index < z)]; r = x[col] if L is None else run(x.b, L); c, d, v = st(r); out += f'{c:>+10.1%} /{d:>5.0%} /{v:>5.0%}  '
    print(f'  {nm:<38}' + out, flush=True)
