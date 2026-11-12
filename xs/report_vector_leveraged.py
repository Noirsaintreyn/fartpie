"""VECTOR ON LEVERAGED / INVERSE SINGLE-STOCK ETFs (Rey, Nov 11 2026). Fixed indicator (xs/vector_pine.py), nothing fitted. DECLARED BEFORE RESULTS.
Signal = the vector on the UNDERLYING's 1h bars (pre+RTH, as live). Instruments traded long-only: S1 = hold the bull ETF while the underlying is bull; S2 = hold the bull ETF while bull and the INVERSE ETF while bear (always in one of them);
S0 = baseline, same signal on the underlying itself (long when bull, flat otherwise). Fills at the next RTH bar's open; return = open-to-open of the held bar (gaps included). 1h bars from Yahoo (Dec 2023+; AMDD/PLTx from listing).
Per-bar edge (bp) vs 1000 circular shifts of the state series (same time in market). Classes: ALL | HV-pre (previous day's range in the top third of its history, known at the open).
Also: compounded return and max drawdown of S1/S2 vs buy-and-hold of the same ETF."""
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_vector_random.py').read(); exec(src[:src.index('# ---------------- Universe A')].replace("print('Universe A (random, stratified):'", "(lambda *a, **k: None)('Universe A'"))
PAIRS = [('NVDA', 'NVDL', 'NVDD'), ('TSLA', 'TSLL', 'TSLQ'), ('AAPL', 'AAPU', 'AAPD'), ('MSFT', 'MSFU', 'MSFD'), ('GOOGL', 'GGLL', 'GGLS'), ('AMZN', 'AMZU', 'AMZD'), ('COIN', 'CONL', None), ('PLTR', 'PLTU', 'PLTD'), ('AMD', None, 'AMDD'), ('BTC-USD', 'BITX', None)]
def rth_df(B):
    m = B.index.hour * 60 + B.index.minute; return B[(m >= 570) & (m < 960)]
def get(tk):
    d = yf.download(tk, period='730d', interval='1h', prepost=True, progress=False, auto_adjust=False); return clean(d) if len(d) else None
def sim_pair(U, L, I):
    BU = get(U)
    if BU is None or len(BU) < 500: return None
    P = prep(BU); reg = pd.Series(P['reg'], index=BU.index); R = {}
    for nm, tk in (('L', L), ('I', I)):
        if tk: R[nm] = rth_df(get(tk))
    Ur = rth_df(BU); idx = Ur.index
    for k in R: idx = idx.intersection(R[k].index)
    if len(idx) < 400: return None
    idx = idx.sort_values(); st = reg.reindex(idx).values.astype(float); st = np.r_[0.0, st[:-1]]; day = idx.normalize()   # state at the PREVIOUS bar's close drives this bar (filled at its open): no look-ahead
    rets = {}
    for nm, X in [('U', Ur)] + list(R.items()):
        o = X.reindex(idx).open.values; rets[nm] = np.r_[o[1:] / o[:-1] - 1, 0.0]
    cls = {'ALL': np.ones(len(idx), bool), 'HV-pre': pd.Series(P['cls']['HV-pre'], index=BU.index).groupby(BU.index.normalize()).last().reindex(day).fillna(False).values.astype(bool)}
    return dict(idx=idx, st=st, rets=rets, cls=cls, n=len(idx))
def strat(S, name):
    st, r = S['st'], S['rets']
    if name == 'S0': return np.where(st > 0, r['U'], 0.0), (st > 0)
    if name == 'S1': return (np.where(st > 0, r['L'], 0.0), (st > 0)) if 'L' in r else (None, None)
    if name == 'S2':
        if 'L' in r and 'I' in r: return np.where(st > 0, r['L'], np.where(st < 0, r['I'], 0.0)), (st != 0)
        if 'I' in r: return np.where(st < 0, r['I'], 0.0), (st < 0)          # AMD: inverse only on bear
        return None, None
def edge(S, name, cls, nsh=1000):
    x, inm = strat(S, name)
    if x is None: return None
    M = S['cls'][cls]; k = M.sum()
    if k < 50: return None
    base = x.copy(); obs = base[M].mean() * 1e4; nul = []
    st = S['st']; r = S['rets']
    for _ in range(nsh):
        sh = rng.integers(40, S['n'] - 40); s2 = np.roll(st, sh); S2 = dict(S, st=s2); y, _ = strat(S2, name); nul.append(y[M].mean() * 1e4)
    nul = np.array(nul); sd = base[M].std() * 1e4
    return dict(obs=obs, nul=nul, nm=nul.mean(), p=(nul >= obs).mean(), w=k, t=obs / (sd / np.sqrt(k) + 1e-9), tim=inm[M].mean())
def curve(x): e = np.cumprod(1 + x); return e[-1] - 1, (e / np.maximum.accumulate(e) - 1).min()
SS = {}
for U, L, I in PAIRS:
    S = sim_pair(U, L, I)
    if S: SS[U] = (S, L, I)
print(f'pairs with data: {len(SS)}  ' + ', '.join(f'{u}({l or "-"}/{i or "-"}, n={s["n"]})' for u, (s, l, i) in SS.items()), flush=True)
for cls in ('ALL', 'HV-pre'):
    print(f'\n######## day class: {cls}  (per-bar edge in bp per RTH hour-bar; random = circular-shift null; p = share of random >= observed)', flush=True)
    for name, lab in (('S0', 'S0 underlying itself, long/flat'), ('S1', 'S1 bull ETF only'), ('S2', 'S2 bull ETF / inverse ETF switch')):
        res = [(u, edge(S, name, cls, 500)) for u, (S, l, i) in SS.items()]; res = [(u, e) for u, e in res if e]
        if not res: continue
        W = np.array([e['w'] for _, e in res], float); obs = np.average([e['obs'] for _, e in res], weights=W); pooled = np.average(np.vstack([e['nul'] for _, e in res]), axis=0, weights=W)
        print(f'  {lab:36s} pooled edge {obs:+7.2f} bp/bar (random mean {pooled.mean():+.2f}) p={(pooled>=obs).mean():.3f} | pairs p<.10: {sum(e["p"]<.1 for _,e in res)}/{len(res)} | ' + ' '.join(f'{u}:{e["obs"]:+.1f}' for u, e in res), flush=True)
print('\n######## compounded over the whole sample (all days, open-to-open, no costs): total return / max drawdown', flush=True)
for u, (S, l, i) in SS.items():
    out = []
    for name in ('S0', 'S1', 'S2'):
        x, _ = strat(S, name)
        if x is not None: tr, dd = curve(x); out.append(f'{name} {tr:+.0%}/{dd:.0%}')
    for nm, lab in (('U', f'hold {u}'), ('L', f'hold {l}'), ('I', f'hold {i}')):
        if nm in S['rets']: tr, dd = curve(S['rets'][nm]); out.append(f'{lab} {tr:+.0%}/{dd:.0%}')
    print(f'  {u:8s} ' + ' | '.join(out), flush=True)
