"""Nov 4 2026: current model vs the 3-universe book, and slots x ratio-bar sweep. DECLARED BEFORE RESULTS. Usage: python3 xs/uni_slots_ratio.py live|union
Books: LIVE = S&P (sp flag) ratio >= T; UNION = LIVE + Nasdaq-100 (>= T) + Dow (>= max(T, 1.0)).  Slots N in {6, 8, 10} at T = 1.5; T in {2.0, 2.5, 3.0} at N = 6; plus (N=8, T=2.0) and (N=10, T=2.0).
All vs the live baseline (LIVE, 6 slots, 1.5): 40 matched runs, per-period CAGR/DD/ret-DD, beats-live share, full 2019-2026 CAGR/DD and the median value of $280,000."""
import sys
which = sys.argv[1]; sys.argv = sys.argv[:1]
_s = open('xs/universe_test.py').read(); exec(_s[:_s.index("B0 = ev(W[W.sp]")].replace(" & (W.ratio >= 1.5)].copy()", "].copy()").replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
W = W[W.ratio.notna()].reset_index(drop=True)
def cat(*fs): return pd.concat(fs).drop_duplicates(['tk', 'entry', 'h1']).sort_values('entry').reset_index(drop=True)
def book(kind, T):
    L = W[W.sp & (W.ratio >= T)]
    return L.reset_index(drop=True) if kind == 'live' else cat(L, W[W.ndx & (W.ratio >= T)], W[W.dow & (W.ratio >= max(T, 1.0))])
body = src[src.index('def sim('):src.index('def ev(')]
def mksim(N): ns = {}; exec(body.replace('>= 6', f'>= {N}').replace('/ 6', f'/ {N}').replace('def sim(', 'def simN('), globals(), ns); return ns['simN']
def full(s):
    eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); e = np.exp(eq - eq.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25; return e.iloc[-1] ** (1 / yr) - 1, (e / e.cummax() - 1).min(), e.iloc[-1]
def evN(Y, N):
    sm = mksim(N); HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); res = []; fl = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl; s = sm(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m]); res.append(stat2(s)); fl.append(full(s))
    return np.array(res), np.array(fl)
def show(nm, Y, N, base):
    r, f = evN(Y, N); m = np.median(r, axis=0); fm = np.median(f, axis=0); beat = ''
    if base is not None: beat = f'  beats live {(r[:, 2] > base[0][:, 2]).mean():.0%} / {(r[:, 5] > base[0][:, 5]).mean():.0%}'
    print(f'  {nm:<34} {len(Y):>6} tr | CAGR {m[0]:+.1%} / {m[3]:+.1%}  DD {m[1]:.0%} / {m[4]:.0%}  ret/DD {m[2]:.2f} / {m[5]:.2f}{beat} | full CAGR {fm[0]:+.1%} DD {fm[1]:.0%}  $280k -> {280000 * fm[2]:,.0f}', flush=True); return r, f
base = evN(book('live', 1.5), 6)
if which == 'live': show('LIVE 6 slots, ratio>=1.5 (current)', book('live', 1.5), 6, base)
print(f'== {which.upper()} book (CAGR 2019-21 / 2022-26 | median of 40 matched runs)')
for N, T in ((6, 1.5), (8, 1.5), (10, 1.5), (6, 2.0), (6, 2.5), (6, 3.0), (8, 2.0), (10, 2.0)):
    show(f'{which} N={N} ratio>={T}', book(which, T), N, base)
