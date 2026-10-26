exec(open('/tmp/claude-0/ml/rangez.py').read().split("\nP = [(np.datetime64")[0].replace("print(f'trades", "#").replace("print(f'\\n{c} quintiles", "#").replace("print(g.round(3)", "#"))
LO = {}
for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/*_15m_*.parquet'):
    tk = os.path.basename(f).split('_15m_')[0]
    if tk not in F: continue
    B = pd.read_parquet(f, columns=['datetime', 'low', 'open']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); LO[tk] = (B.dt.values, B.low.values, B.open.values)
def limit_fill(Z, k, nbars, which):
    """stretched 15m entries (used > 1.07): instead of buying the open, rest a limit k expected-ranges under it for nbars bars (same day);
    filled -> same exit, return from the fill; not filled -> no trade (slot stays free)"""
    out = Z.copy(); keep = np.ones(len(Z), bool); nf = nm = 0
    for i, (tk, e, x, r, p0, used, h1) in enumerate(zip(Z.tk, Z.entry.values, Z.exit.values, Z.ret.values, Z.px0.values, Z.used.values, Z.h1.values)):
        if not (used > 1.07) or (which == '15m' and h1) or (which == '1h' and not h1): continue
        d, lo, op = LO[tk]; k0 = np.searchsorted(d, e); ex = F[tk][5][k0] if k0 < len(F[tk][5]) else np.nan
        lim = p0 * (1 - k * ex); day = d[k0].astype('datetime64[D]'); filled = None
        for j in range(k0, min(k0 + nbars, len(d))):
            if d[j].astype('datetime64[D]') != day or d[j] >= x: break
            if lo[j] <= lim: filled = (d[j], min(lim, op[j])); break
        nm += 1
        if filled is None: keep[i] = False; continue
        nf += 1; xpx = p0 * (1 + r + 2e-4); out.iloc[i, out.columns.get_loc('ret')] = xpx / filled[1] - 1 - 2e-4; out.iloc[i, out.columns.get_loc('entry')] = filled[0] + np.timedelta64(15, 'm')
    return out[keep], nf, nm
T['px0'] = [F[t][1][min(np.searchsorted(F[t][0], e), len(F[t][0]) - 1)] for t, e in zip(T.tk, T.entry.values)]
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<64}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
print(f'{"stretched entries: limit order instead of buying the open":<66}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live (buy the open)', T)
for which in ('15m', 'both'):
    for k, nb in ((.15, 4), (.25, 4), (.25, 8), (.4, 8), (.25, 26)):
        Z, nf, nm = limit_fill(T, k, nb, which); s = Z[Z.used > 1.07] if which == 'both' else Z[(Z.used > 1.07) & ~Z.h1]
        show(f'{which}: limit {k} exp. ranges under, {nb} bars ({nf}/{nm} filled, avg {s.ret.mean():+.2%})', Z)
