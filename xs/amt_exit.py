"""Nov 1 2026: auction 'end of move' exits (rules fixed in advance). While a trade is up >= 2%:
 excess    a 15m bar makes a new high since entry, then rejects: upper wick >= 50% of the bar and volume >= 2x its 130-bar median -> exit next open
 absorb    a new high on negative delta proxy (volume x close location < 0) with volume >= 2x median -> exit next open
40 matched runs vs live."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True)
NEW = {m: (Ls.exit.copy(), Ls.ret.copy()) for m in ('excess', 'absorb')}
for tk, g in Ls.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    mm = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(mm >= 240) & (mm < 960) & (H.volume > 0)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    t = H.dt.values; o, h, l, c, v = (H[k].values for k in ('open', 'high', 'low', 'close', 'volume'))
    rg = np.where(h > l, h - l, np.nan); uw = (h - np.maximum(o, c)) / rg; rv = v / pd.Series(v).rolling(130, min_periods=40).median().shift(1).values
    dl = v * ((c - l) - (h - c)) / rg
    for i, e, x in zip(g.index, g.entry.values, g.exit.values):
        a = np.searchsorted(t, e); b = np.searchsorted(t, x)
        if a >= len(t) or b <= a + 1: continue
        px = o[a]; mx = h[a]
        for j in range(a + 1, min(b, len(t) - 1)):
            newhi = h[j] > mx; mx = max(mx, h[j])
            if not newhi or c[j] / px - 1 < .02: continue
            for mode, hit in (('excess', uw[j] >= .5 and rv[j] >= 2), ('absorb', dl[j] < 0 and rv[j] >= 2)):
                if hit and NEW[mode][0][i] == Ls.exit[i]: NEW[mode][0][i] = pd.Timestamp(t[j + 1]); NEW[mode][1][i] = o[j + 1] / px - 1 - 2e-4
    print(tk, end=' ', flush=True)
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<34} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
rng = np.random.default_rng(67); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(Z): return np.array([[v_ for cd in stat(dret(simw(Z[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
print(); B0 = ev(Ls); line('live', B0)
for mode in NEW:
    Z = Ls.copy(); Z['exit'], Z['ret'] = NEW[mode]; ch = (Z.exit != Ls.exit)
    print(f'  {mode}: changes {ch.mean():.1%} of trades; changed trades avg {Ls.ret[ch].mean():+.2%} -> {Z.ret[ch].mean():+.2%}')
    line(f'exit on {mode}', ev(Z), B0)
