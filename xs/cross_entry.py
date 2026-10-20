"""Buy at the intrabar cross of the vector trigger (RTH bars only) instead of the next open after the 15m close.
  confirmed (bar closes flipped bull): same trade as live, entry price = max(trigger, bar open) instead of the next open
  fakeout (touches trigger, closes still bear): bought at the trigger, sold at that bar's close (15-min trade)
Live model otherwise (premarket+RTH vector, 15m + 1h tracks, sticky, LOCK5/0.5). Stage 1 TK..: -> xs/ext/crossx; stage 2: sims."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
C = 2e-4
if sys.argv[1] == '1':
    exec(open('xs/ah_sticky.py').read().split("if sys.argv[1] == '1':")[0])        # bars(), run()
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = bars(fs[tk], 960)
        _, lv, atr, _ = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); trig = np.asarray(lv, float) + 0.35 * np.asarray(atr, float)
        reg, o, hi, c, d, m = H.reg.values, H.open.values, H.high.values, H.close.values, H.dt.values, H.m.values
        rows = []
        for h1t in (False, True):
            for e_t, x_t, r in run(H, h1t, True):
                rows.append((tk, 'live', h1t, e_t, x_t, r))
                if h1t: rows.append((tk, 'cross', h1t, e_t, x_t, r)); continue
                j = int(np.searchsorted(d, e_t)); f = j - 1                    # f = flip bar (closed bull), entry at j's open in live
                if f >= 1 and 570 <= m[f] < 960 and d[f].astype('datetime64[D]') == d[j].astype('datetime64[D]') and reg[f - 1] == -1 and hi[f] >= trig[f - 1]:
                    px = max(trig[f - 1], o[f]); rows.append((tk, 'cross', h1t, d[f], x_t, (1 + r + C) * o[j] / px - 1 - C))
                else: rows.append((tk, 'cross', h1t, e_t, x_t, r))
        for f in range(1, len(H)):                                             # fakeouts: touched the trigger in RTH, closed still bear
            if 570 <= m[f] < 960 and reg[f - 1] == -1 and reg[f] == -1 and hi[f] >= trig[f - 1]:
                px = max(trig[f - 1], o[f]); rows.append((tk, 'cross', False, d[f], d[f] + np.timedelta64(15, 'm'), c[f] / px - 1 - 2 * C))
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(f'xs/ext/crossx/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/crossx/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
    D = pd.read_parquet('etf_daily/SPY.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); cc = D.drop_duplicates('d').set_index('d').adjclose
    T['SPY50'] = (cc > cc.rolling(50).mean()).shift(1).reindex(T.entry.dt.normalize(), method='ffill').fillna(True).values.astype(bool)
    X = T[T['var'] == 'cross']; fk = X.exit - X.entry == pd.Timedelta('15min')
    L = T[(T['var'] == 'live') & ~T.h1]; Xc = X[~fk & ~X.h1]
    print(f"15m-track flips: {len(L):,}   cross fakeouts: {fk.sum():,} (avg {X[fk].ret.mean():+.2%}, {(X[fk].ret > 0).mean():.0%} green)")
    print(f"avg confirmed trade: wait for close {L.ret.mean():+.2%}  vs  buy the cross {Xc.ret.mean():+.2%}")
    def sim(Z, n=6):
        Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
        for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
            op = [o for o in op if o > e]
            if len(op) < n: op.append(x); ev.append((x, r / n))
        return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    def st(s):
        e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
        return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
    P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))
    print(f'\n{"6 slots, ratio >= 2.5, sticky":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
    def show(nm, Z):
        res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
    for v, nm in (('live', 'wait for the 15m close (live)'), ('cross', 'buy at the cross')):
        Z = T[T['var'] == v]; show(nm, Z); show('   + SPY > 50d', Z[Z.SPY50])
