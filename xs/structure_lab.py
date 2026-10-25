"""Structure lab (Oct 25 2026): market-structure ideas in the live engine (15m + 1h tracks, sticky 1h stop, LOCK5/0.5, live book via
signal_lab stage 2). Trades -> OUT/<tk>.parquet with these 'var' labels:
 vbar26 / vbar13   the vector on VOLUME bars (a new bar every 1/26 or 1/13 of the 20-day median daily volume; the 1h track uses 4x
                   bigger bars) instead of 15-minute bars - an activity clock (Lopez de Prado information-driven bars)
 rel_only          the vector on the stock's price relative to QQQ (SPY 15m history stops in Aug 2026)
 rel_gate          buy only when the stock's vector AND its relative-to-QQQ vector are both bull; exits as live
 avwap             vector entry; exit when price closes below the VWAP anchored at the swing low before the flip (replaces the vector exit)
 va_daily          auction-market value-area migration on daily volume profiles: bull when value (POC and value-area low) moves higher,
                   bear when POC and value-area high move lower; next-open entries/exits (daily track)
 orb_rv2 / orb_all opening-range breakout: first 15m bar green -> buy the 09:45 open, stop at that bar's low, out at the close;
                   rv2 = only when the first bar's volume is >= 2x its 20-day norm (stock in play, Zarattini-Aziz)
python3 xs/structure_lab.py BARS_DIR OUT_DIR WIDE_T [TK ...]"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])   # vec, load, agg, known, states, run, daily_track
BARS, OUT, WT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True); C2 = 2e-4

def aggv(H, freq, offset):
    x = H.set_index('dt')[['open', 'high', 'low', 'close', 'volume']]
    g = x.groupby([x.index.normalize(), pd.Grouper(freq=freq, origin='start_day', offset=offset)])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')).dropna()
    r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']; return r

def vol_bars(H, per_day):
    dv = H.groupby(H.dt.dt.normalize()).volume.sum(); thr = (dv.rolling(20, min_periods=5).median().shift(1) / per_day).reindex(H.dt.dt.normalize()).values
    o, h, l, c, v, t = H.open.values, H.high.values, H.low.values, H.close.values, H.volume.values.astype(float), H.dt.values
    out = []; cum = 0.0; bo = bh = bl = None
    for i in range(len(H)):
        if np.isnan(thr[i]): continue
        if bo is None: bo, bh, bl = o[i], h[i], l[i]
        bh, bl, cum = max(bh, h[i]), min(bl, l[i]), cum + v[i]
        if cum >= thr[i]: out.append((t[i] + np.timedelta64(15, 'm'), bo, bh, bl, c[i])); cum = 0.0; bo = None
    return pd.DataFrame(out, columns=['dt', 'open', 'high', 'low', 'close'])

def on_15m(H, r, st):                      # state of bars that end at r.dt, known on the 15m timeline
    return known(H, r.assign(dt=r.dt), np.asarray(st), '0min')

def avwap_state(B, reg):
    """vector bull flip -> long until a close below the VWAP anchored at the lowest low since the last bear flip; then bear until the next flip"""
    lo, hi, c, v = B.low.values, B.high.values, B.close.values, B.volume.values.astype(float); tp = (hi + lo + c) / 3
    st = np.zeros(len(B), int); s = -1; last_bear = 0; pv = vv = 0.0
    for i in range(1, len(B)):
        if reg[i] == -1 and reg[i - 1] != -1: last_bear = i
        if reg[i] == 1 and reg[i - 1] != 1:
            a = last_bear + int(np.argmin(lo[last_bear:i + 1])); pv, vv = float((tp[a:i + 1] * v[a:i + 1]).sum()), float(v[a:i + 1].sum()); s = 1
        elif s == 1:
            pv += tp[i] * v[i]; vv += v[i]
            if vv > 0 and c[i] < pv / vv: s = -1
        st[i] = s
    return st

def value_area(R):
    """daily POC / VAH / VAL from 15m RTH bars (each bar's volume spread evenly over its high-low)"""
    out = {}
    for d, g in R.groupby(R.dt.dt.normalize()):
        lo, hi = g.low.min(), g.high.max()
        if not hi > lo: continue
        edges = np.linspace(lo, hi, 61); prof = np.zeros(60); mid = (edges[:-1] + edges[1:]) / 2
        for l_, h_, v_ in zip(g.low.values, g.high.values, g.volume.values):
            k = (mid >= l_) & (mid <= h_); n = k.sum()
            if n: prof[k] += v_ / n
        poc = int(prof.argmax()); a = b = poc; tot = prof.sum(); acc = prof[poc]
        while acc < .7 * tot and (a > 0 or b < 59):
            up = prof[b + 1] if b < 59 else -1; dn = prof[a - 1] if a > 0 else -1
            if up >= dn: b += 1; acc += prof[b]
            else: a -= 1; acc += prof[a]
        out[d] = (mid[poc], mid[b], mid[a], g.open.iloc[0], g.dt.iloc[0])
    return pd.DataFrame.from_dict(out, orient='index', columns=['poc', 'vah', 'val', 'open', 'dt0']).sort_index()

def va_trades(R):
    V = value_area(R); s = 0; st = []
    for i in range(len(V)):
        if i:
            p, q = V.iloc[i], V.iloc[i - 1]
            if p.poc > q.poc and p.val > q.val: s = 1
            elif p.poc < q.poc and p.vah < q.vah: s = -1
        st.append(s)
    T = []; pos = False
    for i in range(1, len(V) - 1):
        if not pos and st[i] == 1 and st[i - 1] != 1: pos, e = True, i + 1
        elif pos and st[i] == -1: T.append((V.dt0.iloc[e], V.dt0.iloc[i + 1], V.open.iloc[i + 1] / V.open.iloc[e] - 1 - C2)); pos = False
    return T

def orb(H, need_rv):
    R = H[H.m >= 570]; T = []
    first = R[R.m == 570].set_index(R[R.m == 570].dt.dt.normalize())
    rv = first.volume / first.volume.shift(1).rolling(20, min_periods=10).mean()
    for d, g in R.groupby(R.dt.dt.normalize()):
        if d not in first.index or len(g) < 3: continue
        f = first.loc[d]
        if not f.close > f.open or (need_rv and not rv.get(d, 0) >= 2): continue
        rest = g[g.m >= 585]; px = rest.open.iloc[0]; stop = f.low; xp, xt = rest.close.iloc[-1], rest.dt.iloc[-1] + pd.Timedelta('15min')
        for bl, bo, bt in zip(rest.low.values, rest.open.values, rest.dt):
            if bl <= stop: xp, xt = min(stop, bo), bt + pd.Timedelta('15min'); break
        T.append((rest.dt.iloc[0], xt, xp / px - 1 - C2))
    return T

W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
Q = load(fs['QQQ']).set_index('dt').close
for tk in (sys.argv[4:] or CORE):
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []
    def add(var, T, h1): rows.extend((tk, var, h1, 'trade') + t for t in T)
    reg, s1 = states(H, vec)
    for nm, k in (('vbar26', 26), ('vbar13', 13)):                                   # 1 volume bars
        b15, b1 = vol_bars(H, k), vol_bars(H, k / 4); r15, r1 = on_15m(H, b15, vec(b15)), on_15m(H, b1, vec(b1))
        for h1t in (False, True): add(nm, run(H, r15, r1, h1t)[0], h1t)
    q = Q.reindex(H.dt.values).ffill().values                                        # 2 relative to QQQ
    Rl = H.assign(open=H.open / q, high=H.high / q, low=H.low / q, close=H.close / q).dropna(subset=['close'])
    if len(Rl) > 5000:
        rr, rs1 = states(Rl, vec); rr = pd.Series(rr, index=Rl.index).reindex(H.index).fillna(0).values; rs1 = pd.Series(rs1, index=Rl.index).reindex(H.index).fillna(0).values
        for h1t in (False, True): add('rel_only', run(H, rr, rs1, h1t)[0], h1t)
        g15 = np.where((reg == 1) & (rr == 1), 1, np.where(reg == -1, -1, 0)); g1 = np.where((s1 == 1) & (rs1 == 1), 1, np.where(s1 == -1, -1, 0))
        for h1t in (False, True): add('rel_gate', run(H, g15, g1, h1t)[0], h1t)
    a15 = avwap_state(H, reg)                                                        # 3 anchored VWAP exits
    hb = aggv(H, '1h', '9h30min'); hreg = vec(hb); a1 = known(H, hb, avwap_state(hb, hreg), '1h')
    for h1t in (False, True): add('avwap', run(H, a15, a1, h1t)[0], h1t)
    add('va_daily', va_trades(H[H.m >= 570]), True)                                  # 4 value-area migration
    add('orb_rv2', orb(H, True), False); add('orb_all', orb(H, False), False)          # 5 opening-range breakout
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
