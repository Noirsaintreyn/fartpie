"""Nov 4 2026: Keltner channels as a trend signal (replacement test, same universe/book/harness as xs/ema_replace.py). DECLARED BEFORE RESULTS.
 K1 1h: EMA20 +/- 2 x ATR10 (1h): long when close > upper band, exit when close < EMA20 (mid) | K2 15m: same on 15m bars | K3 1h with exit at the LOWER band (looser)
Fills at RTH bar opens (entries before 14:00), 2 bp, 6 slots, 40 matched runs."""
exec(open('xs/ema_common.py').read())
def kstate(B, tf, exit_lower=False):
    if tf == '15m': c, h, l = B['c'], B['h'], B['l']; idx = None
    t = B['t']; x = pd.DataFrame({'o': B['o'], 'h': B['h'], 'l': B['l'], 'c': B['c']}, index=pd.DatetimeIndex(t))
    if tf == '1h':
        r = x.resample('1h', offset='30min').agg(dict(o='first', h='max', l='min', c='last')).dropna(); r['known'] = r.index + pd.Timedelta('1h')
    else: r = x.copy(); r['known'] = r.index + pd.Timedelta('15min')
    c, h, l = r.c.values, r.h.values, r.l.values; pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); atr = pd.Series(tr).ewm(alpha=1 / 10, adjust=False).mean().values
    mid = ema_(c, 20); up, lo = mid + 2 * atr, mid - 2 * atr; st = np.zeros(len(r)); s = 0
    for i in range(30, len(r)):
        if s == 0 and c[i] > up[i]: s = 1
        elif s == 1 and c[i] < (lo[i] if exit_lower else mid[i]): s = 0
        st[i] = s
    r['st'] = st; K = pd.merge_asof(pd.DataFrame({'q': t + np.timedelta64(15, 'm')}), r[['known', 'st']].sort_values('known'), left_on='q', right_on='known', direction='backward'); return K.st.values
def gen(tk, tf, ex):
    B = BAR[tk]; S = kstate(B, tf, ex); n = len(B['o']); o, m, t = B['o'], B['m'], B['t']; out = []; pos = False; armed = False
    for i in range(1, n):
        s = S[i - 1]
        if np.isnan(s): continue
        rth = 570 <= m[i] < 956
        if not pos:
            if s == 0: armed = True
            if armed and s == 1 and rth and m[i] < 840: pos, e0, armed = True, i, False
        elif rth and s == 0: out.append((tk, t[e0], t[i], o[i] / o[e0] - 1 - 2e-4)); pos = False
    if pos: out.append((tk, t[e0], t[n - 1], o[n - 1] / o[e0] - 1 - 2e-4))
    return out
def mk(tf, ex):
    S = pd.DataFrame([r for tk in BAR for r in gen(tk, tf, ex)], columns=['tk', 'entry', 'exit', 'ret']); S['entry'] = pd.to_datetime(S.entry); S['exit'] = pd.to_datetime(S.exit); S = S[S.entry >= '2019-01-01'].copy(); S['d'] = S.entry.dt.normalize()
    S = pd.merge_asof(S.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)].sort_values('entry').reset_index(drop=True); S['h1'] = False; return S
B0 = ev(XL); print('40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
for nm, tf, ex in (('K1 1h Keltner, exit at mid', '1h', False), ('K2 15m Keltner, exit at mid', '15m', False), ('K3 1h Keltner, exit at lower band', '1h', True)):
    S = mk(tf, ex); a = S[S.entry < '2022-01-01']; b = S[S.entry >= '2022-01-01']
    print(f'  {nm}: trades {len(S)}, avg {a.ret.mean()*100:+.2f}% / {b.ret.mean()*100:+.2f}%, win {a.ret.gt(0).mean():.0%} / {b.ret.gt(0).mean():.0%}, runners>=15% {(S.ret >= .15).sum()} (live {(XL.ret >= .15).sum()}), avg hold d {((S.exit - S.entry).dt.total_seconds() / 86400).mean():.1f}', flush=True); line(nm, ev(S), B0)
