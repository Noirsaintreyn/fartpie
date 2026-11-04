"""Nov 4 2026: published-style trend signals as REPLACEMENTS for the vector. DECLARED BEFORE RESULTS. Same universe (ratio >= 1.5 & sp), 54 names, same bars, long-only, fills at RTH bar opens
(entries before 14:00), 2 bp cost, signal-off exit at the next RTH open, same 6-slot book / ratio tie-break / 8% breaker, 40 matched runs. NOTE live also has lock/sticky exits; these do not (they use their own exits).
 R1 1h EMA20 > EMA50 (cross) | R2 1h Supertrend(10,3) | R3 daily EMA10 > EMA20 | R4 daily Donchian-20 breakout (close > prior 20d high), exit: close < highest daily close since entry - 3 ATR14 (Zarattini-style trailing)
 R5 Donchian ensemble (10/20/40/80): fraction of channels with close above the channel mid; enter >= .75, exit < .5  (adapted from Zarattini-Pagani-Barbon 2025, crypto; no vol sizing here)"""
exec(open('xs/ema_common.py').read())
def gen(tk, kind):
    B = BAR[tk]; n = len(B['o']); o, m, t = B['o'], B['m'], B['t']; out = []; pos = False; armed = False; hc = 0
    if kind == 'R1': S = B['x1']
    elif kind == 'R2': S = (B['st1'] == 1).astype(float); S[np.isnan(B['st1'])] = np.nan
    elif kind == 'R3': S = (B['d_e10'] > B['d_e20']).astype(float); S[np.isnan(B['d_e10'])] = np.nan
    elif kind == 'R4': S = (B['d_c'] > B['d_hi20']).astype(float); S[np.isnan(B['d_hi20'])] = np.nan
    else:
        sc = np.zeros(n); ok = np.ones(n, bool)
        for k in (10, 20, 40, 80):
            mid = (B[f'd_hi{k}'] + B[f'd_lo{k}']) / 2; sc += (B['d_c'] > mid); ok &= ~np.isnan(mid)
        sc /= 4; S = np.where(ok, sc, np.nan)
    for i in range(1, n):
        s = S[i - 1]
        if np.isnan(s): continue
        rth = 570 <= m[i] < 956
        if not pos:
            if kind == 'R5': on, off = s >= .75, s < .5
            else: on, off = s == 1, s == 0
            if off or (kind != 'R5' and s == 0): armed = True
            if kind == 'R5' and s < .75: armed = True
            if armed and on and rth and m[i] < 840: pos, e0, armed = True, i, False; hc = B['d_c'][i - 1]
        else:
            if not rth: continue
            if kind == 'R4': hc = max(hc, B['d_c'][i - 1]); ex = B['d_c'][i - 1] < hc - 3 * B['d_atr'][i - 1]
            elif kind == 'R5': ex = s < .5
            else: ex = s == 0
            if ex: out.append((tk, t[e0], t[i], o[i] / o[e0] - 1 - 2e-4)); pos = False; armed = (kind != 'R4') or True
    if pos: out.append((tk, t[e0], t[n - 1], o[n - 1] / o[e0] - 1 - 2e-4))
    return out
def mk(kind):
    S = pd.DataFrame([r for tk in BAR for r in gen(tk, kind)], columns=['tk', 'entry', 'exit', 'ret']); S['entry'] = pd.to_datetime(S.entry); S['exit'] = pd.to_datetime(S.exit); S = S[S.entry >= '2019-01-01'].copy(); S['d'] = S.entry.dt.normalize()
    S = pd.merge_asof(S.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)].sort_values('entry').reset_index(drop=True); S['h1'] = False; return S
B0 = ev(XL); print('40 matched runs, 2019-21 / 2022-26'); line('L live (vector + its exits)', B0)
for kind, nm in (('R1', '1h EMA20/50 cross'), ('R2', '1h Supertrend 10,3'), ('R3', 'daily EMA10/20 cross'), ('R4', 'daily Donchian20 + 3ATR trail'), ('R5', 'Donchian ensemble 10-80')):
    S = mk(kind); a = S[S.entry < '2022-01-01']; b = S[S.entry >= '2022-01-01']
    print(f'  {kind} {nm}: trades {len(S)} (19-21 {len(a)}, 22-26 {len(b)}), avg {a.ret.mean()*100:+.2f}% / {b.ret.mean()*100:+.2f}%, win {a.ret.gt(0).mean():.0%} / {b.ret.gt(0).mean():.0%}, runners>=15% {(S.ret >= .15).sum()} (live {(XL.ret >= .15).sum()}), avg hold d {((S.exit - S.entry).dt.total_seconds() / 86400).mean():.1f}', flush=True)
    line(f'{kind} {nm}', ev(S), B0)
