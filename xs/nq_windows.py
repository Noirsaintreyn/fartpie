"""Nov 4 2026: realistic months, time windows, ES, RSI / z-score. DECLARED BEFORE RESULTS. Never held overnight: every trade is flat by its window's end.
Base trade: 15m vector flip (state from premarket + RTH bars), up to 2 a day, fixed target / stop, 1 pt cost (NQ) / 0.5 pt (ES); QQQ proxy -> NQ pts, SPY proxy -> ES pts.
REALISTIC MONTH: every calendar month is a fresh 50K eval: day P&L floored at -$1,000 (daily loss), stop at the $2,000 trailing max loss (end of day) = account lost, stop when passed
(+$3,000 with best day <= 50%). Outcome per month: passed / lost / neither, and the month's P&L under those rules (so the worst month is what the account can actually lose).
WINDOWS (NQ, 1 NQ 75/50): signal bars closing in  PRE 08:00-09:15 (flat 09:30; QQQ premarket is thin, read with care) | AM 09:30-11:30 (flat 12:00) | MID 11:30-13:30 (flat 14:00)
         | PM 13:30-15:15 (flat 15:55)  -- any day and negative-gamma days.
ES: SPY 15m bars 2018 - Aug 2026, NY AM, 1 ES at 30/20 ES pts ($1,500 / $1,000), path on 15m bars (stop first if both).
RSI / Z (NQ AM): filter longs only when RSI14(15m) < 70 and shorts only when > 30 | filter |z20| < 2 (z = (close - 20-bar mean) / 20-bar std) |
         trigger: RSI14 crosses back above 30 (long) / below 70 (short) in the AM window (mean reversion), same 75/50."""
_s = open('xs/nq_month.py').read(); exec(_s[:_s.index("SETS = {")])
def cands(B, reg, lo, hi, gneg=None):
    out = []; r = reg; me = B.me.values if 'me' in B else None
    for k in range(2, len(B)):
        if r[k] == r[k - 1] or r[k - 1] == 0 or not (lo < B.me.iloc[k] <= hi): continue
        day = B.dt.iloc[k].normalize(); j = np.searchsorted(t5, np.datetime64(B.end.iloc[k]))
        if j < len(H5) and d5[j] == np.datetime64(day): out.append((day, int(j), int(r[k])))
    C = pd.DataFrame(out, columns=['day', 'j', 'dir']); C['k'] = NQs.reindex(pd.DatetimeIndex(C.day) - pd.Timedelta(days=1), method='ffill').values / o5[C.j.values]
    C['neg'] = [bool(len(G[G.index < d]) and G[G.index < d].iloc[-1] < 0) for d in C.day]; return C
def run_w(Cc, T, S, per_day, flat_m, cost=1.0):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for _, c in g.iterrows():
            if n >= per_day or c.j <= busy: continue
            e = o5[c.j]; d = c.dir; kk = c.k; res = None
            for q in range(c.j, len(H5)):
                if d5[q] != d5[c.j] or m5[q] >= flat_m: res = ((c5[q - 1] - e) * d * kk, q - 1); break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo_ <= -S: res = (-S, q); break
                if hi_ >= T: res = (T, q); break
            if res: out.append((day, res[0] - cost)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net'])
def months(R, usd, a='2022-01-01', target=3000, mll=2000, dll=1000):
    v = (R.net * usd).groupby(R.day).sum().reindex(days).fillna(0.0); v = v[v.index >= a]; out = []
    for mo, x in v.groupby(v.index.to_period('M')):
        bal, peak, floor, best, res = 0.0, 0.0, -mll, 0.0, 'neither'
        for day in x.values:
            day = max(day, -dll); bal += day; best = max(best, day)
            if bal <= floor: res = 'lost'; break
            if bal >= target and best <= .5 * bal: res = 'passed'; break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        out.append((str(mo), res, bal))
    M_ = pd.DataFrame(out, columns=['month', 'res', 'pnl']); return M_
def line(nm, R, usd=20):
    if not len(R): print(f'  {nm:<44} no trades'); return
    M_ = months(R, usd); M6 = M_[M_.month >= '2026-01']; post = R[R.day >= P0]; a, b = post[post.day < '2024-01-01'], post[post.day >= '2024-01-01']
    print(f'  {nm:<44} trades/mo {len(post) / max(len(M_), 1):4.1f} win {post.net.gt(0).mean():.0%} avg {post.net.mean():+5.1f} (22-23 {a.net.mean():+5.1f} / 24-26 {b.net.mean():+5.1f}) | months: passed {M_.res.eq("passed").mean():.0%} lost {M_.res.eq("lost").mean():.0%} neither {M_.res.eq("neither").mean():.0%}, P&L mean {M_.pnl.mean():+,.0f} worst {M_.pnl.min():+,.0f} | 2026: passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}', flush=True)
print('REALISTIC MONTHS 2022 - Aug 2026 (fresh eval each calendar month)', flush=True)
print('\nWINDOWS, NQ 1 NQ 75/50', flush=True)
for wn, lo, hi, fl in (('PRE 08:00-09:15', 480, 555, 570), ('AM 09:30-11:30', 570, 690, 720), ('MID 11:30-13:30', 690, 810, 840), ('PM 13:30-15:15', 810, 915, 955)):
    C = cands(H15, H15.reg.values, lo, hi); line(f'{wn} any day', run_w(C, 75, 50, 2, fl)); line(f'{wn} negative gamma', run_w(C[C.neg], 75, 50, 2, fl))
print('\nRSI / Z-SCORE, NQ AM, any day', flush=True)
cl = pd.Series(H15.close.values); dlt = cl.diff(); up = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
H15['rsi'] = (100 - 100 / (1 + up / dn)).values; H15['z20'] = ((cl - cl.rolling(20).mean()) / cl.rolling(20).std()).values
CA = cands(H15, H15.reg.values, 570, 690); endmap = pd.Series(np.arange(len(H15)), index=pd.DatetimeIndex(H15.end))
kk_ = endmap.reindex(pd.DatetimeIndex(t5[CA.j.values])).values; CA['rsi'] = H15.rsi.values[kk_.astype(int)]; CA['z20'] = H15.z20.values[kk_.astype(int)]
line('base (AM any day)', run_w(CA, 75, 50, 2, 720))
line('RSI filter (no longs > 70, no shorts < 30)', run_w(CA[((CA.dir > 0) & (CA.rsi < 70)) | ((CA.dir < 0) & (CA.rsi > 30))], 75, 50, 2, 720))
line('RSI with the trend (longs > 50, shorts < 50)', run_w(CA[((CA.dir > 0) & (CA.rsi > 50)) | ((CA.dir < 0) & (CA.rsi < 50))], 75, 50, 2, 720))
line('|z20| < 2 filter', run_w(CA[CA.z20.abs() < 2], 75, 50, 2, 720))
rs = H15.rsi.values; MR = []
for k in range(1, len(H15)):
    if not (570 < H15.me.iloc[k] <= 690): continue
    d_ = 1 if (rs[k - 1] < 30 <= rs[k]) else (-1 if (rs[k - 1] > 70 >= rs[k]) else 0)
    if not d_: continue
    day = H15.dt.iloc[k].normalize(); j = np.searchsorted(t5, np.datetime64(H15.end.iloc[k]))
    if j < len(H5) and d5[j] == np.datetime64(day): MR.append((day, int(j), d_))
MR = pd.DataFrame(MR, columns=['day', 'j', 'dir']); MR['k'] = NQs.reindex(pd.DatetimeIndex(MR.day) - pd.Timedelta(days=1), method='ffill').values / o5[MR.j.values]
line('RSI 30/70 reversal trigger (mean reversion)', run_w(MR, 75, 50, 2, 720))
# ---------------------------------------------------------------- ES via SPY 15m
print('\nES (SPY 15m proxy), NY AM, 1 ES 30/20 ES pts ($50/pt)', flush=True)
S15 = load(glob.glob(f'{ROOT}/lse_ohlcv_cache/SPY_15m_*.parquet')[0]).reset_index(drop=True); S15['reg'] = np.asarray(vec(S15[['open', 'high', 'low', 'close']]))
S15['end'] = S15.dt + pd.Timedelta('15min'); S15['me'] = S15.end.dt.hour * 60 + S15.end.dt.minute
ESd = yf.download('ES=F', start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].squeeze(); ESd.index = ESd.index.normalize(); ESd = ESd.sort_index()
so, sh, sl, sc, sm, sd = S15.open.values, S15.high.values, S15.low.values, S15.close.values, S15.m.values, S15.dt.dt.normalize().values
def es_run(neg_only, T=30, S=20, per_day=2, cost=.5):
    rg = S15.reg.values; out = []; cur_day = None; n = 0; busy = -1
    for k in range(2, len(S15) - 1):
        if rg[k] == rg[k - 1] or rg[k - 1] == 0 or not (570 < S15.me.iloc[k] <= 690): continue
        day = pd.Timestamp(sd[k]); j = k + 1
        if sd[j] != sd[k]: continue
        if neg_only:
            g = G[G.index < day]
            if not len(g) or g.iloc[-1] >= 0: continue
        if day != cur_day: cur_day, n, busy = day, 0, -1
        if n >= per_day or j <= busy: continue
        lvl = ESd.reindex([day - pd.Timedelta(days=1)], method='ffill').values[0]; kk = lvl / so[j]; e = so[j]; d = int(rg[k]); res = None
        for q in range(j, len(S15)):
            if sd[q] != sd[j] or sm[q] >= 720: res = ((sc[q - 1] - e) * d * kk, q - 1); break
            lo_ = ((sl[q] if d > 0 else sh[q]) - e) * d * kk; hi_ = ((sh[q] if d > 0 else sl[q]) - e) * d * kk
            if lo_ <= -S: res = (-S, q); break
            if hi_ >= T: res = (T, q); break
        if res: out.append((day, res[0] - cost)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net'])
RE = es_run(False); RN = es_run(True); line('ES AM any day', RE, 50); line('ES AM negative gamma', RN, 50)
RA = run_w(CA, 75, 50, 2, 720); both = pd.concat([RA.assign(net=RA.net * 20), RE.assign(net=RE.net * 50)]); both['net'] = both.net / 20
line('NQ + ES together (1 NQ + 1 ES, same account)', both)
