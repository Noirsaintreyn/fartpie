"""Nov 8 2026: MEAN REVERSION WHERE THE VECTOR FAILS - one ideology: dealers decide the day type (Greeks), the volume profile decides where, the vector flip decides when.
On trend days (negative gamma, low DEX) follow the flip; on POSITIVE-gamma days a flip / break into an edge is usually a failed auction -> fade it back to fair value.
DECLARED BEFORE RESULTS. Real NQ futures 5m bars, RTH; regime from QQQ GEX at the prior close (own file 2018+); 1 contract, 1 pt cost; ADR = 20-day average RTH range.
  MR1 FADE THE STRETCHED FLIP   a 15m vector flip whose entry is > 1 VWAP sd past VWAP in the flip's direction -> trade the other way at the next 5m open;
                                target = VWAP at the signal, stop = 0.10 ADR, out after 90 minutes or at 15:55; <= 2 a day
  MR2 FAILED AUCTION            price trades beyond the prior-day high / VAH (or low / VAL), then a 15m bar closes back inside the prior value area before 14:00 ->
                                fade toward the prior POC; stop = today's extreme on that side + 0.03 ADR; one a day per side; out at 15:55
  MR3 GAP FILL                  RTH opens 0.1 - 0.5 ADR away from the prior close and inside the prior day's range -> fade toward the prior close at 09:35;
                                stop 0.15 ADR; out at 12:00
Each engine on every day, and split by regime (negative / positive GEX; GEX top third by the greeks-file percentile 2014+). By period 2014-17 / 2018-21 / 2022-25."""
_e = open('/home/claude/fartpie/xs/dte_engines.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
import sys; sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
G1o = pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); G1o.index = G1o.index.normalize()
g2 = (G1o.call_gex - G1o.put_gex); g2p = g2.rolling(252, min_periods=120).rank(pct=True); DY['g2p'] = g2p.reindex(DY.index, method='ffill').shift(1)
# 15m bars from 5m, vector on 15m (RTH, carried across days)
H['b15'] = H.dt.dt.floor('15min'); B15 = H.groupby('b15').agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), day=('day', 'first'))
B15['reg'] = np.asarray(vec(B15[['open', 'high', 'low', 'close']])); B15['flip'] = B15.reg != B15.reg.shift(1); B15['end'] = B15.index + pd.Timedelta(minutes=15)
tpos5 = {t: i for i, t in enumerate(H.dt.values)}
def mr1():
    rows = []; cnt = {}; busy = {}
    for t, r_ in B15[B15.flip].iterrows():
        i = tpos5.get(np.datetime64(r_.end, 'ns'))
        if i is None or dd[i] != np.datetime64(r_.day) or not ok(r_.day) or m[i] > 900 or m[i] < 585: continue
        d = r_.day; fl = int(r_.reg)
        if cnt.get(d, 0) >= 2 or i <= busy.get(d, -1) or vsd[i - 1] <= 0: continue
        z = (o[i] - vw[i - 1]) / vsd[i - 1] * fl
        if z < 1: continue
        dr = -fl; e = o[i]; ix = gi[d]; j = int(np.where(ix == i)[0][0]); T = vw[i - 1]; S = e - dr * .10 * DY.adr[d]
        if (T - e) * dr <= 0: continue
        pts = walk(ix, j, dr, e, T, S, maxb=18); rows.append((d, m[i], dr, pts - 1.0)); cnt[d] = cnt.get(d, 0) + 1; busy[d] = i + 18
    return pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts'])
def mr2():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        val, poc, vah = VA[days[k - 1]]; pd_ = days[k - 1]; ph, pl = DY.h[pd_], DY.l[pd_]; ix = gi[d]; done = {1: False, -1: False}; hi_s, lo_s = -np.inf, np.inf
        up_lvl, dn_lvl = min(ph, vah) if vah < ph else ph, max(pl, val) if val > pl else pl
        broke_up = broke_dn = False
        for p, i in enumerate(ix[:-1]):
            hi_s, lo_s = max(hi_s, h[i]), min(lo_s, l[i])
            if h[i] > max(vah, ph if ph < vah else vah): broke_up = True
            if l[i] < min(val, pl if pl > val else val): broke_dn = True
            if m[i] >= 840 or (m[i] + 5) % 15 != 0: continue
            if broke_up and not done[-1] and val <= c[i] <= vah and c[i] > poc:
                e = o[ix[p + 1]]; S = hi_s + .03 * DY.adr[d]; rows.append((d, m[i], -1, walk(ix, p + 1, -1, e, poc, S) - 1.0)); done[-1] = True
            if broke_dn and not done[1] and val <= c[i] <= vah and c[i] < poc:
                e = o[ix[p + 1]]; S = lo_s - .03 * DY.adr[d]; rows.append((d, m[i], 1, walk(ix, p + 1, 1, e, poc, S) - 1.0)); done[1] = True
    return pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts'])
def mr3():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        pd_ = days[k - 1]; pc = DY.c[pd_]; ix = gi[d]; o0 = o[ix[0]]; g = (o0 - pc) / DY.adr[d]
        if not (.1 <= abs(g) <= .5) or not (DY.l[pd_] <= o0 <= DY.h[pd_]): continue
        dr = -int(np.sign(g)); e = o[ix[1]]; S = e - dr * .15 * DY.adr[d]
        rows.append((d, m[ix[1]], dr, walk(ix, 1, dr, e, pc, S, until_m=720) - 1.0))
    return pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts'])
if __name__ == '__main__':
    gneg = DY.gex < 0; gk = DY.gex.notna(); PER = [(2014, 2017), (2018, 2021), (2022, 2026)]
    for nm, T in (('MR1 fade the stretched flip', mr1()), ('MR2 failed auction', mr2()), ('MR3 gap fill', mr3())):
        T = T[T.day.isin(DY.index)]; yr = T.day.dt.year
        per = lambda msk: ' / '.join(f'{T.pts[msk & (yr >= a) & (yr <= b)].mean():+.2f}' if (msk & (yr >= a) & (yr <= b)).sum() >= 15 else '   .' for a, b in PER)
        al = np.ones(len(T), bool); pg = (T.day.map(gk) & ~T.day.map(gneg)).values.astype(bool); ng = (T.day.map(gk) & T.day.map(gneg)).values.astype(bool)
        hg = (T.day.map(DY.g2p) >= 2 / 3).values; lg = (T.day.map(DY.g2p) < 2 / 3).values
        print(f'\n{nm}: {len(T)} trades ({len(T) / len(DY):.2f}/day)', flush=True)
        for lab, msk in (('all days', al), ('positive gamma (2018+)', pg), ('negative gamma (2018+)', ng), ('GEX top third (2014+)', hg), ('GEX not top third', lg)):
            t = T.pts[msk]; print(f'  {lab:<24} n{len(t):5d} win {np.mean(t > 0):3.0%} avg {t.mean():+6.2f} | 14-17 / 18-21 / 22-25: {per(msk)}', flush=True)
        T.to_parquet(f'/tmp/claude-0/ml/dte_{nm.split()[0]}.parquet')
