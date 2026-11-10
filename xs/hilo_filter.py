"""Nov 9 2026: REY'S HIGHS / LOWS MODEL ON NQ FUTURES (screener's high-in / low-in, xs/lod_hod.py + xs/range_models.py). DECLARED BEFORE RESULTS.
Signal (15m checkpoints 10:30 - 15:00, RTH 5m bars): HIGH LOOKS IN = the day's high so far was set >= 30 min earlier and price is >= 0.5 sigma below it (mirror: low in).
Sigma, two anchors: HTF = HAR forecast of the day's high/low range / 1.665 (range-based, as range_models.py, fit on NQ 2014-19) | 4H = Garman-Klass sigma of the last 60
completed 4h bars at 09:30 (24h NQ bars), scaled to a day by calibration. Both known before the signal.
Calibration on 2018-21 only: P(the 'in' extreme holds to the close) by checkpoint; the distance from the in-extreme to the day's final other extreme in sigma units ->
projected other extreme = median (q50) and the level reached 80% of the time (q20).
TRADE: first signal of the day per side; short at the next 5m open after 'high in' (long after 'low in'); stop = the in-extreme + 0.1 sigma; target = projected other extreme
(q50 or q20) if it is at least 0.05 sigma beyond the entry; out at 15:55. 1 pt cost. Judged 2022-25 vs 2018-21.
FILTER: the system's RTH trades (noise momentum, V15a, rules algo, Rule A) taken AGAINST an extreme that looks in at entry (long while the high looks in) vs the rest."""
_ft = open('/home/claude/fartpie/xs/flex50_thod.py').read(); exec(_ft[:_ft.rindex("if __name__ == '__main__':")])
def gk4(t):
    k = X4.index.searchsorted(t, 'left'); r = X4.iloc[max(0, k - 60):k]
    if len(r) < 40: return np.nan
    v_ = np.maximum(.5 * np.log(r.high / r.low) ** 2 - (2 * np.log(2) - 1) * np.log(r.close / r.open) ** 2, 1e-9); return np.sqrt(v_.mean()) * r.close.values[-1]
S4 = pd.Series({d: gk4(d + pd.Timedelta(hours=9, minutes=30)) for d in DY.index if d >= pd.Timestamp('2014-03-01')})
SH = FC / 1.665
def events():
    rows = []
    for d in days:
        if not ok(d) or DY.roll[d] or d not in S4.index: continue
        ix = gi[d]; endp = np.where(m[ix] < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; fh, fl = h[ix].max(), l[ix].min(); sh_, s4_ = SH.get(d, np.nan), S4.get(d, np.nan); hi_ = -np.inf; lo_ = np.inf; th = tl = 570; done = {1: False, -1: False}
        for p in range(len(ix)):
            i = ix[p]; t = m[i] + 5
            if h[i] > hi_: hi_, th = h[i], t
            if l[i] < lo_: lo_, tl = l[i], t
            if t % 15 or t < 630 or t > 900 or p + 1 > endp: continue
            for src, sg in (('HTF', sh_), ('4H', s4_)):
                if not np.isfinite(sg): continue
                if t - th >= 30 and c[i] <= hi_ - .5 * sg and not done.get((src, -1)):
                    rows.append(dict(day=d, src=src, side=-1, t=t, p=p, anchor=hi_, sg=sg, held=fh <= hi_, dist=(hi_ - fl) / sg, e=o[ix[p + 1]])); done[(src, -1)] = True
                if t - tl >= 30 and c[i] >= lo_ + .5 * sg and not done.get((src, 1)):
                    rows.append(dict(day=d, src=src, side=1, t=t, p=p, anchor=lo_, sg=sg, held=fl >= lo_, dist=(fh - lo_) / sg, e=o[ix[p + 1]])); done[(src, 1)] = True
    return pd.DataFrame(rows)
def trade(r_, q):
    d = r_.day; ix = gi[d]; endp = np.where(m[ix] < 955)[0][-1]; j = r_.p + 1; dr = -r_.side if False else (1 if r_.side == 1 else -1)
    e = r_.e; S = r_.anchor - dr * .1 * r_.sg; T = r_.anchor + dr * q * r_.sg
    if (T - e) * dr < .05 * r_.sg or (e - S) * dr <= 0: return np.nan, np.nan
    hi2, lo2, op2, cl2 = h[ix[j:endp + 1]], l[ix[j:endp + 1]], o[ix[j:endp + 1]], c[ix[j:endp + 1]]
    adv = (lo2 <= S) if dr > 0 else (hi2 >= S); fav = (hi2 >= T) if dr > 0 else (lo2 <= T)
    ia = int(np.argmax(adv)) if adv.any() else 10 ** 9; ib = int(np.argmax(fav)) if fav.any() else 10 ** 9
    if ia == ib == 10 ** 9: return (cl2[-1] - e) * dr - 1.0, 0
    if ia < ib or (ia == ib and (cl2[ia] >= op2[ia]) == (dr > 0)): return (S - e) * dr - 1.0, 0
    return (T - e) * dr - 1.0, 1

import pickle
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
"""Nov 10 2026: HIGH-IN / LOW-IN AS A FILTER, SIZE RULE AND DIAGNOSTIC ON THE SYSTEM'S RTH TRADES. DECLARED BEFORE RESULTS.
State at each trade's entry bar (last completed 5m bar before entry, HAR-based sigma, no future data): high looks in = day's high set >= 30 min earlier and price >= 0.5 sigma below it; low looks in = mirror.
Class: AGAINST = long while the high looks in, or short while the low looks in; WITH = long while the low looks in (or short while the high looks in); NEUTRAL = neither.
Trades: S6 V2 trades with the pipeline's mv30 skip applied (sz > 0), engines RA, M1, R1, RULES, V15a, plus the noise engine (nz_core). Periods 2018-21 | 2022-26.
TEST A pts per trade by class, per engine and pooled (t of against-vs-rest and with-vs-rest).  PASS = same sign, |t| >= 2, both periods.
TEST B system level (only RA and V15a carry a direction column in the live tables; M1, R1, RULES and the ON legs are untouched): multiplier on the live size (against x0 skip, against x0.5, with x1.25, both), weighted points / day, daily std, mean/std, worst month, both periods (RTH engines only; ON legs untouched)."""
def state(d, mm_, dr):
    d = pd.Timestamp(d)
    if d not in gi or d not in SH.index or not np.isfinite(SH[d]): return np.nan
    ix = gi[d]; w = np.where(m[ix] == mm_)[0]
    if not len(w): return np.nan
    p = w[0] - 1
    if p < 0: return 0.0
    sg = SH[d]; seg = ix[:p + 1]; hi_ = h[seg].max(); lo_ = l[seg].min(); th = m[seg][np.argmax(h[seg])] + 5; tl = m[seg][np.argmin(l[seg])] + 5; t = m[ix[p]] + 5; px = c[ix[p]]
    hin = t - th >= 30 and px <= hi_ - .5 * sg; lin = t - tl >= 30 and px >= lo_ + .5 * sg
    ag = (dr > 0 and hin) or (dr < 0 and lin); wi = (dr > 0 and lin) or (dr < 0 and hin)
    return -1.0 if ag and not wi else (1.0 if wi and not ag else 0.0)
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); BM = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb'))); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'ON+': .4, 'ON-': .1, 'RULES': 1.0, 'V15a': .3}
TR = {}
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n in ('ON+', 'ON-'): continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); B_ = BM[n].copy(); B_['day'] = pd.to_datetime(B_.day).dt.normalize(); B_ = B_[B_.sz > 0][['day', 'm', 'sz']].rename(columns={'sz': 'szb'}); t = t.merge(B_, on=['day', 'm'], how='inner'); t['sz'] = t.szb; t = t[(t.day >= '2018-02-01') & (t.day <= '2026-01-30')]
    t['st'] = [state(d, int(mm_), int(dr)) for d, mm_, dr in zip(t.day, t.m, t.dir)] if 'dir' in t.columns else np.nan; TR[n] = t
import nz_core as Z
nz = pd.DataFrame([r for d in Z.DAYS for r in Z.noise(d)], columns=['day', 'j', 'dir', 'pts', 'mae']); nz['day'] = pd.to_datetime(nz.day); nz = nz[nz.day >= '2018-02-01']; nz['m'] = Z.GRID[nz.j.values.astype(int)]; nz['sz'] = 1.0
nz['st'] = [state(d, int(mm_), int(dr)) for d, mm_, dr in zip(nz.day, nz.m, nz.dir)]; TR['NZ'] = nz
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
PER = (('18-21', lambda d: d < '2022-01-01'), ('22-26', lambda d: d >= '2022-01-01'))
print('=== TEST A: pts per trade by class (against | neutral | with), 18-21 // 22-26; t(against vs rest), t(with vs rest) ===')
ALL = pd.concat([t.assign(eng=n) for n, t in TR.items() if n != 'NZ'])
for nm, t in list(TR.items()) + [('POOLED RTH engines', ALL)]:
    t = t[t.st.notna()]
    if not len(t): continue
    out = []; tag = []
    for lab, f in PER:
        x = t[f(t.day)]; g = {k: x[x.st == k].pts for k in (-1.0, 0.0, 1.0)}
        out.append(' | '.join(f'{g[k].mean():+6.1f} (n{len(g[k])})' for k in (-1.0, 0.0, 1.0))); tag.append((tt(g[-1.0], x[x.st != -1.0].pts), tt(g[1.0], x[x.st != 1.0].pts)))
    pa = np.sign(tag[0][0]) == np.sign(tag[1][0]) and all(abs(q[0]) >= 2 for q in tag if np.isfinite(q[0])) and np.isfinite(tag[0][0]) and np.isfinite(tag[1][0])
    pw = np.sign(tag[0][1]) == np.sign(tag[1][1]) and all(abs(q[1]) >= 2 for q in tag if np.isfinite(q[1])) and np.isfinite(tag[0][1]) and np.isfinite(tag[1][1])
    print(f'  {nm:<19} {out[0]} // {out[1]} | t against {tag[0][0]:+5.2f}/{tag[1][0]:+5.2f}{"  <== PASS" if pa else ""} | t with {tag[0][1]:+5.2f}/{tag[1][1]:+5.2f}{"  <== PASS" if pw else ""}', flush=True)
print('\n=== TEST B: system multipliers (RTH engines x live size x engine weight, noise engine not included) ===')
DAYS_ = [d for d in Z.DAYS if d >= pd.Timestamp('2018-02-01')]
def system(fa, fw):
    s = pd.Series(0.0, index=pd.DatetimeIndex(DAYS_))
    for n, t in TR.items():
        if n == 'NZ': continue
        mu = np.where(t.st == -1, fa, np.where(t.st == 1, fw, 1.0)); s = s.add((t.pts * t.sz * mu * UW[n]).groupby(t.day.values).sum().reindex(s.index).fillna(0.0), fill_value=0)
    return s
for lab, fa, fw in (('base', 1, 1), ('skip against', 0, 1), ('half against', .5, 1), ('with x1.25', 1, 1.25), ('half against + with x1.25', .5, 1.25), ('skip against + with x1.25', 0, 1.25)):
    s = system(fa, fw); o = []
    for nm, f in PER:
        x = s[f(s.index)]; mo = x.groupby(x.index.to_period('M')).sum(); o.append(f'{nm}: {x.mean():+5.2f}/d std {x.std():5.1f} m/s {x.mean() / x.std():.3f} worstM {mo.min():+5.0f}')
    print(f'  {lab:<28}' + ' | '.join(o), flush=True)
