"""DAILY READ (facts only) for the website's Daily Analysis tab.

build(snapshot=None) -> dict:
  events   US calendar (CPI / PPI / Fed / jobs ...) from the free ForexFactory JSON (actual vs forecast when released), plus headlines (Yahoo) for SPY / QQQ and the held names
  indices  SPY QQQ IWM NQ ES: gap and its rank, HAR expected range vs a normal day, Yang-Zhang rank, VIX, analog size read (xs/morning_read.py)
  profile  prior-session volume profile (POC / VAH / VAL) and TPO shape, 5-day composite, where today's open sits (15m bars)
  options  GEX / DEX / IV / vanna / vega from live_data/greeks_snapshot.json when it is fresh (pushed from the Mac's ThetaData), else says how old
  screener what the above means for the held names (event risk, vol regime, stop room)
  futures  today's NQ plan: gap-size rule, HAR size, event flag, squeeze flag, noise bands, profile levels
Every line is a fact with a number; no essay.  Nothing here changes a signal.
"""
import datetime as dt, json, os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'xs'))
ET = 'America/New_York'
FUT = {'NQ': 'NQ=F', 'ES': 'ES=F'}
_cache = {}


def _c(key, ttl, fn):
    h = _cache.get(key)
    if h and time.time() - h[0] < ttl: return h[1]
    v = fn(); _cache[key] = (time.time(), v); return v


# ------------------------------------------------------------------ events: calendar + headlines
def calendar():
    import requests
    def get():
        r = requests.get('https://nfs.faireconomy.media/ff_calendar_thisweek.json', timeout=15, headers={'User-Agent': 'Mozilla/5.0'}); r.raise_for_status(); return r.json()
    try: raw = _c('cal', 900, get)
    except Exception: return None
    out = []
    for e in raw:
        if e.get('country') != 'USD' or e.get('impact') not in ('High', 'Medium'): continue
        t = pd.Timestamp(e['date']).tz_convert(ET)
        out.append(dict(when=t.strftime('%a %H:%M'), date=str(t.date()), title=e['title'], impact=e['impact'], forecast=e.get('forecast') or '', previous=e.get('previous') or '', actual=e.get('actual') or ''))
    return out


def headlines(tickers, n=3):
    import yfinance as yf
    out = []
    for tk in tickers:
        try: items = _c('news' + tk, 900, lambda: yf.Ticker(tk).news or [])
        except Exception: items = []
        for it in items[:n]:
            c = it.get('content') or {}
            if c.get('title'): out.append(dict(tk=tk, title=c['title'], pub=(c.get('provider') or {}).get('displayName', ''), t=c.get('pubDate', '')[:16].replace('T', ' ') + 'Z'))
    seen, res = set(), []
    for h in sorted(out, key=lambda x: x['t'], reverse=True):
        if h['title'] in seen: continue
        seen.add(h['title']); res.append(h)
    return res


# ------------------------------------------------------------------ price context (daily)
def indices(now):
    import morning_read as MR
    out = {}
    for lab, sym in (('SPY', 'SPY'), ('QQQ', 'QQQ'), ('IWM', 'IWM')):
        try:
            d, F, Y, lines, ratio = _c('mr' + sym, 600, lambda s=sym: MR.report(s, None))
            t = d.index[-1]; f = F.loc[t]
            dts, w, cols, z, dist = MR.knn(F, Y.drop(columns=['gapfill_n']), t); A = Y.loc[dts]
            B = Y[(Y.index < t) & (Y.index >= t - pd.Timedelta(days=365 * 6))].dropna()
            out[lab] = dict(date=str(t.date()), last=round(float(d.Close.iloc[-1]), 2), gap=round(float(f.gap), 2), gap_rank=round(float(f.gabs_rank), 2) if f.gabs_rank == f.gabs_rank else None,
                            size=round(float(ratio), 2), range_med=round(float(MR.wq(A.range.values, w, .5)), 2), range_base=round(float(B.range.median()), 2),
                            p_up=round(float(np.average(A.ret > 0, weights=w)), 2), gap_fill=round(float(np.average(A.fill, weights=w)), 2), gap_fill_base=round(float(B.fill.mean()), 2),
                            squeeze=bool(f.ttm == 1), bbw=round(float(f.bbw), 2), adx=round(float(f.adx), 0), vol1=round(float(f.vol1), 2), yz_rank=None,
                            vix=round(float(np.exp(f.vix)), 1), vix5=round(float(f.vix5 * 100), 1), vixgap=round(float(f.vixgap * 100), 1) if f.vixgap == f.vixgap else None)
        except Exception as ex:
            out[lab] = dict(error=str(ex)[:100])
    return out


# ------------------------------------------------------------------ volume profile + TPO from 15m bars
def _bars15(sym):
    import yfinance as yf
    def get():
        h = yf.Ticker(sym).history(period='59d', interval='15m', prepost=False, auto_adjust=False)
        h = h.reset_index(); t = pd.to_datetime(h.iloc[:, 0]); t = t.dt.tz_convert(ET).dt.tz_localize(None) if t.dt.tz is not None else t
        df = pd.DataFrame({'dt': t, 'o': h.Open.values, 'h': h.High.values, 'l': h.Low.values, 'c': h.Close.values, 'v': h.Volume.values}).dropna()
        mm = df.dt.dt.hour * 60 + df.dt.dt.minute; df = df[(mm >= 570) & (mm < 960) & (df.dt.dt.dayofweek < 5)]
        df['day'] = df.dt.dt.normalize(); return df
    return _c('b15' + sym, 600, get)


def _profile(df, tick):
    """volume at price (each bar's volume spread over its range) and TPO counts (30-min letters) -> POC, value area 70%"""
    lo, hi = df.l.min(), df.h.max(); n = max(int(round((hi - lo) / tick)) + 1, 2); px = lo + tick * np.arange(n); vol = np.zeros(n); tpo = np.zeros(n)
    for _, b in df.iterrows():
        m = (px >= b.l - 1e-9) & (px <= b.h + 1e-9)
        if m.any(): vol[m] += b.v / m.sum()
    b30 = df.assign(k=(df.dt.dt.hour * 60 + df.dt.dt.minute - 570) // 30).groupby('k').agg(l=('l', 'min'), h=('h', 'max'))
    for _, b in b30.iterrows(): tpo[(px >= b.l - 1e-9) & (px <= b.h + 1e-9)] += 1
    def va(arr):
        i = int(arr.argmax()); lo_i = hi_i = i; tot = arr.sum(); acc = arr[i]
        while acc < .7 * tot and (lo_i > 0 or hi_i < len(arr) - 1):
            a = arr[lo_i - 1] + (arr[lo_i - 2] if lo_i > 1 else 0) if lo_i > 0 else -1; b = arr[hi_i + 1] + (arr[hi_i + 2] if hi_i < len(arr) - 2 else 0) if hi_i < len(arr) - 1 else -1
            if a >= b: lo_i -= 1 if lo_i > 0 else 0; acc += arr[lo_i]
            else: hi_i += 1 if hi_i < len(arr) - 1 else 0; acc += arr[hi_i]
        return px[i], px[lo_i], px[hi_i]
    pv, lv, hv = va(vol); pt, lt, ht = va(tpo)
    mid = (hi + lo) / 2; skew = (vol[px > mid].sum() - vol[px < mid].sum()) / vol.sum()
    shape = 'P (volume high in the range)' if skew > .25 else 'b (volume low in the range)' if skew < -.25 else 'D (balanced)'
    single = int(((tpo == 1) & (px > lo + tick * 2) & (px < hi - tick * 2)).sum())
    return dict(poc=round(float(pv), 2), vah=round(float(hv), 2), val=round(float(lv), 2), tpo_poc=round(float(pt), 2), tpo_vah=round(float(ht), 2), tpo_val=round(float(lt), 2), hi=round(float(hi), 2), lo=round(float(lo), 2), shape=shape,
                single_prints=single, ib=None)


def profiles(symbols=(('SPY', 'SPY', .05), ('QQQ', 'QQQ', .05), ('NQ', 'NQ=F', 5.0))):
    out = {}
    for lab, sym, tick in symbols:
        try:
            df = _bars15(sym); days = sorted(df.day.unique())
            if len(days) < 7: raise RuntimeError('not enough bars')
            today = days[-1]; live_today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize() == pd.Timestamp(today)
            prior = days[-2] if live_today else days[-1]; hist = [d for d in days if d <= prior][-5:]
            P = _profile(df[df.day == prior], tick); C = _profile(df[df.day.isin(hist)], tick)
            o = float(df[df.day == today].o.iloc[0]) if live_today else None
            ib = df[(df.day == prior) & (df.dt.dt.hour * 60 + df.dt.dt.minute < 630)]; P['ib'] = [round(float(ib.l.min()), 2), round(float(ib.h.max()), 2)]
            pos = None
            if o is not None: pos = 'above the prior value area' if o > P['vah'] else 'below the prior value area' if o < P['val'] else 'inside the prior value area'
            out[lab] = dict(prior=str(pd.Timestamp(prior).date()), prior_profile=P, composite5=C, open=round(o, 2) if o else None, open_vs_va=pos,
                            open_vs_poc=round(o - P['poc'], 2) if o else None)
        except Exception as ex:
            out[lab] = dict(error=str(ex)[:100])
    return out


# ------------------------------------------------------------------ options greeks (pushed from the Mac)
def options():
    f = os.path.join(ROOT, 'live_data', 'greeks_snapshot.json')
    try: d = json.load(open(f))
    except Exception: d = None
    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    if d:
        age = (today - pd.Timestamp(d.get('asof', '2000-01-01')).normalize()).days
        d['age_days'] = age; d['stale'] = age > 3; return d
    try:                                                                       # fall back to the last row of the research files (stale by construction)
        g = pd.read_csv(os.path.join(ROOT, 'options_greeks_daily_QQQ.csv'), parse_dates=['date']).sort_values('date').iloc[-1]
        age = (today - g.date.normalize()).days
        return dict(asof=str(g.date.date()), age_days=age, stale=True, symbols={'QQQ': dict(gex=float(g.call_gex + g.put_gex), dex=float(g.call_dex + g.put_dex), iv=float(g.atm_iv), skew=float(g.skew_25d))})
    except Exception:
        return dict(stale=True, age_days=None, symbols={})


# ------------------------------------------------------------------ NQ plan
def nq_plan(idx, prof, events, now):
    import yfinance as yf
    plan = dict(lines=[], legs={})
    try:
        d = yf.download('NQ=F', period='2y', interval='1d', progress=False, auto_adjust=False); d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        d.index = pd.to_datetime(d.index).normalize(); H, L, C, O = d.High, d.Low, d.Close, d.Open; rg = H - L
        b15 = _bars15('NQ=F'); dd = sorted(b15.day.unique()); t0 = dd[-1]; live = pd.Timestamp.now(tz=ET).tz_localize(None).normalize() == pd.Timestamp(t0)
        o_rth = float(b15[b15.day == t0].o.iloc[0]) if live else float(b15[b15.day == dd[-1]].c.iloc[-1]); pc_rth = float(b15[b15.day == (dd[-2] if live else dd[-1])].c.iloc[-1])
        gap = o_rth - pc_rth; rg_rth = b15.groupby('day').apply(lambda g: g.h.max() - g.l.min()); g20 = float(rg_rth[rg_rth.index < t0].iloc[-20:].mean()); gx = abs(gap) / g20
        gr = (idx.get('QQQ') or {}).get('gap_rank')       # NQ and QQQ share the 09:30 gap vs the 16:00 close; the QQQ series has the long RTH history the rank needs
        gr = float(gr) if gr is not None else float('nan')
        tr = pd.concat([rg, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean(); sd = C.rolling(20).std()
        sq = bool(2 * sd.iloc[-2] < 1.5 * atr.rolling(20).mean().iloc[-2]) if len(d) > 40 else None
        plan['nq'] = dict(open=round(o_rth, 2), prior_close=round(pc_rth, 2), gap_pts=round(float(gap), 2), gap_x=round(float(gx), 2), gap_rank=round(float(gr), 2), squeeze=sq,
                          rng20=round(float(g20), 1))
        mult = 0.5 if gr == gr and gr <= .33 else 1.0
        plan['gap_rule'] = dict(rank=round(float(gr), 2), mult=mult, legs=['RA', 'M1', 'R1', 'RULES', 'V15a'], untouched=['ON+', 'ON-'])
        plan['lines'].append(f"Gap {gap:+.1f} pts = {gx:.2f}x the 20-day range; |gap| rank {gr:.0%} of the last year (QQQ's, same 09:30 gap) -> {'BOTTOM THIRD: RA, M1, R1, RULES, V15a at x0.5' if mult < 1 else 'not bottom third: full size'} (ON+/ON- unchanged)")
        if sq: plan['lines'].append('TTM squeeze is ON (candidate: skip the noise leg and ON-; unproven in the Lucid sim, not applied)')
    except Exception as ex:
        plan['error'] = str(ex)[:100]
    hi_today = [e for e in (events or []) if e['date'] == str(now.date()) and e['impact'] == 'High']
    if hi_today: plan['lines'].append('High-impact today: ' + ', '.join(f"{e['title']} {e['when'][4:]}" + (f" (actual {e['actual']} vs {e['forecast']})" if e['actual'] else f" (fcst {e['forecast']})") for e in hi_today))
    p = (prof.get('NQ') or {}).get('prior_profile')
    if p: plan['lines'].append(f"NQ prior session: POC {p['poc']:.0f}, value area {p['val']:.0f}-{p['vah']:.0f}, shape {p['shape']}; open {prof['NQ'].get('open_vs_va') or 'n/a'}")
    # noise bands (14-session mean |close/open - 1| by time of day, from 5m bars)
    try:
        h = yf.Ticker('NQ=F').history(period='30d', interval='5m', prepost=False, auto_adjust=False).reset_index()
        t = pd.to_datetime(h.iloc[:, 0]).dt.tz_convert(ET).dt.tz_localize(None); x = pd.DataFrame({'dt': t, 'o': h.Open.values, 'c': h.Close.values}).dropna()
        x['day'] = x.dt.dt.normalize(); x['mm'] = x.dt.dt.hour * 60 + x.dt.dt.minute; x = x[(x.mm >= 570) & (x.mm < 960)]
        op = x.groupby('day').o.first(); x['mv'] = (x.c / x.day.map(op) - 1).abs(); days = sorted(x.day.unique())
        today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize(); past = [d for d in days if d < today][-14:]
        sig = x[x.day.isin(past)].groupby('mm').mv.mean(); o_ = float(op.get(today, op.iloc[-1])); pc = plan.get('nq', {}).get('prior_close', o_); ref_hi, ref_lo = max(o_, pc), min(o_, pc)
        plan['noise_bands'] = [dict(t=f'{m // 60}:{m % 60:02d}', upper=round(ref_hi * (1 + float(sig[m])), 2), lower=round(ref_lo * (1 - float(sig[m])), 2)) for m in (600, 660, 720, 840, 930) if m in sig.index]
    except Exception as ex:
        plan['noise_error'] = str(ex)[:80]
    return plan


# ------------------------------------------------------------------ screener implications
def screener_view(snapshot, events, news, idx):
    pos = (snapshot or {}).get('positions') or []
    held = [p['base'] for p in pos]
    lines = []
    for p in pos:
        try:
            e, s, l = float(p['entry']), float(p['stop']), float(p['last']); lines.append(dict(tk=p['base'], pnl=p.get('pnl'), room=round((l / s - 1) * 100, 1), stop=s, last=l, selling=bool(p.get('selling'))))
        except Exception: pass
    ev = [f"{e['title']} {e['when']}" for e in (events or []) if e['impact'] == 'High' and e['date'] >= str(pd.Timestamp.now(tz=ET).date())][:4]
    held_news = [n for n in news if n['tk'] in held][:6]
    return dict(positions=lines, n=len(pos), slots=(snapshot or {}).get('slots'), events=ev, held_news=held_news, asof=(snapshot or {}).get('asof'), market_open=(snapshot or {}).get('market_open'))


def build(snapshot=None):
    now = pd.Timestamp.now(tz=ET).tz_localize(None)
    ev = calendar(); idx = indices(now); prof = profiles(); opt = options(); held = [p['base'] for p in ((snapshot or {}).get('positions') or [])]
    news = headlines(['SPY', 'QQQ'] + held[:6], n=3)
    out = dict(asof=now.strftime('%Y-%m-%d %H:%M ET'), date=str(now.date()), events=ev, news=news, indices=idx, profile=prof, options=opt,
               screener=screener_view(snapshot, ev, news, idx), futures=nq_plan(idx, prof, ev, now))
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == '__main__':
    r = build(); print(json.dumps(r, indent=1)[:6000])
