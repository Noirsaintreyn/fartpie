"""One-paragraph MARKET READ for the 09:30 post: market state -> stocks above vector / EMA cross -> VIX vs its vector -> correlation -> plausible move.
State inputs use PRIOR daily closes (vector regime = backtest_pivot_ratchet state machine, EMA cross = 9 vs 21) plus pre-market moves.
What the 2025 test supports (234 NQ sessions, walk-forward): the state does NOT predict the 09:35-12:00 direction (accuracy ~ coin flip), but it does flag a WIDE-range regime (VIX above its 60d median, or <= 40% of the basket above vector, or SPY vector negative):
mean |move| 68 bps vs 38 bps otherwise.  So the read is about range and conviction, not a directional bias."""
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import vchart as VC

BASKET = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', 'TSLA', 'COST', 'NFLX', 'AMD', 'ADBE']
NYTZ = ZoneInfo('America/New_York')
STATS = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'market_state_regime_stats.json')))
_cache = {}


def _daily(tk):
    import yfinance as yf
    d = yf.Ticker(tk).history(period='2y', interval='1d', auto_adjust=False)[['Open', 'High', 'Low', 'Close']].dropna(); d.index = d.index.tz_localize(None).normalize()
    today = pd.Timestamp(datetime.now(NYTZ).date()); d = d[d.index < today]                      # prior closes only
    f = pd.DataFrame(dict(dt=d.index, open=d.Open.values, high=d.High.values, low=d.Low.values, close=d.Close.values))
    e9, e21 = d.Close.ewm(span=9, adjust=False).mean(), d.Close.ewm(span=21, adjust=False).mean()
    return pd.DataFrame(dict(close=d.Close, ret=d.Close.pct_change() * 100, vec=VC.vector_regime(f), ema=np.sign(e9 - e21).values), index=d.index)


def _states():
    if 'st' in _cache and time.time() - _cache['st'][0] < 900: return _cache['st'][1]
    S = {tk: _daily(tk) for tk in BASKET + ['SPY', 'QQQ', '^VIX', 'NQ=F']}; _cache['st'] = (time.time(), S); return S


def _premarket():
    import yfinance as yf
    out = {}
    for tk in BASKET:
        try: fi = yf.Ticker(tk).fast_info; out[tk] = (float(fi['last_price']) / float(fi['previous_close']) - 1) * 100
        except Exception: pass
    return out


def market_story(resp, L):
    """resp = backend 5m data response (for NQ's overnight move); L = VT.render.last (targets, price)"""
    S = _states(); pm = _premarket(); last = lambda tk, c: S[tk][c].iloc[-1]
    vec = np.array([last(t, 'vec') for t in BASKET]); ema = np.array([last(t, 'ema') for t in BASKET]); nv, ne = int((vec > 0).sum()), int((ema > 0).sum()); n = len(BASKET)
    spy, qqq = int(last('SPY', 'vec')), int(last('QQQ', 'vec')); vix = S['^VIX']; vlvl = float(vix.close.iloc[-1]); vhi = vlvl > float(vix.close.iloc[-60:].median()); vvec, vema = int(vix.vec.iloc[-1]), int(vix.ema.iloc[-1]); vret = float(vix.ret.iloc[-1])
    wide = bool(vhi or nv / n <= 0.4 or spy < 0); risk_on = nv / n >= 0.6 and spy > 0 and vema <= 0 and not vhi; risk_off = nv / n <= 0.4 or spy < 0 or (vema > 0 and vhi)
    label = 'RISK-ON' if risk_on else ('RISK-OFF' if risk_off else 'MIXED')
    up_vec = [t for t, v in zip(BASKET, vec) if v > 0 and pm.get(t, -1) > 0]; bpm = float(np.mean(list(pm.values()))) if pm else None
    # correlations: NQ daily returns vs VIX and vs the basket average, last 60 sessions
    r = pd.DataFrame({'nq': S['NQ=F'].ret, 'vix': vix.ret, 'bk': pd.concat([S[t].ret for t in BASKET], axis=1).mean(axis=1)}).dropna().iloc[-60:]; cv, cb = float(r.nq.corr(r.vix)), float(r.nq.corr(r.bk))
    # NQ overnight: last price vs prior-day 16:00 close, from the 5m bars
    on = None
    try:
        df = VC._frame(resp['priceData']); today = pd.Timestamp(datetime.now(NYTZ).date()); prev = df[df['dt'] < today]
        if len(prev): pday = prev['dt'].iloc[-1].normalize(); pc = prev[prev['dt'] <= pday + pd.Timedelta(hours=16)]['close'].iloc[-1]; on = (float(df['close'].iloc[-1]) / float(pc) - 1) * 100
    except Exception: pass
    s1 = f"**Market state: {label}.** {nv} of {n} mega-caps are above their daily vector ({ne} with EMA 9 over 21)" + (f", and {len(up_vec)} of those {'is' if len(up_vec) == 1 else 'are'} also up pre-market" if pm else '') + f"; SPY vector {'positive' if spy > 0 else 'negative'}, QQQ {'positive' if qqq > 0 else 'negative'}."
    s2 = (f" VIX {vlvl:.1f} is {'above' if vhi else 'below'} its 60-day median, its vector is {'positive' if vvec > 0 else 'negative'} and its EMA cross is {'rising' if vema > 0 else 'falling'} ({vret:+.1f}% yesterday), "
          f"which reads {'as stress' if (vema > 0 and vhi) else 'as calm' if (vema <= 0 and not vhi) else 'as mixed'}.")
    s3 = f" Over the last 60 sessions NQ has moved {cv:+.2f} with VIX and {cb:+.2f} with the stock basket"
    if on is not None and bpm is not None:
        conf = (on > 0) == (bpm > 0); s3 += f"; overnight NQ is {on:+.2f}% and the basket {bpm:+.2f}% pre-market, so the stocks are {'confirming' if conf else 'diverging from'} NQ."
    elif on is not None: s3 += f"; NQ is {on:+.2f}% overnight."
    else: s3 += '.'
    w, nr = STATS['wide'], STATS['normal']
    s4 = (f" Today reads {'WIDE' if wide else 'NORMAL'}: in 2025, sessions in a wide state (VIX above its median, weak breadth or SPY vector negative) moved about {w['mean_abs_bps']/100:.2f}% from 09:35 to 12:00 on average versus {nr['mean_abs_bps']/100:.2f}% otherwise"
          f", so expect {'a bigger-than-usual' if wide else 'a normal'} range. The odds model's realistic target is T1 ±{L['t1']:.0f}pt (50% to be touched by 12:00) with T2 ±{L['t2']:.0f}pt as the stretch (25%)." + (f" Odds of reaching T1 before a stop at half that distance are {L['trade']['t1_half']*100:.0f}% ({L['trade']['t1_full']*100:.0f}% with a full-distance stop; weak signal)." if L.get('trade') else ''))
    s5 = " Direction: none of these states predicted the morning's direction in testing (about a coin flip), so this is a read on range and conviction, not a bias."
    return s1 + s2 + s3 + s4 + s5
