"""Nov 24 2026 report 450. Rey: look on LSE for TQQQ. Same test as report 449 but on REAL TQQQ / SQQQ 15m bars from the LSE vault (xs/pull_lse_tqqq.py): signal from QQQ (vector flips, real OI gamma), entry/exit at the TQQQ (bull) or SQQQ (bear) bar OPEN of the same timestamp, one position at a time, cost 4bp round trip, overnight allowed, exit = open after the opposite flip. Financing/expense are already in the ETF price. Dividends ignored. DECLARED BEFORE RESULTS (identical to 449): arms A bull ungated | B bull NEG | C bull POS | D bull TQQQ + bear SQQQ | E both NEG; timeframes 15m / 1h / 1d (QQQ bars aggregated from 15m RTH). PASS: NEG-POS per-trade t >= 2 pooled and same sign train (2019-21)/test (2018+2022+); B beats A in train AND test on CAGR/DD. Also report synthetic-vs-real per-trade gap to show how good the synthetic proxy was. Daily-close marks (DD is a floor). SQQQ prices are split-adjusted by the vendor (huge levels), used only as ratios."""
import os
src = open('xs/report_449_tqqq_synth_flips.py').read(); exec(src[:src.index("T = pd.DataFrame(trades)")])
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def load_etf(tk):
    d = pd.read_parquet(SP + f'{tk}_15m_lse.parquet'); d['dt'] = pd.to_datetime(d.ts, utc=True).dt.tz_convert(ET).dt.tz_localize(None); mm = d.dt.dt.hour * 60 + d.dt.dt.minute
    d = d[(mm >= 570) & (mm < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    if TF != '15m':
        d['d'] = d.dt.dt.normalize(); d['b'] = ((d.dt.dt.hour * 60 + d.dt.dt.minute - 570) // 60) if TF == '1h' else 0
        d = d.groupby(['d', 'b'], sort=True).agg(dt=('dt', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
    return d
def keyof(dtser):
    if TF == '15m': return dtser.values
    dd = dtser.dt.normalize(); b = ((dtser.dt.hour * 60 + dtser.dt.minute - 570) // 60) if TF == '1h' else 0
    return (dd + pd.to_timedelta(np.asarray(b) * 60, unit='m')).values
qk = keyof(H.dt)
def align(tk):
    E = load_etf(tk); E['k'] = keyof(E.dt); E = E.drop_duplicates('k').set_index('k'); op = E.open.reindex(qk).values; cl = E.close.reindex(qk).values; nmiss = int(np.isnan(op).sum()); cl = pd.Series(cl).ffill().values; op = pd.Series(op).fillna(pd.Series(cl).shift(1)).ffill().values
    chg = np.abs(np.diff(np.log(E.close.values))); print(tk, 'bars', len(E), 'missing vs QQQ (filled from prev close)', nmiss, 'max |bar log-ret|', round(float(chg.max()), 3), flush=True); return op, cl
oT, cT = align('TQQQ'); oS, cS = align('SQQQ')
def dayclose_of(cl):
    dc = np.full(len(ud), np.nan)
    for d in range(len(ud)):
        v = cl[first[d]:last_[d] + 1]; v = v[~np.isnan(v)]
        if len(v): dc[d] = v[-1]
    return pd.Series(dc).ffill().values
dcT, dcS = dayclose_of(cT), dayclose_of(cS)
trades2 = [t for t in trades if not (np.isnan(oT[t['e']]) or np.isnan(oT[t['x']]) or np.isnan(oS[t['e']]) or np.isnan(oS[t['x']]))]; print('trades', len(trades), 'usable', len(trades2), flush=True)
T = pd.DataFrame(trades2)
def parts_real(e, x, o, dc):
    out = []; de, dx = di[e], di[x]
    for d in range(de, dx + 1):
        a = o[e] if d == de else dc[d - 1]; b = dc[d] if d < dx else o[x]
        if d == dx and d == de: b = o[x]
        out.append((d, b / a - 1))
    return out
ret, PARTS = [], []
for r in T.itertuples():
    o, dc = (oT, dcT) if r.side == 1 else (oS, dcS); ret.append(o[r.x] / o[r.e] - 1 - 4e-4); PARTS.append(parts_real(r.e, r.x, o, dc))
T['ret'] = ret; syn = []
for r in T.itertuples():
    v = 1.0
    for d, q in daily_parts(r.e, r.x): v *= 1 + 3 * r.side * q
    syn.append(v - 1 - drag_d(pd.Timestamp(ud[di[r.e]]).year, max((pd.Timestamp(ud[di[r.x]]) - pd.Timestamp(ud[di[r.e]])).days, 0)) - 4e-4)
T['syn'] = syn; T['hold'] = [(H.dt[r.x] - H.dt[r.e]).total_seconds() / 86400 for r in T.itertuples()]; T['qqq'] = [r.side * (o_[r.x] / o_[r.e] - 1) for r in T.itertuples()]
T['mon'] = T.entry.dt.to_period('M'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); nd = len(ud)
print('synthetic vs real per-trade (bull): corr', round(T[T.side == 1][['ret', 'syn']].corr().iloc[0, 1], 3), 'mean real', round(T[T.side == 1].ret.mean() * 100, 2), '% synthetic', round(T[T.side == 1].syn.mean() * 100, 2), '%')
def tst(A, B):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.ret[x.s == 1].sum(), s0=x.ret[x.s == -1].sum()))); m1, m0 = A.ret.mean(), B.ret.mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 100, (m1 - m0) / se
f = lambda x: f"n{len(x):4d} win {(x.ret>0).mean()*100:4.1f} mean {x.ret.mean()*100:5.2f}% med {x.ret.median()*100:5.2f}% hold {x.hold.mean():4.1f}d (QQQ {x.qqq.mean()*100:5.2f}%)"
Bl = T[T.side == 1]; A_, B_ = Bl[Bl.neg], Bl[~Bl.neg]; d_, t_ = tst(A_, B_)
print('PER TRADE (bull flips -> REAL TQQQ)'); print(' ALL ', f(Bl)); print(' NEG ', f(A_)); print(' POS ', f(B_)); print(f' NEG-POS {d_:+.2f}pp t {t_:+.2f} | train {(A_[A_.half=="train"].ret.mean()-B_[B_.half=="train"].ret.mean())*100:+.2f} test {(A_[A_.half=="test"].ret.mean()-B_[B_.half=="test"].ret.mean())*100:+.2f}')
print(' by era NEG / POS mean %: ' + ' | '.join(f"{e} {A_[A_.era==e].ret.mean()*100:+.2f}(n{(A_.era==e).sum()}) / {B_[B_.era==e].ret.mean()*100:+.2f}(n{(B_.era==e).sum()})" for e in ('18-19', '20-21', '22', '23-26')))
Br = T[T.side == -1]; print('PER TRADE (bear flips -> REAL SQQQ)'); print(' ALL ', f(Br)); print(' NEG ', f(Br[Br.neg])); print(' POS ', f(Br[~Br.neg]))
def book(sel, nm):
    r = np.zeros(nd); held = np.zeros(nd, bool); S = T[sel].sort_values('e'); last_x = -1
    for k in S.index:
        r_ = T.loc[k]
        if r_.e <= last_x: continue
        last_x = r_.x
        for d, q in PARTS[k]:
            if not held[d]: r[d] = q; held[d] = True
        r[di[r_.e]] -= 2e-4; r[di[r_.x]] -= 2e-4
    idx = pd.DatetimeIndex(ud); s = pd.Series(r, index=idx)
    def stat(rr):
        e = (1 + rr).cumprod(); return e.iloc[-1] ** (252 / max(len(rr), 1)) - 1, (e / e.cummax() - 1).min(), rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan
    out = {k: stat(s[mask]) for k, mask in (('all', s.index >= '2018-07-01'), ('train', (s.index >= '2019-01-01') & (s.index < '2022-01-01')), ('test', (s.index < '2019-01-01') | (s.index >= '2022-01-01')))}
    print(f"{nm:30s} exposure {held.mean()*100:3.0f}% | " + ' | '.join(f"{k}: CAGR {v[0]*100:6.1f}% DD {v[1]*100:6.1f}% Sh {v[2]:4.2f}" for k, v in out.items()), flush=True)
print('\nBOOKS (real prices, one position at a time, daily marks)')
book(T.side == 1, 'A bull ungated -> TQQQ'); book((T.side == 1) & T.neg, 'B bull NEG -> TQQQ'); book((T.side == 1) & ~T.neg, 'C bull POS -> TQQQ'); book(T.side != 0, 'D bull TQQQ + bear SQQQ'); book(T.neg, 'E bull+bear NEG only')
for nm, dc in (('B&H real TQQQ', dcT),):
    s = pd.Series(np.r_[0, dc[1:] / dc[:-1] - 1], index=pd.DatetimeIndex(ud)); out = []
    for k, mask in (('all', s.index >= '2018-07-01'), ('train', (s.index >= '2019-01-01') & (s.index < '2022-01-01')), ('test', (s.index < '2019-01-01') | (s.index >= '2022-01-01'))):
        e = (1 + s[mask]).cumprod(); out.append(f"{k}: CAGR {(e.iloc[-1]**(252/mask.sum())-1)*100:6.1f}% DD {((e/e.cummax()-1).min())*100:6.1f}% Sh {s[mask].mean()/s[mask].std()*np.sqrt(252):4.2f}")
    print(f"{nm:30s} exposure 100% | " + ' | '.join(out))
