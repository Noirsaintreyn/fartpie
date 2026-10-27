"""Oct 27 2026: which SPY vector timeframe defines the 'downtrend' for 1.5x entry sizing? SPY vector on 15m / 30m / 45m / 1h / 4h (bars from
premarket+RTH 15m, RTH-aligned at 9:30) and daily; state of the last bar completed before each entry. With and without SPY under its
200-day. 1.5x size on entries in the downtrend, 1x otherwise; live book otherwise."""
import glob, numpy as np, pandas as pd
src = open('xs/market_gate.py').read(); exec(src[:src.index("def show(nm, Z, cap=False)")])
ss2 = ss.replace('def sim_cap(', 'def sim_cw(').replace("for e, x, r_, ww, cp in", "for e, x, r_, ww, cp, wn in").replace("Z.w.values, Z.cap.values)", "Z.w.values, Z.cap.values, Z.wn.values)").replace("pend.append((x, r_, ww / n))", "pend.append((x, r_, ww / wn))")
exec(ss2)
f = glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/SPY_15m_*.parquet') or glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/SPY_15m_*.parquet')
H = load(f[0]); R = H[H.m >= 570]
def tf_state(freq, minutes):
    b = H if freq == '15min' else agg(R, freq, '9h30min')
    st = np.asarray(vec(b[['dt', 'open', 'high', 'low', 'close']] if 'dt' in b else b)); k = pd.Series(st, index=pd.to_datetime(b.dt) + pd.Timedelta(minutes=minutes))
    return k[~k.index.duplicated(keep='last')].sort_index()
TF = {'15m': tf_state('15min', 15), '30m': tf_state('30min', 30), '45m': tf_state('45min', 45), '1h': tf_state('1h', 60), '4h': tf_state('4h', 240)}
ent = L.entry.values
def at(s): i = np.searchsorted(s.index.values, ent, side='right') - 1; return np.where(i >= 0, s.values[np.clip(i, 0, None)], 0)
u200 = at(S['sma200'].rename(None).set_axis(S.index + pd.Timedelta(hours=9, minutes=30))) < 0     # S already prior-close; usable from the open
dD = at(S['vec_d'].set_axis(S.index + pd.Timedelta(hours=9, minutes=30))) < 0
print('SPY 15m bars through', H.dt.max())
def show2(nm, mask):
    z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = np.where(mask, 1.5, 1.0); R_ = [sim_cw(z, a, b) for a, b in PER]
    b = L[L.inbook]; mm = mask[L.inbook.values]; g = b.groupby([np.where(mm, 'down', 'other'), 'half']).ret.mean().unstack() * 100
    print(f'  {nm:<34}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + '  ret/DD ' + ' '.join(f'{c / -d:.1f}' for c, d in R_[:2]) +
          (f'  share {mask.mean():.0%}  admitted avg down {g.loc["down", "19-21"]:+.2f}/{g.loc["down", "22-26"]:+.2f}%' if 'down' in g.index else ''), flush=True)
print(f'{"":<36}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show2('live (no sizing)', np.zeros(len(L), bool))
for nm, s in list(TF.items()) + [('daily', None)]:
    bear = dD if s is None else at(s) < 0
    show2(f'SPY {nm} vector bear', bear); show2(f'SPY {nm} vector bear + under 200d', bear & u200)
show2('SPY under 200d only', u200)
