"""Nov 12 2026: multi-timeframe confluence for the 1h vector flip. DECLARED BEFORE RESULTS. Hypothesis: a 1h bull flip that agrees with the 4h and daily vector regime (both already bull on the last COMPLETED bars) is less chop than a lone 1h flip;
combine with the flip-bar volume (>=1.5x) and the low-vol day flag (C). Sets (entry next 1h open after the flip, exit next open after the 1h bear flip, 40 bp, price band, since 2024-01): base | +4h bull | +D bull | +4h&D | each x B (vol) | x C (vol+lowvol) | all four.
Train < 2025-07-01 | test after, nothing fitted. 15m skipped: Yahoo keeps ~60 days of 15m bars. Bands: $1-2 (screener) and $1-5 (sample size)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004
def htf(Y, tf):
    day = Y.index.normalize(); g = [day, (Y.index.hour >= 13).astype(int)] if tf == '4h' else [day]
    G = Y.groupby(g); o = G.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'))
    first = G.apply(lambda x: x.index[0]); last = G.apply(lambda x: x.index[-1]); o.index = pd.DatetimeIndex(first.values)
    end = pd.DatetimeIndex(last.values) + pd.to_timedelta(np.where(pd.DatetimeIndex(last.values).hour == 15, 30, 60), unit='m')
    reg = run_state_machine(o.assign(datetime=o.index))[0]; return pd.DataFrame({'end': end, 'reg': reg}).sort_values('end')
def trades(tk):
    Y = M.closed_only(data[tk].dropna(subset=['open', 'close']))
    if len(Y) < 200: return []
    F = M.features(Y); h4, hd = htf(Y, '4h'), htf(Y, '1d'); cl = pd.DataFrame({'t': Y.index + pd.to_timedelta(np.where(Y.index.hour == 15, 30, 60), unit='m')})
    r4 = pd.merge_asof(cl, h4, left_on='t', right_on='end', direction='backward').reg.values; rd = pd.merge_asof(cl, hd, left_on='t', right_on='end', direction='backward').reg.values
    reg = F.reg.values; o = Y.open.values; c = Y.close.values; out = []; ent = None
    for i in range(1, len(Y) - 1):
        if reg[i] == reg[i - 1] or reg[i] == 0: continue
        if reg[i] == 1 and ent is None: ent = (i + 1, F.vrel.iat[i], bool(F.lv.iat[i]), r4[i] == 1, rd[i] == 1, c[i])
        elif reg[i] == -1 and ent is not None: out.append(dict(tk=tk, et=Y.index[ent[0]], xt=Y.index[i + 1], epx=o[ent[0]], sig=ent[5], vr=ent[1], lv=ent[2], a4=ent[3], ad=ent[4], ret=o[i + 1] / o[ent[0]] - 1)); ent = None
        elif reg[i] == -1: ent = None
    return out
def book(T, K=8):
    eq = 1.0; op = []; curve = []
    for r in sorted(T.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))):
        still = []
        for p in op:
            if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
    return pd.Series(dict(curve)).sort_index()
def rep(nm, T):
    if len(T) < 15: print(f'  {nm:<24} {len(T)} trades (too few)'); return
    r = T.ret - COST; a, b = r[T.et < '2025-07-01'], r[T.et >= '2025-07-01']; s = book(T); yrs = (s.index[-1] - T.et.min()).days / 365.25; cagr = s.iloc[-1] ** (1 / max(yrs, .5)) - 1
    print(f'  {nm:<24} {len(T):>5} tr {len(T)/yrs:>4.0f}/yr | mean {r.mean()*100:+.2f}% win {(r>0).mean():.0%} p10 {r.quantile(.1)*100:+.1f}% | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cagr:+.0%}/yr DD {(s/s.cummax()-1).min():.0%}', flush=True)
T = pd.DataFrame([x for tk in data for x in trades(tk)]); T = T[T.et >= '2024-01-01'].reset_index(drop=True); T.to_pickle(SC + 'mtf_T.pkl')
for nm, band in (('$1-2', (T.sig >= 1) & (T.sig < 2)), ('$1-5', (T.sig >= 1) & (T.sig < 5))):
    X = T[band]; print(f'== {nm}: {len(X)} flips')
    for sn, m in (('base', X.vr > -1), ('+4h bull', X.a4), ('+D bull', X.ad), ('+4h & D bull', X.a4 & X.ad), ('NOT 4h bull', ~X.a4)):
        print(f' [{sn}]'); Z = X[m]; rep('all flips', Z); rep('vol>=1.5x (B)', Z[Z.vr >= 1.5]); rep('B + lowvol (C)', Z[(Z.vr >= 1.5) & Z.lv]); rep('lowvol only', Z[Z.lv])
