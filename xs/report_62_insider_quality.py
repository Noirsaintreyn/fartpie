"""Nov 14 2026 INSIDER-BUY QUALITY on the wide daily flips (declared before results). Literature: officer purchases and large purchases carry more information than director/small ones. Same flips as report_60 (>=1 insider buyer in prior 30 days is the lead). Within flips with BV>=1.5 and (separately) all flips: OFFICER (any Officer relationship buyer) vs DIR-ONLY; dollar value of purchases in the 30 days (sum shares*price) >= $100k and >= $500k. Strict eras 2019-21 / 2022-26Q1, 30bp, month-clustered t of the feature group vs no-insider flips."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_ins.pkl'); I = pd.read_pickle(SC + 'insider_P_hist.pkl'); I = I[I.fd.notna()].copy(); I['sym'] = I.ISSUERTRADINGSYMBOL.str.upper().str.strip()
I['val'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce'); I['off'] = I.RPTOWNER_RELATIONSHIP.str.contains('Officer', case=False); Ig = {k: g.sort_values('fd') for k, g in I.groupby('sym')}
P['OFF'] = False; P['VAL'] = 0.0; P['ANY'] = False
for tk, ix in P.groupby('tk').groups.items():
    g = Ig.get(tk.upper())
    if g is None: continue
    f = g.fd.values; o = g.off.values; v = g.val.fillna(0).values; t = (P.loc[ix, 't'].dt.normalize() - pd.Timedelta(days=1)).values
    for j, i in enumerate(ix):
        m = (f <= t[j]) & (f > t[j] - np.timedelta64(30, 'D'))
        if m.any(): P.loc[i, 'ANY'] = True; P.loc[i, 'OFF'] = bool(o[m].any()); P.loc[i, 'VAL'] = v[m].sum()
P['ret_n'] = P.ret - 0.003
def mc(S, m, b):
    s = S[m & b]; r = S[(~S.ANY) & b]; a = s.groupby(s.t.dt.to_period('M')).ret_n.mean(); c = r.groupby(r.t.dt.to_period('M')).ret_n.mean(); dm = (a - c).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
G = {'any insider buy': lambda S: S.ANY, 'OFFICER buyer': lambda S: S.ANY & S.OFF, 'director-only': lambda S: S.ANY & ~S.OFF, 'value >= $100k': lambda S: S.ANY & (S.VAL >= 1e5), 'value >= $500k': lambda S: S.ANY & (S.VAL >= 5e5), 'officer & >= $100k': lambda S: S.ANY & S.OFF & (S.VAL >= 1e5)}
for en, a, b in (('2019-21', '2019', '2022'), ('2022-26Q1', '2022', '2026-04'), ('ALL', '2019', '2026-04')):
    S = P[(P.t >= a) & (P.t < b)].reset_index(drop=True); print(f'\n=== {en}')
    for bn, base in (('BV>=1.5', S.BV >= 1.5), ('all flips', pd.Series(True, index=S.index))):
        z = S[base & ~S.ANY]; print(f'  -- {bn}: no-insider mean {z.ret_n.mean()*100:+.2f}% (n {len(z)})')
        for k, fn in G.items():
            m = fn(S); s = S[m & base]
            if len(s) < 5: print(f'     {k:22s} n {len(s)}'); continue
            print(f'     {k:22s} n {len(s):5d} | mean {s.ret_n.mean()*100:+6.2f}% | med {s.ret_n.median()*100:+5.2f} | win {(s.ret_n>0).mean()*100:4.1f} | big {(s.ret_n>=0.15).mean()*100:4.1f} | t vs no-insider {mc(S, m, base):+.1f}')
