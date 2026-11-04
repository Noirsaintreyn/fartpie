"""Nov 4 2026: EMA overlays on the VECTOR's trades. DECLARED BEFORE RESULTS. Book test (40 matched runs vs live, both periods).
Entry filters (state at the last completed bar before the entry):  F1 1h close > 1h EMA50 | F2 daily close > daily EMA20 | F3 1h close > 1h EMA20 and 1h EMA20 > EMA50 | F4 15m EMA21 > EMA55 (the vector's own) | F5 F1 & F2
Exit on EMA rejection (replaces the live exit if it comes first; exit at the next RTH bar open):  X1 1h close < 1h EMA20 | X2 15m close < 15m EMA21 (armed after +1% MFE) | X3 daily close < daily EMA10 | X4 daily close < daily EMA20
Filters drop trades; exits shorten them (slot freed earlier)."""
exec(open('xs/ema_common.py').read())
rows = []
for i, r in XL.iterrows():
    B = BAR.get(r.tk); 
    if B is None: continue
    j = B['pos'].get(r.entry)
    if j is None or np.ndim(j) or j < 1: continue
    j = int(j); je = B['pos'].get(r.exit); jx = int(je) if (je is not None and not np.ndim(je)) else None; px = B['o'][j]; off = 0.0
    if jx is not None:
        g = B['o'][jx] / px - 1
        if abs(r.ret - g) < .005: off = r.ret - g
    rec = dict(i=i, F1=B['c1'][j - 1] > B['h50'][j - 1], F2=B['d_c'][j - 1] > B['d_e20'][j - 1], F3=(B['c1'][j - 1] > B['h20'][j - 1]) and (B['h20'][j - 1] > B['h50'][j - 1]), F4=B['e21'][j - 1] > B['e55'][j - 1])
    end = (jx if jx is not None else len(B['o']) - 1)
    for nm in ('X1', 'X2', 'X3', 'X4'):
        hit = None; armed = False
        for b in range(j + 1, end):
            if B['m'][b] < 570 or B['m'][b] >= 956: continue
            if nm == 'X1' and B['c1'][b - 1] < B['h20'][b - 1]: hit = b
            elif nm == 'X2':
                if B['h'][b - 1] / px - 1 >= .01: armed = True
                if armed and B['c'][b - 1] < B['e21'][b - 1]: hit = b
            elif nm == 'X3' and B['d_c'][b - 1] < B['d_e10'][b - 1]: hit = b
            elif nm == 'X4' and B['d_c'][b - 1] < B['d_e20'][b - 1]: hit = b
            if hit is not None: break
        if hit is not None: rec[nm + '_x'] = B['t'][hit]; rec[nm + '_r'] = B['o'][hit] / px - 1 + off - 0
    rows.append(rec)
F = pd.DataFrame(rows).set_index('i'); Z = XL.join(F, how='inner'); print('joined', len(Z), 'of', len(XL))
Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26')
for f in ('F1', 'F2', 'F3', 'F4'): print(f'  {f} true share {Z[f].mean():.0%};  avg trade true/false: ' + ' | '.join(f'{p} {g[g[f]].ret.mean()*100:+.2f}%/{g[~g[f]].ret.mean()*100:+.2f}%' for p, g in Z.groupby('per')))
B0 = ev(XL); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
def drop(nm, keep):
    Y = XL[XL.index.isin(Z[keep(Z)].index)].reset_index(drop=True); line(nm, ev(Y), B0); print(f'     kept {len(Y)}/{len(XL)}; runners kept {(Y.ret >= .15).sum()}/{(XL.ret >= .15).sum()}', flush=True)
drop('F1 1h close > EMA50', lambda z: z.F1); drop('F2 daily close > EMA20', lambda z: z.F2); drop('F3 1h close>EMA20>EMA50', lambda z: z.F3); drop('F4 15m EMA21>55', lambda z: z.F4); drop('F5 F1 & F2', lambda z: z.F1 & z.F2)
for nm in ('X1', 'X2', 'X3', 'X4'):
    Y = XL.copy(); m = Z[nm + '_r'].notna(); idx = m.index[m]; Y.loc[idx, 'exit'] = Z.loc[idx, nm + '_x']; Y.loc[idx, 'ret'] = Z.loc[idx, nm + '_r']
    print(f'  {nm}: shortened {m.sum()} ({m.mean():.0%}), runners hit {(Z[m].ret >= .15).sum()}/{(XL.ret >= .15).sum()}'); line(f'{nm} exit on EMA rejection', ev(Y), B0)
