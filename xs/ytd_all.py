"""Nov 3 2026: 2026 YTD (Jan 1 - Sep 15) for every combo cell, median of 21 matched runs."""
src = open('/home/claude/fartpie/xs/combo.py').read().split("PX = {nm")[0]; exec(src)
A, B = pd.Timestamp('2025-12-31'), pd.Timestamp('2026-09-15')
def runs(Xv, mode, n=21):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Xv.tk, Xv.entry.astype('int64'))], dtype=np.uint64); out = []
    for i in range(0, n):
        m = np.ones(len(Xv), bool) if i == 0 else ~(((HSH * 2654435761 + i * 40503) % 1000) < 50)
        if mode == 'D': m = m & ~Xv.low.values
        d = Xv[m]; s = simx(d.entry.values, d.exit.values, d.ret.values, d.ratio.values.astype(float), d.tk.values, d.low.values, d.high.values, mode)
        e = np.exp(s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0)); b0 = e[e.index <= A].iloc[-1]; w = e[(e.index > A) & (e.index <= B)]
        seg = np.r_[b0, w.values]; out.append((w.iloc[-1] / b0 - 1, (seg / np.maximum.accumulate(seg) - 1).min()))
    return np.array(out)
PX = {nm: prep(nm) for nm in ('L', 'A5', 'Ainf')}; print('2026 YTD (Jan 1 - Sep 15)   return / max DD   (median of 21 matched runs; min..max)')
for nm in ('L', 'A5', 'Ainf'):
    for ov in ('none', 'M1', 'M3', 'D', 'M3+M1'):
        r = runs(PX[nm], ov); print(f'  {nm:<5}+ {ov:<7} {np.median(r[:, 0]):+6.1%} / {np.median(r[:, 1]):6.1%}   ({r[:, 0].min():+.0%}..{r[:, 0].max():+.0%})', flush=True)
