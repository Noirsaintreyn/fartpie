src = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("B0 = ev(XL)")[0].replace('stop1h_V.parquet', 'level_V.parquet'); exec(src)
B0 = ev(XL); print('40 matched runs, 2019-21 / 2022-26'); line('CURRENT', B0)
for nm, lab in (('AH', 'AH sell after-hours on a break of the stop level'), ('AH2', 'AH2 only on a 2-ATR break')):
    Xv = build(nm); h = Xv[~Xv.h1]; print(f'  {nm}: 15m trades {len(h)}, avg trade {h.ret.mean():+.3%} (current {XL[~XL.h1].ret.mean():+.3%}), worst-5% {h.ret.quantile(.05):+.1%} (current {XL[~XL.h1].ret.quantile(.05):+.1%})')
    line(lab, ev(Xv), B0)
