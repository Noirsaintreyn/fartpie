src = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("B0 = ev(XL)")[0].replace('stop1h_V.parquet', 'hours_V.parquet'); exec(src)
B0 = ev(XL); print('40 matched runs, 2019-21 / 2022-26'); line('CURRENT (original data)', B0)
BC = ev(build('C')); line('C  extended-bar vector, RTH exits (control)', BC, B0)
for nm, lab in (('PM', 'PM exits allowed in premarket'), ('ON', 'ON exits allowed overnight (pre + after-hours)')):
    Xv = build(nm); h = Xv[~Xv.h1]; E_ = ev(Xv)
    print(f'  {nm}: 15m trades {len(h)}, exits in extended hours {h.extexit.mean():.0%}, avg trade {h.ret.mean():+.3%}, worst-5% {h.ret.quantile(.05):+.1%}, avg hold {((h.exit - h.entry).dt.total_seconds() / 86400).mean():.2f}d')
    line(f'{lab} vs current', E_, B0); line(f'{lab} vs control', E_, BC)
h = build('C'); h = h[~h.h1]; print(f'  C control: 15m trades {len(h)}, avg trade {h.ret.mean():+.3%}, worst-5% {h.ret.quantile(.05):+.1%}; original: {XL[~XL.h1].ret.mean():+.3%}')
