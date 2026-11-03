"""Score the Hawkes pairings (xs/hawkes_pair.py) with the matched-runs yardstick vs the CURRENT model and vs Ainf (1h-confirm, no hard stop)."""
src = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("B0 = ev(XL)")[0].replace('stop1h_V.parquet', 'hawkes_V.parquet'); exec(src)
B0 = ev(XL); BA = ev(build('Ainf')); print('40 matched runs, 2019-21 / 2022-26'); line('CURRENT (L)', B0); line('Ainf (1h-confirm, no hard stop)', BA, B0)
for nm in ('AH4_1.0', 'AH4_1.5', 'AH4_2.0', 'AH8_1.0', 'AH8_1.5', 'AH8_2.0', 'LH8_1.5'):
    Xv = build(nm); h = Xv[~Xv.h1]; E_ = ev(Xv)
    print(f'  {nm}: trades {len(Xv)}, 15m avg trade {h.ret.mean():+.3%}, hold {((h.exit - h.entry).dt.total_seconds() / 86400).mean():.1f}d, worst-5% trade {h.ret.quantile(.05):+.1%}')
    line(f'{nm} vs current', E_, B0); line(f'{nm} vs Ainf', E_, BA)
