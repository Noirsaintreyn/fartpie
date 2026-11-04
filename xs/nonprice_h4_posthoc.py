src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
XL = pd.read_parquet('/tmp/claude-0/ml/nonprice_X.parquet')
B0 = ev(XL); line('L live', B0)
for w in (.25, 1.0):
    Y = XL.copy(); Y['ratio'] = XL.ratio * (1 + w * (XL.hi52 < -.25).astype(float)); line(f'H4 POST-HOC bonus >25% below 52w high w{w}', ev(Y), B0)
Y = XL.copy(); Y['ratio'] = XL.ratio * (1 + .25 * (-XL.hi52.fillna(0)).clip(0, 1) * 4); line('H5 POST-HOC bonus scaled by depth below high', ev(Y), B0)
