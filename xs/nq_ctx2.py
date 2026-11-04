_s = open('xs/nq_ctx.py').read(); exec(_s[:_s.index("for T, S in ((100, 50), (150, 75)):")])
T, S = 150, 75
for nm, msk in (('all S0', np.ones(len(C2), bool)), ('SPY 20d up', C2.MKT20 == 'SPY 20d up'), ('VIX low', C2.VIXP == 'VIX low'), ('gamma shallow', C2.GEXQ == 'gamma shallow'), ('SPY z20 normal', C2.Z20 == 'normal'),
                ('combo (exploratory): SPY 20d up & VIX not high', (C2.MKT20 == 'SPY 20d up') & (C2.VIXP != 'VIX high'))):
    sub = C2[msk]; R2 = run_q(sub, T, S); pv, md_, n_ = prop(R2, days)
    print(f'  {nm:<48} trades {len(R2)} ({len(R2) / len(days) * 5:.2f}/wk) win {R2.pts.gt(0).mean():.0%} avg {R2.pts.mean():+.1f} | eval pass {pv:.0%}, median {md_:.0f} sessions to pass', flush=True)
