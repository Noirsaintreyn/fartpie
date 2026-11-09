"""Nov 9 2026: THE PORTFOLIO AS AN INCOME - payout rules on the live vector screener's honest daily backtest (xs/ext/site_wide_daily.parquet: point-in-time S&P,
ratio >= 1.5, 6 slots, sticky, 8% breaker, no entries after 14:00, 2 bp a trade; Jan 2019 - Sep 2026). No new money is ever added. Payouts at each month-end close.
Rules:  FIXED w     pay w% of current equity every month (w = 1, 1.5, 2, 2.5)
        HWM p       pay p% of the equity above its high-water mark (after the last payout) - like a prop payout; nothing in months below the mark (p = 50, 100)
        FLOOR w     pay w% of current equity, but only while equity is above the starting capital (keeps the principal intact)
        SMOOTH w    pay w% of the 12-month average equity (endowment rule: income moves slowly)
Report, per $10,000 started: average monthly income, months with no income, worst and best 12 months of income, ending equity, deepest equity drawdown;
started Jan 2019 and started Jan 2022 (a bad first year)."""
import numpy as np, pandas as pd
J = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); r = J.m
def run(rule, w, start):
    x = r[r.index >= start]; eq = 10000.0; hwm = eq; hist = []; pays = []; path = []
    me = x.groupby(x.index.to_period('M')).apply(lambda z: (1 + z).prod() - 1)
    for mo, g in me.items():
        eq *= 1 + g; hist.append(eq); pay = 0.0
        if rule == 'FIXED': pay = w * eq
        elif rule == 'HWM':
            if eq > hwm: pay = w * (eq - hwm)
        elif rule == 'FLOOR':
            if eq > 10000: pay = min(w * eq, eq - 10000)
        elif rule == 'SMOOTH': pay = w * np.mean(hist[-12:])
        eq -= pay; pays.append(pay); path.append(eq)
        if rule == 'HWM': hwm = max(hwm, eq)
    P = pd.Series(pays, index=me.index); E = pd.Series(path, index=me.index); r12 = P.rolling(12).sum()
    return dict(avg=P.mean(), zero=np.mean(P <= 0), lo12=r12.min(), hi12=r12.max(), end=E.iloc[-1], dd=(E / E.cummax() - 1).min(), yr=P.groupby(P.index.year).sum())
if __name__ == '__main__':
    print(f'screener backtest: {r.index[0].date()} - {r.index[-1].date()}, CAGR {(1 + r).prod() ** (252 / len(r)) - 1:+.1%}', flush=True)
    for start in ('2019-01-01', '2022-01-01'):
        print(f'\nstarted {start[:7]} with $10,000:  avg / month | months with no income | worst 12 mo | best 12 mo | equity at the end | deepest drawdown | income by year', flush=True)
        for rule, ws in (('FIXED', (.01, .015, .02, .025)), ('FLOOR', (.015, .02)), ('SMOOTH', (.015, .02)), ('HWM', (.5, 1.0))):
            for w in ws:
                o = run(rule, w, start); yrs = ' '.join(f"{y % 100:02d}:{v / 1000:.1f}k" for y, v in o['yr'].items())
                print(f'  {rule:<6} {w * 100:>4.1f}%  ${o["avg"]:6.0f} | {o["zero"]:4.0%} | ${o["lo12"]:6.0f} | ${o["hi12"]:6.0f} | ${o["end"]:7.0f} | {o["dd"]:5.0%} | {yrs}', flush=True)
