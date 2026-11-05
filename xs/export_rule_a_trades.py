"""Nov 5 2026: export the Rule A trade log (1 lot, 75 / 50, flat 12:00, 1 pt cost, NQ points) that the account-start COLD signal reads (live_data/nq_rule_a_trades.csv)."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
full = seq('T75'); rows = [(pd.Timestamp(XA.at[i, 'day']).date(), pd.Timestamp(t5[int(XA.at[i, 'j'])]), int(XA.at[i, 'dir']), round(v[0] - 1.0, 2), v[3]) for i, v in full.items()]
L = pd.DataFrame(rows, columns=['day', 'entry', 'dir', 'net_pts', 'exit']).sort_values('entry'); os.makedirs('live_data', exist_ok=True); L.to_csv('live_data/nq_rule_a_trades.csv', index=False)
print(len(L), 'trades', L.day.min(), '->', L.day.max(), '| last 20 net', round(L.net_pts.iloc[-20:].sum(), 1), flush=True)
