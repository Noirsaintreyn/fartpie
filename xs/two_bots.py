"""Nov 3 2026: which bot's bar convention is better? live_screener.py (Render / Discord) builds the vector on regular-session bars only (RTH0 = 570);
the backtest engine and the Mac screener use premarket + regular session (m >= 240).  Same engine, same options-ratio universe, same book rules.
 PRE  premarket + regular session bars (backtest / Mac)       RTH  regular-session bars only (Render screener default)
Both tracks (15m and 1h) are rebuilt for each convention; book = 6 slots, ratio tie-break, 8% breaker, 40 matched runs, both periods."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
if __name__ == '__main__':
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); rows = []
    for tk in sorted(Xo.tk.unique()):
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0])
        for nm, H_ in (('PRE', H), ('RTH', H[H.m >= 570].reset_index(drop=True))):
            reg0, s10 = states(H_, vec)
            rows += [(tk, nm, False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run(H_, reg0, s10, False)[0]]
            rows += [(tk, nm + '_H1', True, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run(H_, reg0, s10, True)[0]]
        print(tk, flush=True)
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet('/tmp/claude-0/ml/twobots_V.parquet')
