"""Nov 1 2026: vector settings sweep in the live engine (same as xs/trend_models.py, models = the vector with one setting moved).
python3 xs/par_sweep.py BARS_DIR OUT_DIR WIDE_T  (SHARD / NSHARD env)"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = A; os.makedirs(OUT, exist_ok=True)
VAR = {'vec': {}, 'piv5': dict(PIVOT_LEN=5), 'piv6': dict(PIVOT_LEN=6), 'piv10': dict(PIVOT_LEN=10), 'piv12': dict(PIVOT_LEN=12),
       'brk0.2': dict(BREAK_ATR=.2), 'brk0.25': dict(BREAK_ATR=.25), 'brk0.45': dict(BREAK_ATR=.45), 'brk0.5': dict(BREAK_ATR=.5),
       'atr10': dict(ATR_LEN=10), 'atr20': dict(ATR_LEN=20), 'mm0.5': dict(MIN_MOVE_ATR=.5), 'mm1.2': dict(MIN_MOVE_ATR=1.2),
       'conf2': dict(CONFIRM_BARS=2), 'ema13_34': dict(FAST_LEN=13, SLOW_LEN=34), 'ema34_89': dict(FAST_LEN=34, SLOW_LEN=89)}
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []
    for nm, kw in VAR.items():
        reg, s1 = states(H, (lambda k: (lambda d: vec(d, **k)))(kw))
        for h1t in (False, True): rows += [(tk, nm, h1t) + t for t in run(H, reg, s1, h1t)[0]]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
