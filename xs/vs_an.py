"""Nov 2 2026: Q1 (transition mechanism) + Q2 (boundary persistence) tables from xs/vec_state.py output. Discovery only.
Per group and period: trades, avg trade, big-winner ratio, median bars held (15m), median worst drawdown inside the trade."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
Y = pd.read_parquet('/tmp/claude-0/ml/vs_flips.parquet'); Y = Y[Y.ok == True].copy(); Y['h'] = np.where(Y.entry.dt.year <= 2021, '2019-21', '2022-26')
MAE, HB = np.full(len(Y), np.nan), np.full(len(Y), np.nan)
for tk, g in Y.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet'); H = pd.read_parquet(f[0], columns=['datetime', 'open', 'low'])
    H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); t = H.dt.values; lo = H.low.values; op = H.open.values
    for k, (e, x) in zip(g.index, zip(g.entry.values, g.exit.values)):
        a, b = np.searchsorted(t, e), np.searchsorted(t, x)
        if a < len(t) and b > a: p = Y.index.get_loc(k); MAE[p] = lo[a:b].min() / op[a] - 1; HB[p] = b - a
Y['mae'], Y['bars'] = MAE, HB
top = Y.groupby('h').ret.transform(lambda s: s >= s.quantile(.95))
def tab(title, groups):
    print(f'\n{title}\n  {"group":<34}{"share":>14}{"avg trade":>20}{"big-winner ratio":>20}{"bars held (med)":>18}{"worst DD in trade (med)":>26}')
    for nm, m in groups:
        m = m.fillna(False).astype(bool); s = []
        for h in ('2019-21', '2022-26'):
            z = Y.h == h; mm = m & z; s.append((mm.sum() / z.sum(), Y.ret[mm].mean(), (top[mm].sum() / top[z].sum()) / (mm.sum() / z.sum()) if mm.sum() else np.nan, Y.bars[mm].median(), Y.mae[mm].median()))
        print(f'  {nm:<34}' + f'{s[0][0]:>6.0%} /{s[1][0]:>5.0%}' + f'{s[0][1]:>+10.2%} /{s[1][1]:>+7.2%}' + f'{s[0][2]:>11.2f} /{s[1][2]:>5.2f}' + f'{s[0][3]:>10.0f} /{s[1][3]:>4.0f}' + f'{s[0][4]:>16.1%} /{s[1][4]:>6.1%}', flush=True)
print(f'{len(Y)} trades with a located flip (2019-21 / 2022-26 in each cell)')
px, lv, at = Y.need_px.astype(bool), Y.need_lv.astype(bool), Y.need_at.astype(bool)
tab('Q1 which change was necessary for the break (counterfactual: freeze each at its previous-bar value)', [
    ('price needed', px), ('level move needed', lv), ('ATR change needed', at),
    ('price only', px & ~lv & ~at), ('level only (level came to price)', lv & ~px), ('price AND level', px & lv), ('ATR involved', at),
    ('none alone (all small)', ~px & ~lv & ~at)])
dom = Y[['dC', 'dL', 'dA']].idxmax(axis=1)
tab('Q1b largest component of the trigger change (in ATR)', [('price push largest', dom == 'dC'), ('level drop largest', dom == 'dL'), ('ATR shrink largest', dom == 'dA')])
q = lambda c: Y.groupby(Y.entry.dt.year)[c].rank(pct=True)
tab('Q2 boundary persistence', [
    ('level age: youngest third', q('age') <= 1 / 3), ('level age: middle third', (q('age') > 1 / 3) & (q('age') <= 2 / 3)), ('level age: oldest third', q('age') > 2 / 3),
    ('level set at the bear flip (never ratcheted)', Y.reason == 1), ('level from a ratchet (pivot)', Y.reason == 2),
    ('no earlier near-approach', Y.appr == 0), ('1-2 near-approaches', Y.appr.between(1, 2)), ('3+ near-approaches', Y.appr >= 3),
    ('higher low before the flip', Y.hl == 1), ('no higher low', Y.hl == 0), ('lower high before the flip', Y.lh == 1)])
for tr, nm in ((False, '15m track'), (True, '1h track')):
    Z = Y[Y.h1 == tr]; print(f'\n  {nm}: price needed {Z.need_px.mean():.0%}, level needed {Z.need_lv.mean():.0%}, ATR needed {Z.need_at.mean():.0%}; level age median {Z.age.median():.0f} bars')
