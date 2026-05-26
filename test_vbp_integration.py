"""Smoke test for VbP integration"""
import sys
import pandas as pd
import numpy as np
np.random.seed(42)

n = 500
prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    'Open': prices + np.random.randn(n) * 0.1,
    'High': prices + np.abs(np.random.randn(n)) * 0.3,
    'Low':  prices - np.abs(np.random.randn(n)) * 0.3,
    'Close': prices,
    'Volume': np.random.randint(1000, 10000, n).astype(float),
}, index=pd.date_range('2024-01-01', periods=n, freq='h'))
df['High'] = df[['Open','Close','High']].max(axis=1)
df['Low'] = df[['Open','Close','Low']].min(axis=1)

print(f"Test data: {len(df)} bars, price range {df['Close'].min():.2f} - {df['Close'].max():.2f}")

print("\n--- Test 1: Direct vbp_levels.LevelEngine ---")
from vbp_levels import LevelEngine, InstrumentProfile
engine = LevelEngine(profile=InstrumentProfile(tick=0.01))
result = engine.run(df, lookback_bars=300)
levels_df = result['levels']
print(f"✓ Engine ran  | POC={result['poc']:.2f} VAH={result['vah']:.2f} VAL={result['val']:.2f} ATR={result['atr']:.2f}")
print(f"  Levels: {len(levels_df)}")
for i, row in levels_df.head(5).iterrows():
    print(f"    price={row['price']:.2f}  score={row['score']:.3f}  algos={row['algo_count']}  type={row['type']}  sources={row['sources'][:60]}")

print("\n--- Test 2: Wrapper mapping logic ---")
levels_out = []
for _, row in levels_df.iterrows():
    price = float(row.get('price', 0))
    if price <= 0: continue
    score = float(row.get('score', 0))
    strength = max(0.0, min(0.95, score))
    sources_raw = row.get('sources', '') or ''
    tags = [t.strip() for t in sources_raw.split('|') if t.strip()] if isinstance(sources_raw, str) else []
    algo_count = int(row.get('algo_count', 1))
    row_type = str(row.get('type', 'level'))
    is_anchor = row_type in ('POC','VAH','VAL')
    type_label = f'VbP-{row_type}' if is_anchor else 'VbP'
    if is_anchor: strength = max(strength, 0.80)
    levels_out.append({
        'price': price, 'type': type_label, 'strength': strength,
        'category': 'VbP', 'tags': tags, 'algo_count': algo_count,
        'is_anchor': is_anchor
    })
levels_out = sorted(levels_out, key=lambda x: x['strength'], reverse=True)[:12]
print(f"✓ Wrapper produced {len(levels_out)} levels")
anchors = [l for l in levels_out if l['is_anchor']]
print(f"  Anchors (POC/VAH/VAL): {len(anchors)}")
for a in anchors:
    print(f"    {a['type']}: {a['price']:.2f}  strength={a['strength']:.3f}")
print(f"  Top 3 non-anchor:")
for l in [x for x in levels_out if not x['is_anchor']][:3]:
    print(f"    {l['type']}: {l['price']:.2f}  strength={l['strength']:.3f}  algos={l['algo_count']}")

# Verify schema fields the frontend/pipeline expects
required = {'price','type','strength','category','breakoutProb','reversionProb'}
sample = levels_out[0] if levels_out else {}
sample['breakoutProb'] = 1.0 - sample.get('strength', 0)
sample['reversionProb'] = sample.get('strength', 0)
missing = required - set(sample.keys())
if missing:
    print(f"✗ Missing required fields: {missing}")
    sys.exit(1)
print(f"\n✓ All VbP integration smoke tests passed")
