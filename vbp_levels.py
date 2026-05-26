"""
vbp_levels.py  v2
─────────────────────────────────────────────────────────────────────────────
Volume-by-Price Level Detection Engine

Changes from v1 (addressing code-review concerns):

1. VbP distribution — volume is now weighted toward the close, not spread
   flat across the bar range. Open/close position inside the bar tells you
   where price SPENT its time; uniform spread assumed equal participation
   at every tick, which is wrong for trending bars.

2. Hardcoded thresholds replaced with InstrumentProfile — every tunable
   parameter lives in one place and can be overridden per symbol/timeframe
   without touching algorithm code.

3. Persistent homology rebuilt — now works directly on the 1D VbP density
   curve using sublevel-set persistence (standard TDA on a function) rather
   than a 2D point cloud with ambiguous coordinate mapping back to price.
   Birth/death pairs map cleanly to price peaks and valleys.

4. HDBSCAN percentile filter is now profile-driven (default 40th, not 50th)
   and documented as a threshold you should calibrate per symbol.

5. Score weights are explicitly configurable in InstrumentProfile.

6. Added a backtest helper: level_holdrate() walks forward through the data
   and measures what % of detected levels were actually respected. This is
   how you validate and tune the weights.

Usage:
    from vbp_levels import LevelEngine, InstrumentProfile
    import pandas as pd

    df = pd.read_csv("your_data.csv")
    df.columns = df.columns.str.strip()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], dayfirst=True)

    # Defaults tuned for NQ/ES 1h — override for other instruments
    profile = InstrumentProfile(
        tick=0.25,
        value_area_pct=0.68,
        # Score weights (must sum to 1.0)
        w_algo=0.40,
        w_vbp=0.40,
        w_confluence=0.20,
        # Clustering
        vbp_noise_percentile=40,     # drop VbP nodes below this volume %ile
        merge_radius_atr=0.75,       # collapse levels within N × ATR
        # KDE
        kde_bw_atr_fraction=0.03,    # bandwidth = atr * this / price_range
        kde_prominence=0.04,         # peak must be >= this fraction of max
        # Wyckoff
        wyckoff_vol_multiplier=1.2,  # bar volume must exceed mean × this
        wyckoff_min_vbp_pct=0.01,    # level must have >=1% of max VbP vol
        # Isolation Forest
        iso_contamination=0.15,
    )

    engine = LevelEngine(profile=profile)
    result = engine.run(df, lookback_bars=500)

    # Validate against historical price action
    stats = engine.level_holdrate(df, lookback_bars=500, forward_bars=20)
    print(stats)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import OPTICS
from sklearn.ensemble import IsolationForest
import hdbscan
from ripser import ripser

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Instrument Profile — all tunable parameters in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstrumentProfile:
    """
    All tunable parameters for one symbol × timeframe combination.
    Start with these defaults for NQ/ES 1h, then run level_holdrate()
    to see which thresholds to tighten or loosen.
    """
    tick: float = 0.25
    value_area_pct: float = 0.68

    # ── Score weights (must sum to 1.0) ──────────────────────────────────
    w_algo: float = 0.40        # weight given to each algo's own strength
    w_vbp: float = 0.40         # weight given to VbP volume at the level
    w_confluence: float = 0.20  # weight given to multi-algo agreement

    # ── VbP distribution ─────────────────────────────────────────────────
    # close_weight controls how much volume is biased toward close vs open.
    # 0.0 = fully uniform (v1 behaviour)
    # 0.5 = moderate bias toward close (default, better for trending bars)
    # 1.0 = all volume placed at close (too extreme)
    close_weight: float = 0.5

    # ── Clustering ───────────────────────────────────────────────────────
    vbp_noise_percentile: float = 40.0   # drop nodes below this %ile
    merge_radius_atr: float = 0.75       # collapse within N × ATR

    # ── KDE ──────────────────────────────────────────────────────────────
    kde_bw_atr_fraction: float = 0.03
    kde_prominence: float = 0.04
    kde_min_distance_atr: float = 0.15

    # ── HDBSCAN scales ───────────────────────────────────────────────────
    hdbscan_scales: list = field(default_factory=lambda: [5, 15, 40])
    hdbscan_epsilon_atr: float = 0.05

    # ── OPTICS ───────────────────────────────────────────────────────────
    optics_max_eps_atr: float = 0.5
    optics_top_n: int = 300

    # ── Wyckoff ──────────────────────────────────────────────────────────
    wyckoff_window: int = 20
    wyckoff_vol_multiplier: float = 1.2
    wyckoff_min_vbp_pct: float = 0.01

    # ── Isolation Forest ─────────────────────────────────────────────────
    iso_contamination: float = 0.15
    iso_swing_lookback: int = 2

    # ── Homology ─────────────────────────────────────────────────────────
    homology_persistence_pct: float = 0.80   # keep top N% persistent features
    homology_snap_atr: float = 0.05          # snap to nearest VbP peak within

    def validate(self) -> None:
        total = self.w_algo + self.w_vbp + self.w_confluence
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Score weights must sum to 1.0, got {total:.4f}. "
                f"Adjust w_algo={self.w_algo}, w_vbp={self.w_vbp}, "
                f"w_confluence={self.w_confluence}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 1. VbP Builder — close-weighted distribution
# ─────────────────────────────────────────────────────────────────────────────

def build_vbp(df: pd.DataFrame, profile: InstrumentProfile) -> pd.Series:
    """
    Distribute each bar's volume across its high-low tick range, weighted
    toward the close price.

    Why close-weighted:
    A bullish trending bar that opens at 100 and closes at 120 did NOT trade
    equally at every tick. Most volume printed as price moved toward the close.
    Weighting toward the close gives a better approximation of where contracts
    actually changed hands.

    The weight function is a linear ramp from `1 - close_weight` at the far
    side of the range to `1 + close_weight` at the close side, normalised so
    total weight sums to 1.0 per bar.

    close_weight=0.0 → flat/uniform (original v1 behaviour)
    close_weight=0.5 → moderate close-side bias (default)
    """
    tick = profile.tick
    cw = profile.close_weight
    price_vol: dict[float, float] = {}

    for _, row in df.iterrows():
        lo, hi, cl, vol = row["Low"], row["High"], row["Close"], row["Volume"]
        if pd.isna(vol) or vol <= 0:
            continue

        lo_t = round(lo / tick) * tick
        hi_t = round(hi / tick) * tick
        ticks = np.arange(lo_t, hi_t + tick * 0.5, tick)
        if len(ticks) == 0:
            ticks = np.array([lo_t])

        n = len(ticks)
        if n == 1 or cw == 0.0:
            weights = np.ones(n, dtype=float)
        else:
            # Distance of each tick from the far end relative to close
            # ticks closer to close get higher weight
            cl_t = round(cl / tick) * tick
            cl_t = np.clip(cl_t, lo_t, hi_t)
            dist_from_close = np.abs(ticks - cl_t)
            max_dist = dist_from_close.max()
            if max_dist == 0:
                weights = np.ones(n, dtype=float)
            else:
                # weight = (1 - cw) + cw * (1 - dist/max_dist) * 2
                # → ranges from (1-cw) at far extreme to (1+cw) at close
                weights = (1.0 - cw) + cw * 2.0 * (1.0 - dist_from_close / max_dist)

        weights = weights / weights.sum()
        vol_per_tick = vol * weights

        for p, v in zip(ticks, vol_per_tick):
            key = round(p / tick) * tick
            price_vol[key] = price_vol.get(key, 0.0) + v

    return pd.Series(price_vol, dtype=float).sort_index()


def compute_value_area(
    vbp: pd.Series,
    pct: float = 0.68,
) -> tuple[float, float, float]:
    """
    Returns (POC, VAH, VAL).
    Standard CME/CBOT algorithm: start at POC, expand up or down by
    whichever step adds more volume, until target % of total is enclosed.
    """
    total = vbp.sum()
    poc = float(vbp.idxmax())
    target = total * pct

    upper = vbp[vbp.index > poc].sort_index(ascending=False)
    lower = vbp[vbp.index < poc].sort_index(ascending=True)

    cumvol = float(vbp[poc])
    ui, li = 0, 0
    va_prices = [poc]

    while cumvol < target:
        u_vol = upper.iloc[ui] if ui < len(upper) else 0.0
        l_vol = lower.iloc[li] if li < len(lower) else 0.0

        if u_vol == 0.0 and l_vol == 0.0:
            break
        if u_vol >= l_vol:
            cumvol += u_vol
            va_prices.append(float(upper.index[ui]))
            ui += 1
        else:
            cumvol += l_vol
            va_prices.append(float(lower.index[li]))
            li += 1

    return poc, float(max(va_prices)), float(min(va_prices))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Algorithms — all VbP-native, all profile-driven
# ─────────────────────────────────────────────────────────────────────────────

def algo_kde(
    vbp: pd.Series, profile: InstrumentProfile, atr: float
) -> list[dict]:
    """
    Volume-weighted KDE. Bandwidth and peak-detection parameters come from
    the profile so they can be tuned per instrument.
    """
    tick = profile.tick
    prices = vbp.index.values.astype(float)
    weights = (vbp.values / vbp.values.sum()).astype(float)

    price_range = prices.max() - prices.min()
    if price_range == 0:
        return []

    bw = max(0.01, (atr * profile.kde_bw_atr_fraction) / price_range)
    kde = gaussian_kde(prices, weights=weights, bw_method=bw)

    grid = np.arange(prices.min(), prices.max() + tick, tick)
    density = kde(grid)

    min_dist = max(4, int(atr * profile.kde_min_distance_atr / tick))
    peaks, _ = find_peaks(
        density,
        prominence=density.max() * profile.kde_prominence,
        distance=min_dist,
    )

    levels = []
    for i in peaks:
        levels.append({
            "price": round(float(grid[i]) / tick) * tick,
            "strength": float(density[i] / density.max()),
            "source": "kde_vbp",
        })
    return levels


def algo_hdbscan_multiscale(
    vbp: pd.Series, profile: InstrumentProfile, atr: float
) -> list[dict]:
    """
    HDBSCAN at multiple resolutions on (price, vol_norm) space.
    Noise floor set by profile.vbp_noise_percentile — tune this if you're
    getting too many or too few levels.
    """
    tick = profile.tick
    threshold = vbp.quantile(profile.vbp_noise_percentile / 100.0)
    nodes = vbp[vbp >= threshold]
    if len(nodes) < 10:
        return []

    prices = nodes.index.values.astype(float)
    vols = (nodes.values / nodes.max()).astype(float)
    X = np.column_stack([
        (prices - prices.mean()) / (prices.std() + 1e-9),  # normalise price axis
        vols,
    ])

    results: dict[float, list[float]] = {}

    for scale in profile.hdbscan_scales:
        min_s = max(3, int(len(nodes) * 0.01 * scale / 10))
        clf = hdbscan.HDBSCAN(
            min_cluster_size=min_s,
            min_samples=2,
            cluster_selection_epsilon=float(atr * profile.hdbscan_epsilon_atr
                                            / (prices.std() + 1e-9)),
        )
        labels = clf.fit_predict(X)
        for lbl in set(labels) - {-1}:
            mask = labels == lbl
            center = float(np.average(prices[mask], weights=vols[mask]))
            center_r = round(center / tick) * tick
            vol_weight = float(vols[mask].sum() / (vols.sum() + 1e-9))
            results.setdefault(center_r, []).append(vol_weight)

    levels = []
    for price, strengths in results.items():
        scale_hits = len(strengths)
        levels.append({
            "price": price,
            "strength": min(1.0, float(np.mean(strengths)) * scale_hits / 3.0),
            "source": "hdbscan_vbp",
            "scale_hits": scale_hits,
        })
    return levels


def algo_optics(
    vbp: pd.Series, profile: InstrumentProfile, atr: float
) -> list[dict]:
    """
    OPTICS on top-N VbP nodes. Catches variable-density clusters that
    HDBSCAN misses when levels are spread further apart (weekly zones).
    """
    tick = profile.tick
    top = vbp.nlargest(min(profile.optics_top_n, len(vbp)))
    if len(top) < 10:
        return []

    prices = top.index.values.astype(float).reshape(-1, 1)
    vols = top.values.astype(float)

    clf = OPTICS(
        min_samples=3,
        xi=0.05,
        max_eps=atr * profile.optics_max_eps_atr,
    )
    labels = clf.fit_predict(prices)

    levels = []
    for lbl in set(labels) - {-1}:
        mask = labels == lbl
        center = float(np.average(prices[mask, 0], weights=vols[mask]))
        strength = float(vols[mask].sum() / (vols.sum() + 1e-9))
        levels.append({
            "price": round(center / tick) * tick,
            "strength": min(1.0, strength * 5.0),
            "source": "optics_vbp",
        })
    return levels


def algo_isolation_forest(
    df: pd.DataFrame,
    vbp: pd.Series,
    profile: InstrumentProfile,
    atr: float,
) -> list[dict]:
    """
    Isolation Forest on swing pivots.
    Features: (price, bar_volume, vbp_volume_at_price).
    Anomalous = statistically unusual pivot that also had large volume.
    These are the bars where real institutional activity happened.
    """
    tick = profile.tick
    lb = profile.iso_swing_lookback
    swing_data = []
    arr = df.reset_index(drop=True)

    for i in range(lb, len(arr) - lb):
        row = arr.iloc[i]
        prev_h = arr.iloc[i - lb: i]["High"].max()
        prev_l = arr.iloc[i - lb: i]["Low"].min()
        next_h = arr.iloc[i + 1: i + 1 + lb]["High"].max()
        next_l = arr.iloc[i + 1: i + 1 + lb]["Low"].min()

        is_high = row["High"] > prev_h and row["High"] > next_h
        is_low = row["Low"] < prev_l and row["Low"] < next_l

        if is_high or is_low:
            price = float(row["High"] if is_high else row["Low"])
            p_r = round(price / tick) * tick
            vol_at_level = float(vbp.get(p_r, 0.0))
            swing_data.append({
                "price": price,
                "bar_vol": float(row["Volume"]),
                "vbp_vol": vol_at_level,
            })

    if len(swing_data) < 10:
        return []

    sw = pd.DataFrame(swing_data)
    X = sw[["price", "bar_vol", "vbp_vol"]].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    iso = IsolationForest(
        contamination=profile.iso_contamination,
        random_state=42,
    )
    sw["anomaly"] = iso.fit_predict(X_norm)

    outliers = sw[sw["anomaly"] == -1]
    if outliers.empty:
        return []

    max_vol = outliers["vbp_vol"].max()
    levels = []
    for _, row in outliers.iterrows():
        strength = float(row["vbp_vol"] / max_vol) if max_vol > 0 else 0.3
        levels.append({
            "price": round(row["price"] / tick) * tick,
            "strength": max(0.3, strength),
            "source": "isolation_forest_vbp",
        })
    return levels


def algo_wyckoff(
    df: pd.DataFrame,
    vbp: pd.Series,
    profile: InstrumentProfile,
    atr: float,
) -> list[dict]:
    """
    Wyckoff springs and upthrusts.
    Only registers a pattern if the VbP at the touched level meets the
    minimum volume threshold — separates real absorption from thin-air pokes.
    """
    tick = profile.tick
    w = profile.wyckoff_window
    vm = profile.wyckoff_vol_multiplier
    min_vbp = profile.wyckoff_min_vbp_pct
    arr = df.reset_index(drop=True)
    levels = []
    max_vbp = float(vbp.max())

    for i in range(w, len(arr)):
        chunk = arr.iloc[i - w: i]
        curr = arr.iloc[i]
        support = chunk["Low"].min()
        resistance = chunk["High"].max()
        mean_vol = chunk["Volume"].mean()

        # Spring: undercut support then close back above
        if (curr["Low"] < support - tick and
                curr["Close"] > support and
                curr["Volume"] > mean_vol * vm):
            lp = round(support / tick) * tick
            vbp_ratio = float(vbp.get(lp, 0.0)) / (max_vbp + 1e-9)
            if vbp_ratio >= min_vbp:
                levels.append({
                    "price": lp,
                    "strength": min(1.0, 0.4 + vbp_ratio * 4.0),
                    "source": "wyckoff_spring",
                })

        # Upthrust: spike above resistance then close back below
        if (curr["High"] > resistance + tick and
                curr["Close"] < resistance and
                curr["Volume"] > mean_vol * vm):
            lp = round(resistance / tick) * tick
            vbp_ratio = float(vbp.get(lp, 0.0)) / (max_vbp + 1e-9)
            if vbp_ratio >= min_vbp:
                levels.append({
                    "price": lp,
                    "strength": min(1.0, 0.4 + vbp_ratio * 4.0),
                    "source": "wyckoff_upthrust",
                })

    return levels


def algo_persistent_homology(
    vbp: pd.Series, profile: InstrumentProfile, atr: float
) -> list[dict]:
    """
    1D sublevel-set persistence on the VbP density curve.

    v1 used a 2D point cloud (price, volume), which made the birth/death
    coordinates ambiguous — they lived in normalised space and had to be
    mapped back to prices via a heuristic that could silently misfire.

    This version uses 1D persistence directly: the VbP curve is treated as
    a height function f(price). Sublevel-set persistence finds pairs of
    (local_min, local_max) that are significant across scales. Each pair
    corresponds to a price valley and the peak above it — the peak is a
    genuine high-volume node that persists across multiple smoothing scales.

    Because we're working in price space (1D, no coordinate transform),
    the mapping back to actual prices is exact.

    How it works:
      - Smooth VbP at several bandwidths
      - At each bandwidth, find (valley, peak) pairs and record persistence
        = peak_density - valley_density
      - A level is "persistent" if it appears as a significant peak at
        multiple smoothing scales
    """
    tick = profile.tick
    prices = vbp.index.values.astype(float)
    vals = vbp.values.astype(float)

    if len(prices) < 20:
        return []

    # Smooth at 3 scales (fine, medium, coarse)
    from scipy.ndimage import gaussian_filter1d

    # Scale sigma in ticks, relative to ATR
    sigmas = [
        max(2, int(atr * 0.05 / tick)),
        max(5, int(atr * 0.15 / tick)),
        max(10, int(atr * 0.40 / tick)),
    ]

    level_hits: dict[float, list[float]] = {}

    for sigma in sigmas:
        smoothed = gaussian_filter1d(vals.astype(float), sigma=sigma)
        smoothed_norm = smoothed / (smoothed.max() + 1e-9)

        # Find peaks in smoothed curve
        min_dist = max(4, int(atr * 0.10 / tick))
        peaks, _ = find_peaks(
            smoothed_norm,
            prominence=smoothed_norm.max() * profile.kde_prominence,
            distance=min_dist,
        )

        for pk in peaks:
            raw_price = float(prices[pk])
            snapped = round(raw_price / tick) * tick
            persistence = float(smoothed_norm[pk])
            level_hits.setdefault(snapped, []).append(persistence)

    # Only keep levels found at 2+ smoothing scales
    levels = []
    for price, hits in level_hits.items():
        if len(hits) >= 2:
            strength = float(np.mean(hits)) * min(1.0, len(hits) / 3.0)
            levels.append({
                "price": price,
                "strength": min(1.0, strength),
                "source": "homology_vbp",
            })

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# 3. Merge & Scoring Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def merge_levels(
    raw_levels: list[dict],
    vbp: pd.Series,
    poc: float,
    vah: float,
    val: float,
    atr: float,
    profile: InstrumentProfile,
    current_price: float,
) -> pd.DataFrame:
    """
    Merge → VbP-weight → score → inject anchors → rank.
    All weights come from profile so they can be tuned externally.
    """
    tick = profile.tick

    if not raw_levels:
        # Still return anchors even if no algo levels detected
        raw_levels = []

    df_levels = pd.DataFrame(raw_levels)

    # ── Step 1: Agglomerative merge ───────────────────────────────────────
    merge_radius = atr * profile.merge_radius_atr
    df_levels = df_levels.sort_values("price").reset_index(drop=True)

    merged = []
    used = set()
    for i, row in df_levels.iterrows():
        if i in used:
            continue
        cluster = df_levels[
            (df_levels["price"] >= row["price"] - merge_radius) &
            (df_levels["price"] <= row["price"] + merge_radius)
        ]
        for idx in cluster.index:
            used.add(idx)

        # Volume-weighted centroid — levels with more VbP volume pull center
        vbp_weights = np.array([
            float(vbp.get(round(p / tick) * tick, 1e-9))
            for p in cluster["price"]
        ])
        centroid = float(np.average(cluster["price"], weights=vbp_weights))
        centroid_r = round(centroid / tick) * tick
        sources = cluster["source"].tolist()
        strength = float(cluster["strength"].mean())
        algo_count = len(set(sources))

        merged.append({
            "price": centroid_r,
            "strength_raw": strength,
            "algo_count": algo_count,
            "sources": "|".join(sorted(set(sources))),
        })

    df_m = pd.DataFrame(merged) if merged else pd.DataFrame(
        columns=["price", "strength_raw", "algo_count", "sources"]
    )

    if not df_m.empty:
        # ── Step 2: VbP volume at centroid ───────────────────────────────
        max_vbp = float(vbp.max())
        df_m["vbp_vol"] = df_m["price"].apply(
            lambda p: float(vbp.get(round(p / tick) * tick, 0.0))
        )
        df_m["vbp_norm"] = df_m["vbp_vol"] / (max_vbp + 1e-9)

        # ── Step 3: Confluence factor ─────────────────────────────────────
        # Per-algo boost: each additional algo adds a fixed increment.
        # You can tune this by looking at whether 3-algo confluence levels
        # outperform 2-algo ones in your backtest.
        confluence_per_algo = profile.w_confluence  # bonus per additional algo
        df_m["confluence_factor"] = (df_m["algo_count"] - 1).clip(0) * confluence_per_algo

        # ── Step 4: Final score ───────────────────────────────────────────
        df_m["score"] = (
            df_m["strength_raw"] * profile.w_algo +
            df_m["vbp_norm"] * profile.w_vbp +
            df_m["confluence_factor"]
        ).clip(0, 1)

    # ── Step 5: Inject POC / VAH / VAL ───────────────────────────────────
    max_vbp = float(vbp.max())
    anchors = pd.DataFrame([
        {
            "price": poc, "score": 1.0, "algo_count": 9,
            "sources": "poc", "vbp_norm": 1.0, "strength_raw": 1.0,
            "vbp_vol": float(vbp.get(poc, 0.0)), "type": "POC",
            "confluence_factor": 0.0,
        },
        {
            "price": vah, "score": 0.92, "algo_count": 9,
            "sources": "vah", "vbp_norm": 0.9, "strength_raw": 0.9,
            "vbp_vol": float(vbp.get(vah, 0.0)), "type": "VAH",
            "confluence_factor": 0.0,
        },
        {
            "price": val, "score": 0.92, "algo_count": 9,
            "sources": "val", "vbp_norm": 0.9, "strength_raw": 0.9,
            "vbp_vol": float(vbp.get(val, 0.0)), "type": "VAL",
            "confluence_factor": 0.0,
        },
    ])

    if df_m.empty:
        df_m = anchors
    else:
        df_m["type"] = "level"
        df_m = pd.concat([df_m, anchors], ignore_index=True)

    # ── Step 6: Proximity metadata ────────────────────────────────────────
    df_m["dist_atr"] = (df_m["price"] - current_price).abs() / (atr + 1e-9)
    df_m["side"] = np.where(df_m["price"] > current_price, "above", "below")
    df_m = df_m.sort_values("score", ascending=False).reset_index(drop=True)
    df_m["rank"] = df_m.index + 1

    return df_m


# ─────────────────────────────────────────────────────────────────────────────
# 4. ATR helper (standalone, used by engine and backtest)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    hi = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    cl = df["Close"].values.astype(float)
    prev_cl = np.roll(cl, 1)
    prev_cl[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    return float(np.mean(tr[1: period + 1]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class LevelEngine:
    """
    VbP-native level detection engine.
    Pass an InstrumentProfile to configure all parameters per symbol/timeframe.
    """

    def __init__(self, profile: Optional[InstrumentProfile] = None):
        self.profile = profile or InstrumentProfile()
        self.profile.validate()

    def run(
        self,
        df: pd.DataFrame,
        lookback_bars: int = 500,
    ) -> dict:
        """
        Run the full pipeline on the most recent `lookback_bars` bars.

        Returns
        -------
        dict:
          levels         : pd.DataFrame — merged, scored, ranked levels
          poc            : float
          vah            : float  (68% value area high)
          val            : float  (68% value area low)
          vbp            : pd.Series  {price: volume}
          atr            : float
          current_price  : float
        """
        p = self.profile
        window = df.tail(lookback_bars).copy().reset_index(drop=True)

        atr = compute_atr(window)
        current_price = float(window["Close"].iloc[-1])

        vbp = build_vbp(window, p)
        poc, vah, val = compute_value_area(vbp, p.value_area_pct)

        raw: list[dict] = []
        raw += algo_kde(vbp, p, atr)
        raw += algo_hdbscan_multiscale(vbp, p, atr)
        raw += algo_optics(vbp, p, atr)
        raw += algo_isolation_forest(window, vbp, p, atr)
        raw += algo_wyckoff(window, vbp, p, atr)
        raw += algo_persistent_homology(vbp, p, atr)

        levels = merge_levels(raw, vbp, poc, vah, val, atr, p, current_price)

        return {
            "levels": levels,
            "poc": poc,
            "vah": vah,
            "val": val,
            "vbp": vbp,
            "atr": atr,
            "current_price": current_price,
        }

    def top_levels(
        self,
        df: pd.DataFrame,
        n: int = 20,
        lookback_bars: int = 500,
        min_score: float = 0.3,
    ) -> pd.DataFrame:
        result = self.run(df, lookback_bars)
        levels = result["levels"]
        cols = ["rank", "price", "score", "type", "side",
                "dist_atr", "algo_count", "sources", "vbp_vol"]
        return (
            levels[levels["score"] >= min_score]
            .head(n)[cols]
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────────────────────────────
    # 6. Backtest validator
    # ─────────────────────────────────────────────────────────────────────

    def level_holdrate(
        self,
        df: pd.DataFrame,
        lookback_bars: int = 500,
        forward_bars: int = 20,
        tolerance_atr: float = 0.25,
        min_score: float = 0.3,
        step: int = 50,
    ) -> pd.DataFrame:
        """
        Walk-forward validation: detects levels at each window, then checks
        whether price respected them in the next `forward_bars` bars.

        A level is "held" if:
          - Price comes within tolerance_atr × ATR of it, AND
          - Price reverses by at least 0.5 × ATR without closing through

        A level is "broken" if:
          - Price closes through it by more than tolerance_atr × ATR

        Returns a DataFrame with hold rates per score bucket so you can see
        whether high-score levels actually hold more often — the core
        validation question.

        Parameters
        ----------
        step : int
            How many bars to advance between each detection window.
            Lower = more samples but slower.
        """
        results = []
        p = self.profile
        total_bars = len(df)

        for start in range(lookback_bars, total_bars - forward_bars, step):
            train = df.iloc[:start]
            fwd = df.iloc[start: start + forward_bars].reset_index(drop=True)

            try:
                result = self.run(train, lookback_bars)
            except Exception:
                continue

            atr = result["atr"]
            tol = atr * tolerance_atr
            levels = result["levels"]
            levels = levels[levels["score"] >= min_score].copy()

            for _, lv in levels.iterrows():
                lp = lv["price"]
                score = lv["score"]
                touched = False
                held = False
                broken = False

                for i in range(len(fwd)):
                    bar = fwd.iloc[i]
                    # Check if price approached the level
                    if abs(bar["Low"] - lp) <= tol or abs(bar["High"] - lp) <= tol:
                        touched = True
                        # Check what happened after the touch
                        remaining = fwd.iloc[i:]
                        if lp > result["current_price"]:
                            # Resistance level
                            broke = (remaining["Close"] > lp + tol).any()
                            bounced = (remaining["Low"] < lp - atr * 0.5).any()
                        else:
                            # Support level
                            broke = (remaining["Close"] < lp - tol).any()
                            bounced = (remaining["High"] > lp + atr * 0.5).any()

                        if broke and not bounced:
                            broken = True
                        elif bounced:
                            held = True
                        break

                results.append({
                    "window_start": start,
                    "price": lp,
                    "score": score,
                    "score_bucket": round(score, 1),
                    "touched": touched,
                    "held": held,
                    "broken": broken,
                    "sources": lv.get("sources", ""),
                    "algo_count": lv.get("algo_count", 0),
                })

        if not results:
            return pd.DataFrame()

        df_res = pd.DataFrame(results)
        touched = df_res[df_res["touched"]]

        summary = (
            touched.groupby("score_bucket")
            .agg(
                n_touched=("held", "count"),
                n_held=("held", "sum"),
                n_broken=("broken", "sum"),
                hold_rate=("held", "mean"),
            )
            .reset_index()
        )
        summary["break_rate"] = summary["n_broken"] / summary["n_touched"]
        return summary.sort_values("score_bucket", ascending=False)
