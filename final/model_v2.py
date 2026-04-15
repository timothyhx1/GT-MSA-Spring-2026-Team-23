"""Dynamic DCA weight computation using 200-day MA strategy.

This module computes daily investment weights for a Bitcoin DCA strategy
based on a simple 200-day moving average signal:
- Buy more when price is below the 200-day MA
- Buy less when price is above the 200-day MA
"""

import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.parent
# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"

# Strategy parameters
MIN_W = 1e-6
MA_WINDOW = 200  # 200-day simple moving average
DYNAMIC_STRENGTH = 2.0  # Multiplier for weight adjustments
W = [1,0,0]
B = [0.5,0.5]
D = [0.5,0.5]
# Feature column names (for compatibility)
FEATS = [
    "price_vs_ma",
]

base_params = {
        # Within-group weights
        "w_val_mvrv": 0.5,
        "w_val_pma": 0.35,
        "w_val_rsi": 0.15,
        "W_val": 3,
        "val_range": 20,

        "w_grad1": 0.5,
        "w_grad2": 0.2,
        "w_grad_min": 0.8,
        "w_grad_max": 1.2,

        "w_accel1": 0.3,
        "w_accel2": 0.2,
        "w_accel_min": 0.9,
        "w_accel_max": 1.1,

        "w_ml_prob70": 0.3,
        "w_ml_min_gap": 0.2,

        "w_risk_price_vol": 0.5,
        "w_risk_mvrv_vol": 0.5,

        # Top-level group weights

        "W_risk": 0.2,

        # Multiplier controls
        # "base_mult": trial.suggest_float("base_mult", 0.80, 1.2),
        # "scale": trial.suggest_float("scale", 0.05, 1.2),
        # "min_mult": trial.suggest_float("min_mult", 0.01, 0.5),
        # "max_mult": trial.suggest_float("max_mult", 1.05, 10.0),
        #
        # # Smoothness
        # "mult_ema_span": trial.suggest_int("mult_ema_span", 1, 14),
    }

best_params = {'w_val_mvrv': 0.7580741558532044,
     'w_val_pma': 0.4522114540755125,
     'w_val_rsi': 0.009116793007697986,
     'W_val': 4.9021983896396515,
     'val_range': 20.0,
     'w_grad1': 0.38808897437757395,
     'w_grad2': 0.3795357058207186,
     'w_grad_min': 0.7281129589206239,
     'w_grad_max': 1.4670224770322686,
     'w_accel1': 0.007218875397044383,
     'w_accel2': 0.010296215521067471,
     'w_accel_min': 0.7011512885465969,
     'w_accel_max': 1.3036004444360998,
     'w_ml_prob70': 0.32132810909763554,
     'w_ml_min_gap': 0.03555647237071294,
     'w_risk_price_vol': 0.635760429352899,
     'w_risk_mvrv_vol': 0.9239592693631066,
     'W_risk': 0.05559167039286142}

# Choose here to use best guess or optimized parameters
# params = base_params.copy()
params = best_params.copy()
# =============================================================================
# Helper Functions
# =============================================================================


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 200-day MA feature for weight calculation.

    Features (all lagged 1 day to prevent look-ahead bias):
    - price_vs_ma: Normalized distance from 200-day MA, clipped to [-1, 1]

    Args:
        df: DataFrame with price column

    Returns:
        DataFrame with price and computed features
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 200-day MA and distance
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1)


    btc_df = pd.read_parquet(base_dir/"eda/btc_df_final.parquet")
    min_ratio_log_pred = pd.read_parquet(base_dir/"eda/min_gap.parquet")['min_ratio_log_pred']
    prob_top30_zscore = pd.read_parquet(base_dir/"eda/prob_top30.parquet")['prob_top30_zscore']

    FEATURES = [
        'oc_MVRVCur_Z',
        'oc_MVRVCur_grad',
        'oc_MVRVCur_accel',
        'RSI_zscore',
        'oc_MVRVCur_vol_Z',
        'oc_vol_30d_Z'
    ]

    btc_df.set_index(btc_df['time_basis'], inplace=True)
    btc_df.index.name = 'time'
    metrics = btc_df[FEATURES].copy()

    # Build and lag features
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma.shift(1).fillna(0),  # Lag 1 day
        },
        index=price.index,
    )
    features = features.join(metrics, on="time")
    features = features.join(prob_top30_zscore, on = "time")
    features = features.join(min_ratio_log_pred, on="time")
    # print(features.tail(20))

    return features

def _safe_zscore(series: pd.Series, clip: float = 5.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)

    if pd.isna(sigma) or sigma < 1e-12:
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mu) / sigma

    return z.clip(-clip, clip).fillna(0.0)

# =============================================================================
# Weight Allocation
# =============================================================================


def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights using cumulative mean normalization.

    signal[i] = raw[i] / mean(raw[0:i+1])

    This ensures weights only depend on past data.
    """
    n = len(raw)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean
    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate weights with lock-on-compute stability.

    Past weights are locked and never change. Future days absorb remainder.

    Args:
        raw: Raw weight values for all dates
        n_past: Number of past/current dates (locked)
        locked_weights: Optional pre-computed locked weights from database

    Returns:
        Weights summing to 1.0
    """
    n = len(raw)
    if n == 0:
        return np.array([])
    if n_past <= 0:
        return np.full(n, 1.0 / n)

    n_past = min(n_past, n)
    w = np.zeros(n)
    base_weight = 1.0 / n

    # Compute or use locked weights for past days
    if locked_weights is not None and len(locked_weights) >= n_past:
        w[:n_past] = locked_weights[:n_past]
    else:
        for i in range(n_past):
            signal = _compute_stable_signal(raw[: i + 1])[-1]
            w[i] = signal * base_weight

    # Scale past weights if they exceed budget
    past_sum = w[:n_past].sum()
    target_budget = n_past / n
    if past_sum > target_budget + 1e-10:
        w[:n_past] *= target_budget / past_sum

    # Future days (except last): uniform
    n_future = n - n_past
    if n_future > 1:
        w[n_past : n - 1] = base_weight

    # Last day absorbs remainder
    w[n - 1] = max(1.0 - w[: n - 1].sum(), 0)

    return w


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_value_multiplier(oc_MVRVCur_Z, price_vs_ma, RSI_zscore) -> np.ndarray:
    """Compute weight multiplier from 200-day MA signal.

    Simple strategy: buy more when price is below MA, less when above.

    Args:
        price_vs_ma: Distance from 200-day MA in [-1, 1]
            Negative values = below MA (buy more)
            Positive values = above MA (buy less)

    Returns:
        Multipliers centered around 1.0
    """
    # Signal: negative price_vs_ma = below MA = buy more
    w_sum = params["w_val_mvrv"] + params["w_val_pma"] + params["w_val_rsi"]
    w_val_mvrv = params["w_val_mvrv"]/w_sum
    w_val_pma = params["w_val_pma"]/w_sum
    w_val_rsi = params["w_val_rsi"]/w_sum

    signal = -(
            w_val_mvrv * oc_MVRVCur_Z
            + w_val_pma * price_vs_ma
            + w_val_rsi * RSI_zscore
        ) * params["W_val"]

    # Scale and clip

    adjustment = np.clip(signal, -params["val_range"], params["val_range"])

    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)

def trend_modifier(oc_MVRVCur_Z, oc_MVRVCur_grad):
    threshold = np.where(
        oc_MVRVCur_Z < -1, 0.1,
        np.where(oc_MVRVCur_Z > 1.5, 0.4, 0.2)
    )
    modifier = np.where(
        oc_MVRVCur_grad > threshold, 1.0 + params['w_grad1'] * np.minimum(oc_MVRVCur_grad, 1.0),
        np.where(oc_MVRVCur_grad < -threshold, 1.0 + params['w_grad2'] * np.maximum(oc_MVRVCur_grad, -1.0), 1.0)
    )

    return np.clip(modifier, params['w_grad_min'], params['w_grad_max'])

def accel_modifier(oc_MVRVCur_grad, oc_MVRVCur_accel):
    same_direction = (oc_MVRVCur_grad * oc_MVRVCur_accel) > 0

    modifier = np.where(
        same_direction,
        1.0 + params['w_accel1'] * np.abs(oc_MVRVCur_accel),  # Amplify if momentum building
        1.0 - params['w_accel2'] * np.abs(oc_MVRVCur_accel)  # Dampen if potential reversal
    )
    return np.clip(modifier, params['w_accel_min'], params['w_accel_max'])

def compute_risk_score(oc_vol_30d_zscore, oc_MVRVCur_vol_Z):
    w_sum = params["w_risk_price_vol"] + params["w_risk_mvrv_vol"]
    w_risk_price_vol = params["w_risk_price_vol"] / w_sum
    w_risk_mvrv_vol = params["w_risk_mvrv_vol"] / w_sum
    signal = -(w_risk_price_vol * oc_vol_30d_zscore + w_risk_mvrv_vol * oc_MVRVCur_vol_Z)
    multiplier = (np.clip(signal, -3, -1) + 1) * params["W_risk"] / 2 + 1
    return multiplier

# =============================================================================
# Weight Computation API
# =============================================================================


def _clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.where(np.isfinite(arr), arr, 0)


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Extract and clean features

    # Valuation features
    oc_MVRVCur_Z = pd.Series(_clean_array(df["oc_MVRVCur_Z"].to_numpy()), index=df.index)
    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    RSI_zscore = _safe_zscore(df["RSI_zscore"])

    # Timing features
    oc_MVRVCur_grad = _clean_array(df["oc_MVRVCur_grad"].values)
    oc_MVRVCur_accel = _clean_array(df["oc_MVRVCur_accel"].values)

    # ML features
    prob_top30_zscore = _clean_array(df["prob_top30_zscore"].values)
    min_ratio_log_pred = _clean_array(df["min_ratio_log_pred"].values)

    # Risk features
    oc_MVRVCur_vol_Z = _clean_array(df["oc_MVRVCur_vol_Z"].values)
    oc_vol_30d_zscore = _clean_array(df["oc_vol_30d_Z"].values)

    # Compute dynamic weights
    valuation_score = compute_value_multiplier(oc_MVRVCur_Z, price_vs_ma, RSI_zscore)
    # valuation_adj = 1

    grad_score = trend_modifier(oc_MVRVCur_Z, oc_MVRVCur_grad)
    accel_score = accel_modifier(oc_MVRVCur_grad, oc_MVRVCur_accel)

    # Boost purchase to 1-1.3 if show high confidence
    ml1_score = (
            (np.clip(prob_top30_zscore, 0.5, 2) - 0.5) * params["w_ml_prob70"] / 1.5 + 1
    )


    ml2_score = (
            (np.clip(min_ratio_log_pred, -0.4, -0.1) + 0.1) * params["w_ml_min_gap"] / 0.3 + 1
    )


    risk_score = compute_risk_score(oc_vol_30d_zscore, oc_MVRVCur_vol_Z)

    # valuation_score = 1
    grad_score = 1
    accel_score = 1
    ml1_score = 1
    ml2_score = 1
    risk_score = 1

    raw = base * valuation_score * grad_score * accel_score * ml1_score * ml2_score * risk_score

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability.

    Two modes:
    1. BACKTEST (locked_weights=None): Signal-based allocation
    2. PRODUCTION (locked_weights provided): DB-backed stability

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Investment window start
        end_date: Investment window end
        current_date: Current date (past/future boundary)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights summing to 1.0
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features for future dates
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    # Determine past/future split
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)
