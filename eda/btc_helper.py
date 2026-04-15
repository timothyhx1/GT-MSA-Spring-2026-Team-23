"""
Exploratory Data Analysis (EDA) Helper Functions
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import psutil
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from fredapi import Fred

import time
import requests
import pandas as pd
import ta

MVRV_GRADIENT_WINDOW = 30
MVRV_ACCEL_WINDOW = 14

def zscore(series: pl.Expr, window: int) -> pl.Series:
    """Compute rolling z-score."""
    mean = series.rolling_mean(window_size=window, min_periods=window // 2)
    std = series.rolling_std(window_size=window, min_periods=window // 2)
    return (
        pl.when(std > 1e-8)
        .then((series - mean) / std)
        .otherwise(None)
    )

class OnChainGenerator:
    def __init__(self, df):
        self.df = df

    # oc_MVRVCur_Z: (CapMVRVCur - rolling_mean) / rolling_std
    def oc_MVRVCur_Z(self, x=365):

        z = zscore(pl.col('CapMVRVCur'), x)

        # Smoothed gradient using EMA
        gradient_raw = z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm_mean(
            span=MVRV_GRADIENT_WINDOW, adjust=False
        )

        mvrv_gradient = (gradient_smooth * 2).tanh()

        # MVRV acceleration (second derivative - momentum detection)
        mvrv_acceleration = (
                mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
                .ewm_mean(span=MVRV_ACCEL_WINDOW, adjust=False)
                * 3
        ).tanh()

        log_chg = (pl.col("CapMVRVCur") / pl.col("CapMVRVCur").shift(1)).log()

        vol30 = log_chg.rolling_std(
            window_size=30,
            min_periods=30
        )

        vol30z = zscore(vol30, x)

        self.df = self.df.with_columns([
            z.clip(-4, 4).alias('oc_MVRVCur_Z'),
            gradient_smooth.alias('oc_MVRVCur_grad'),
            mvrv_acceleration.alias('oc_MVRVCur_accel'),
            vol30z.alias('oc_MVRVCur_vol_Z')
        ])

        return self.df
    #
    def oc_CapMrktCurUSD_Z(self, x=365):
        self.df = self.df.with_columns(
            zscore(pl.col('CapMrktCurUSD'), x).alias('oc_CapMrktCurUSD_Z')
        )

        self.df = self.df.with_columns(
            pl.when(pl.col('oc_CapMrktCurUSD_Z')>2).then(1)
            .otherwise(0)
            .alias('oc_MVRV_high')
        )

        return self.df

    # oc_Puell_proxy
    def oc_Puell_proxy(self, x=365):
        iss_daily = pl.col("IssTotNtv") - pl.col("IssTotNtv").shift(1)
        rev = (iss_daily + pl.col("FeeTotNtv")) * pl.col("PriceUSD")
        mean = rev.rolling_mean(window_size=x, min_periods=x)

        puell = pl.when(mean > 1e-8) \
            .then(rev / mean) \
            .otherwise(None)

        self.df = self.df.with_columns(
            puell.alias("oc_Puell_proxy")
        )

        return self.df

    # oc_miner_stress: RevHashRateUSD/rolling_mean(RevHashRateUSD) _365d
    def oc_miner_stress(self, x=365):
        mean = pl.col('HashRate').rolling_mean(window_size=x, min_periods=x)

        stress = pl.when(mean > 1e-8) \
            .then(pl.col('HashRate') / mean) \
            .otherwise(None)

        self.df = self.df.with_columns(
            stress
            .alias('oc_miner_stress')
        )

        return self.df

    # oc_netexchangenorm: FlowInExUSD - FlowOutExUSD
    def oc_netexchangenorm(self):
        self.df = self.df.with_columns(
            ((pl.col('FlowInExUSD') - pl.col('FlowOutExUSD'))/pl.col('CapMrktCurUSD'))
            .alias('oc_netexchangenorm')
        )

        return self.df

    def oc_flowimbalance(self):
        self.df = self.df.with_columns(
            ((pl.col('FlowInExUSD') - pl.col('FlowOutExUSD'))/(pl.col('FlowInExUSD') + pl.col('FlowOutExUSD')))
            .alias('oc_flowimbalance')
        )

        return self.df

    # oc_flow_ratio: FlowInExUSD / FlowOutExUSD
    def oc_flow_ratio(self):
        out = pl.col('FlowOutExUSD')

        ratio = pl.when(out > 1e-8) \
            .then(pl.col('FlowInExUSD') / out) \
            .otherwise(None)

        self.df = self.df.with_columns(
            ratio.alias('oc_flow_ratio')
        )

        return self.df

    def oc_vol_30d(self, window=30):
        log_ret = (pl.col("PriceUSD") / pl.col("PriceUSD").shift(1)).log()

        vol = log_ret.rolling_std(
            window_size=window,
            min_periods=window
        )

        volz = zscore(vol, 365)

        self.df = self.df.with_columns(
            vol.alias("oc_vol_30d"),
            volz.alias('oc_vol_30d_Z')
        )

        return self.df

    def oc_turnover(self):
        self.df = self.df.with_columns(
            (pl.col('volume_reported_spot_usd_1d')/pl.col('CapMrktCurUSD')).alias('oc_turnover')
        )

class TechnicalGenerator:
    def __init__(self, df):
        self.df = df

    # Inspired from https://medium.com/@mjbryan8/creating-technical-indicators-for-trading-using-polars-and-python-b3b72370e7b1

    def sma_bb(self, x=200):
        # Simple Moving Average, rolling stdev, and Bollinger Bands
        sma = pl.col("PriceUSD").rolling_mean(window_size=x, min_periods=x)
        smastd = pl.col("PriceUSD").rolling_std(window_size=x, min_periods=x)
        self.df = self.df.with_columns([
            sma.alias('sma_'+str(x)),
            smastd.alias('smastd_'+str(x)),
            # (sma + 2 * smastd).alias('sma_bb_upper_'+str(x)),
            # (sma - 2 * smastd).alias('sma_bb_lower_'+str(x)),
            (pl.col('PriceUSD')/sma-1).alias('distance_sma_'+str(x))
        ])

        return self.df

    def ema_bb(self, x=200):
        # Exponential Moving Average, rolling stdev, and Bollinger Bands
        ema = pl.col("PriceUSD").ewm_mean(span=x, adjust=False)
        emastd = pl.col("PriceUSD").ewm_std(span=x)
        self.df = self.df.with_columns([
            ema.alias('ema_'+str(x)),
            emastd.alias('emastd_'+str(x)),
            # (ema + 2 * emastd).alias('ema_bb_upper_'+str(x)),
            # (ema - 2 * emastd).alias('ema_bb_lower_'+str(x)),
            (pl.col('PriceUSD') / ema-1).alias('distance_ema_' + str(x))
        ])

        return self.df

    def drawdown(self):
        # All time high
        ath = pl.col('PriceUSD').cum_max()
        self.df = self.df.with_columns(
            ((pl.col('PriceUSD')-ath)/ath)
            .alias('drawdown')
        )

    def RSI(self):
        rsi = ta.momentum.rsi(close=self.df["PriceUSD"].to_pandas(), window=14)
        self.df = self.df.with_columns(
            pl.Series("RSI", rsi)
        )

        self.df = self.df.with_columns(
            zscore(pl.col("RSI"), 365).alias("RSI_zscore")
        )
    # def momentum(self, x):

class MacroGenerator:


    def __init__(self):
        self.df = None

    def load_macro_data(self):
        codes = [
            "M2SL",
            "WALCL",
            "DFII10",
            "DTWEXBGS",
            "DGS10",
            "CPIAUCSL"
        ]
        fred = Fred(api_key="4d92c5d4285a1bbfc7fd50d43586314d ")
        series = {c: fred.get_series(c) for c in codes}

        self.df = pd.concat(series, axis=1)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.resample("D").ffill()
        return self.df

    def cleanup(self):
        self.df["fed_balance_growth"] = self.df["WALCL"].pct_change(365)
        self.df["real_rate"] = self.df["DFII10"]
        self.df["usd_strength"] = self.df["DTWEXBGS"]
        self.df["10y_yield"] = self.df["DGS10"]
        self.df["inflation_yoy"] = self.df["CPIAUCSL"].pct_change(365, fill_method=None)
        self.df = self.df.drop(columns=["DFII10", "DTWEXBGS", "DGS10"])
        self.df = self.df.reset_index().rename(columns={'index': 'time'})

        self.df['time'] = self.df['time'].astype('datetime64[us]')
        self.df = self.df.ffill()
        self.df = pl.from_pandas(self.df)
        return self.df

class BinanceGenerator:
    def __init__(self):
        self.df = None

    def load_binance_klines(
            self,
            symbol: str = "BTCUSDT",
            interval: str = "1d",
            start: str = "2017-01-01",
            end: str | None = None,
            limit: int = 1000,
        ) -> pd.DataFrame:
        url = "https://api.binance.com/api/v3/klines"

        cols = [
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = None if end is None else int(pd.Timestamp(end).timestamp() * 1000)

        all_rows = []

        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "limit": limit,
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            if not data:
                break

            all_rows.extend(data)

            last_open_time = data[-1][0]
            next_start_ms = last_open_time + 1

            # if next_start_ms <= start_ms:
            #     break
            #
            start_ms = next_start_ms

            if len(data) < limit:
                break
            # time.sleep(0.2)
        self.df = pd.DataFrame(all_rows, columns=cols).drop_duplicates(subset=["time"])
        self.df["time"] = pd.to_datetime(self.df["time"], unit="ms", utc=True).dt.tz_localize(None)
        self.df["close_time"] = pd.to_datetime(self.df["close_time"], unit="ms", utc=True).dt.tz_localize(None)

        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_base", "taker_buy_quote"
        ]
        self.df[numeric_cols] = self.df[numeric_cols].astype(float)
        self.df["trades"] = self.df["trades"].astype(int)

        self.df = self.df.sort_values("time")
        return self.df

    def funding_rate_history(self, symbol="BTCUSDT", start="2017-01-01", end=None, limit=1000) -> pd.DataFrame:
        """
        Backfill funding rate history from Binance USD-M futures.
        """
        FAPI_BASE = "https://fapi.binance.com"
        url = f"{FAPI_BASE}/fapi/v1/fundingRate"

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = None if end is None else int(pd.Timestamp(end).timestamp() * 1000)

        rows = []

        while True:
            params = {
                "symbol": symbol,
                "startTime": start_ms,
                "limit": limit,
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            if not data:
                break

            rows.extend(data)

            last_time = int(data[-1]["fundingTime"])
            next_start = last_time + 1

            # if next_start <= start_ms:
            #     break

            start_ms = next_start

            if len(data) < limit:
                break

            # time.sleep(0.2)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_localize(None)
        df = df[df['fundingTime'].dt.time == pd.to_datetime('00:00:00').time()]
        df["fundingRate"] = df["fundingRate"].astype(float)
        if "markPrice" in df.columns:
            df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")

        df = df.rename(columns={
            "fundingTime": "time",
            "fundingRate": "funding_rate",
        }).sort_values("time")
        df = df.reset_index(drop=True)
        self.df = pd.merge(self.df, df, on="time", how="left")

        return self.df

    def feature_volatility_volume(self):

        self.df["bin_returns_1d"] = self.df["close"].pct_change()

        self.df["bin_volatility_30d"] = (
            self.df["bin_returns_1d"].rolling(30).std()
        )

        self.df["bin_volume_zscore"] = (
                (self.df["volume"] - self.df["volume"].rolling(30).mean())
                / self.df["volume"].rolling(30).std()
        )

        return self.df


def OnChainIndicators(df):
    generator = OnChainGenerator(df)
    generator.oc_MVRVCur_Z(365)
    generator.oc_CapMrktCurUSD_Z(365)
    generator.oc_Puell_proxy(365)
    generator.oc_miner_stress(365)
    generator.oc_netexchangenorm()
    generator.oc_flowimbalance()
    generator.oc_flow_ratio()
    generator.oc_vol_30d()
    generator.oc_turnover()
    return generator.df

def TechnicalIndicators(df):
    generator = TechnicalGenerator(df)
    # generator.sma_bb(30)
    generator.sma_bb(200)
    # generator.ema_bb(30)
    generator.ema_bb(200)
    generator.drawdown()
    generator.RSI()
    return generator.df

def BinanceIndicators():
    Binance = BinanceGenerator()
    Binance.load_binance_klines("BTCUSDT", "1d", "2017-01-01")
    Binance.funding_rate_history("BTCUSDT")
    Binance.feature_volatility_volume()
    bin_df = pl.from_pandas(Binance.df)
    bin_df = bin_df.with_columns(
        pl.col("time").cast(pl.Datetime("us"))
    )
    return bin_df

def fwd_MA(df, x=30):
    # Label column
    df = df.with_columns(
        (pl.col("PriceUSD").shift(-x) / pl.col("PriceUSD") - 1)
        .alias(f"ROI_{x}d")
    )

    q = df.select(pl.col(f"ROI_{x}d").quantile(0.7)).item()

    df = df.with_columns(
        (pl.col(f"ROI_{x}d") >= q).cast(pl.Int8).alias(f"ROI_{x}d_top30")
    )

    df = df.with_columns(
        pl.col('PriceUSD').rolling_mean(
            window_size = x,
            # 'center=False' is default, meaning window includes current and previous rows
            # Achieve "future" look by shifting the result backward
        ).shift(-(x - 1)  # Shift result backward by (window_size - 1) rows
                ).over(  # Apply window function over the entire dataframe
            # This will calculate the mean of the current and next x-1 rows
            pl.lit(1)  # a constant to treat the whole DF as one partition
        ).alias("future_" + str(x) + "d_avg"),
    )

    df = df.with_columns(
        (pl.col("future_" + str(x) + "d_avg") / pl.col("PriceUSD") - 1).alias("ROI_" + str(x) + "d_avg")
    )

    df = df.with_columns(
        pl.when(pl.col("ROI_" + str(x) + "d_avg") > 0).then(1)
        .otherwise(0)
        .alias("target_up_" + str(x) + "d")
    )

    df = df.with_columns(
        pl.when(pl.col("ROI_" + str(x) + "d_avg") > 0.05).then(1)
        .otherwise(0)
        .alias("target_up5_" + str(x) + "d")
    )

    df = df.with_columns(
        (pl.col("PriceUSD").shift(-x) / pl.col("PriceUSD").rolling_mean(window_size=30) - 1)
        .alias(f"future_{x}d_vs_30d_SMA")
    )

    k = 5  # number of lowest values to average

    df = df.with_columns(
        pl.col("PriceUSD")
        .reverse()
        .rolling_map(
            lambda s: s.sort().head(k).mean() if len(s) >= k else 0,
            window_size=x
        )
        .reverse()
        .shift(-1) # exclude current day
        .alias(f"future_{x}d_kmin_{k}")
    )

    df = df.with_columns(
        (pl.col(f"future_{x}d_kmin_{k}") / pl.col("PriceUSD")).log().fill_null(strategy="forward")
        .alias(f"Future_{x}d_min_rel")
    )

    df = df.with_columns(
        pl.col("ROI_" + str(x) + "d_avg").cut(
            breaks = [-0.2, -0.05, 0.05, 0.2],
            labels = ['1','2','3','4','5'],
        ).alias("target_rank_" + str(x) + "d")
    )

    df = df.with_columns(
        pl.col("target_rank_" + str(x) + "d").cast(pl.Int64, strict=False).alias("target_rank_" + str(x) + "d")
    )

    df = df.with_columns(
        (pl.col("PriceUSD").shift(-x) / pl.col("PriceUSD") - 1)
        .alias(f"ROI_{x}d")
    )

    return df

def btc_target(df):
    temp = fwd_MA(df, 30)
    temp = fwd_MA(temp, 60)
    return temp

class CleanUp:
    def __init__(self, df):
        self.df = df

    def dropcol(self, list):
        self.df = self.df.drop(list)
        return self.df

    def fill_zero(self, list):
        for colname in list:
            self.df = self.df.with_columns(pl.col(colname).fill_null(0))
        return self.df

    def fill_05(self, list):
        for colname in list:
            self.df = self.df.with_columns(pl.col(colname).fill_null(0.5))
        return self.df

    def fill_1(self, list):
        for colname in list:
            self.df = self.df.with_columns(pl.col(colname).fill_null(1))
        return self.df



