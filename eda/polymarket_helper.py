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

def zscore(series: pl.Expr, window: int) -> pl.Series:
    """Compute rolling z-score."""
    mean = series.rolling_mean(window_size=window, min_periods=window // 2)
    std = series.rolling_std(window_size=window, min_periods=window // 2)
    return (
        pl.when(std > 1e-8)
        .then((series - mean) / std)
        .otherwise(None)
    )

def categorize_markets(markets_df: pl.DataFrame) -> pl.DataFrame:
    # Cleanup and combine smaller categories
    markets_df = markets_df.with_columns(
        pl.when(pl.col("category") == "Politics")
        .then(pl.lit("Global Politics"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    markets_df = markets_df.with_columns(
        pl.when(pl.col("category") == "Ukraine & Russia")
        .then(pl.lit("Global Politics"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    markets_df = markets_df.with_columns(
        pl.when(pl.col("category") == "Coronavirus-")
        .then(pl.lit("Other"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    markets_df = markets_df.with_columns(
        pl.when(pl.col("category") == "Pop-Culture ")
        .then(pl.lit("Other"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    markets_df = markets_df.with_columns(
        pl.when(pl.col("category") == "Coronavirus")
        .then(pl.lit("Other"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    markets_df = markets_df.with_columns(
        pl.when(pl.col("category") == "Tech")
        .then(pl.lit("Other"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    # Training to identify correct category
    markets_df_not_null = markets_df.filter(pl.col("category") != "")
    texts = markets_df_not_null["question"]
    labels = markets_df_not_null["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            analyzer="word"
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, digits=3))

    markets_df_new = categorize_blank(markets_df, model)

    patterns_list = ["btc", "bitcoin"]
    markets_df_new = markets_df_new.with_columns(
        pl.when(
            pl.col("question").str.contains_any(
                patterns_list,
                ascii_case_insensitive=True
            ) & (pl.col("category") == "Crypto")
        )
        .then(pl.lit("Bitcoin"))
        .otherwise(pl.col("category"))
        .alias("category")
    )

    return markets_df_new

def categorize_blank(
        df: pl.DataFrame,
        model,
        textcol="question",
        catcol="category",
        unknown_label: str = "Other",
        threshold: float = 0.0
) -> pl.DataFrame:
    # Create mask for empty values
    mask = (pl.col(catcol) == "") | (pl.col(catcol).is_null())
    df_blank = df.filter(mask).select(textcol)
    texts = df_blank[textcol].to_list()

    # Predict
    preds = model.predict(texts)
    scores = model.decision_function(texts)
    scores = np.asarray(scores)

    # Multiclass margin handling
    if scores.ndim == 1:
        conf = np.abs(scores)
    else:
        conf = np.max(scores, axis=1)

    thresh_preds = np.where(conf >= threshold, preds, unknown_label)

    # Output
    df_idx = df.with_row_index("_idx")
    df_blank_idx = df_idx.filter(mask).select("_idx")
    pred_df = pl.DataFrame({
        "_idx": df_blank_idx["_idx"],
        "new_cat": thresh_preds.tolist()
    })

    out = (
        df_idx.join(pred_df, on="_idx", how="left")
        .with_columns(
            pl.when((pl.col(catcol) == "") | pl.col(catcol).is_null())
            .then(pl.col("new_cat"))
            .otherwise(pl.col(catcol))
            .alias(catcol)
        )
        .drop(["_idx", "new_cat"])
    )
    return out

def categorize_market_sentiment(df: pl.DataFrame)-> pl.DataFrame:
    # Keywords
    bullish_kw = [
        "reach", "approve", "hit", "sec approve", "ath", "all time high", "blackrock", "allow", "vanguard",
        "up or down",
        "approval", "above", "etf", "spot etf", "buy", "reach", "break", "rally", "listed", "purchase", "hold",
        "acquires", "accepts"
    ]
    bearish_kw = [
        "dips", "drop", "fall", "below", "dip",
        "reject", "deny", "lawsuit", "ban", "crackdown", "hack",
        "crash", "dump", "bear", "down", "not approve", "sell", "liquidate"
    ]
    neutral_kw = [
        "or", "better", "interview", "between",
        "which", "or", "first", "perform better", "vs", "versus",
        "say", "tweet", "0 times", "fees", "capture more", "single day fees", "less than"
    ]

    q = pl.col("question")

    rule1 = (
            q.str.contains(r"(?i)hit")
            & q.str.contains(r"(?i)\sor\s")
            & q.str.contains(r"(?i)first")
    )

    rule2 = (
            q.str.contains(r"(?i)up")
            & q.str.contains(r"(?i)\sor\s")
            & q.str.contains(r"(?i)down")
    )

    temp = df[:600].with_columns(
        pl.when(rule1)
        .then(pl.lit("neutral"))
        .when(rule2)
        .then(pl.lit("bullish"))
        .when(any_contains(q, bearish_kw))
        .then(pl.lit("bearish"))
        .when(any_contains(q, bullish_kw))
        .then(pl.lit("bullish"))
        .when(any_contains(q, neutral_kw))
        .then(pl.lit("neutral"))
        .otherwise(pl.lit("neutral"))  # default
        .alias("poly_mkt_sentiment")
    )

    texts = temp["question"]
    labels = temp["poly_mkt_sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    sentiment_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            analyzer="word"
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    sentiment_model.fit(X_train, y_train)
    pred = sentiment_model.predict(X_test)
    print(classification_report(y_test, pred, digits=3))
    df = df.join(temp[["market_id", "poly_mkt_sentiment"]], on="market_id",
                                                 how="left")
    df = categorize_blank(df, sentiment_model, "question", "poly_mkt_sentiment", "neutral")
    df = df.with_columns(
        pl.when(pl.col("poly_mkt_sentiment") == "neutral").then(0)
        .when(pl.col("poly_mkt_sentiment") == "bullish").then(1)
        .when(pl.col("poly_mkt_sentiment") == "bearish").then(-1)
        .alias("poly_mkt_sentiment_dir")
    )

    return df

def categorize_token_sentiment(df:pl.DataFrame)->pl.DataFrame:
    positive_kw = ["Yes", "Long", "Bitcoin", "BTC", "$BITCOIN", "Up"]
    negative_kw = ["No", "Short", "Down"]

    q = pl.col("outcome")

    df = df.with_columns(
        pl.when(any_contains(q, negative_kw))
        .then(-1)
        .when(any_contains(q, positive_kw))
        .then(1)
        .otherwise(0)  # default
        .alias("poly_outcome_dir")
    )

    return df

def merge_tokens_trades(tokens:pl.DataFrame, trades:pl.DataFrame)->pl.DataFrame:
    df = trades.join(tokens, on="token_id", how="left").filter(
        pl.col("question").is_not_null())

    df = df.with_columns(
        pl.when(pl.col("side") == "BUY").then(1)
        .when(pl.col("side") == "SELL").then(-1)
        .otherwise(0)
        .alias("poly_token_dir")
    )

    df = df.with_columns(
        (pl.col("poly_token_dir") * pl.col("poly_mkt_sentiment_dir")).alias("poly_trade_sentiment_dir"))

    df = df.with_columns(
        # Transaction value
        (pl.col("price") * pl.col("size")).alias("poly_trade_trx"),
        # Directional bet volume
        (pl.col("size") * pl.col("poly_trade_sentiment_dir")).alias("poly_trade_size_dir"),
        # Directional bet size
        (pl.col("price") * pl.col("size") * pl.col("poly_trade_sentiment_dir")).alias("poly_trade_trx_dir")
    )

    df = df.with_columns(
        pl.when(pl.col("poly_mkt_sentiment") == "bullish").then(pl.col("price"))
        .when(pl.col("poly_mkt_sentiment") == "bearish").then(1 - pl.col("price"))
        .alias("odds")
    )

    df = df.with_columns(
        (pl.col("end_date") - pl.col("timestamp")).dt.total_days().alias("odds_duration")
    )

    # odds weight
    df = df.with_columns(
        pl.when(pl.col("odds_duration") > 30).then(pl.lit("long"))
        .when(pl.col("odds_duration") <= 30).then(pl.lit("short"))
        .otherwise(pl.lit("unknown"))
        .alias("odds_long_short")
    )

    df = df.with_columns(
        pl.when(pl.col("odds") > 0.5).then(1)
        .otherwise(0)
        .alias("odds_is_bullish")
    )

    return df

def any_contains(col: pl.Expr, kws: list[str]) -> pl.Expr:
    # case-insensitive substring match
    return pl.any_horizontal([col.str.contains(rf"(?i){k}") for k in kws])

def trades_daily_agg(df:pl.DataFrame)->pl.DataFrame:
    # Distance to expiry

    # markets_df_bitcoin_new = markets_df_bitcoin_new.with_columns([
    #     (pl.col("end_date") - pl.col("timestamp")).dt.days().alias("time_to_expiry")
    # ])
    temp = df.group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        # Trade frequency
        pl.count().alias("poly_dtrade_count"),
        # Unique traders (maker/taker)
        pl.n_unique("taker_address").alias("poly_dtrade_active_traders"),
        # Average trade value
        pl.mean("poly_trade_trx").alias("poly_dtrade_avg_trx"),
        # Total trade value
        pl.col("poly_trade_trx").sum().alias("poly_dtrade_trx"),
        # Net trade directional transaction value or flow imbalance
        pl.col("poly_trade_trx_dir").sum().alias("poly_dtrade_trx_dir"),
        # Dispersion
        pl.col("poly_trade_trx_dir").std().alias("poly_dtrade_trx_std"),

        # No of markets
        pl.col("market_id").n_unique().alias("odds_markets"),
        pl.col("odds").mean().alias("odds_avg"),
        pl.col("odds").std().alias("odds_dispersion"),
        # Percentage of markets bullish
        pl.col("odds_is_bullish").mean().alias("odds_breadth")
    ).sort("time")



    # Long term market
    long = df.filter(
        pl.col("odds_long_short") == "long"
    ).group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        pl.col("market_id").n_unique().alias("odds_long_market"),
        pl.col("odds").mean().alias("odds_long_avg"),
    )

    # Short term market
    short = df.filter(
        pl.col("odds_long_short") == "short"
    ).group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        pl.col("market_id").n_unique().alias("odds_short_market"),
        pl.col("odds").mean().alias("odds_short_avg"),
    )

    temp = temp.with_columns(
        pl.col("odds_avg").diff().alias("odds_momentum"),
    )

    temp = temp.join(long, on="time", how="left")
    temp = temp.join(short, on="time", how="left")

    temp = temp.with_columns(
        (pl.col("odds_long_avg") - pl.col("odds_short_avg")).alias("odds_duration_divergence")
    )

    # z-score
    temp = temp.with_columns(
        zscore(pl.col("poly_dtrade_trx"), 30).alias("poly_dtrade_trx_zscore"),
        zscore(pl.col("poly_dtrade_trx_dir"), 30).alias("poly_dtrade_trx_dir_zscore"),
        zscore(pl.col("odds_avg"), 30).alias("odds_avg_zscore"),
        zscore(pl.col("odds_long_avg"), 30).alias("odds_long_zscore"),
        zscore(pl.col("odds_short_avg"), 30).alias("odds_short_zscore")
    )

    # Conviction
    temp = temp.with_columns(
        (pl.col("poly_dtrade_trx_dir") / pl.col("poly_dtrade_trx")).alias(
            "poly_dtrade_trx_dir_ratio")
    )

    temp = temp.with_columns(
        pl.col("time").cast(pl.Datetime("us")).alias("time")
    )

    return temp

def merge_markets_odds(markets:pl.DataFrame, odds:pl.DataFrame)->pl.DataFrame:
    # Filter only for BTC and non-neutral questions
    df = odds.join(markets, on="market_id", how="inner").filter(
        (pl.col("question").is_not_null()) & (pl.col("poly_mkt_sentiment") != "neutral"))

    # Calculate bullish odds
    df = df.with_columns(
        pl.when(pl.col("poly_mkt_sentiment") == "bullish").then(pl.col("price"))
        .when(pl.col("poly_mkt_sentiment") == "bearish").then(1-pl.col("price"))
        .alias("odds")
    )

    # Calculate duration until market ends
    df = df.with_columns(
        (pl.col("end_date") - pl.col("timestamp")).dt.total_days().alias("odds_duration")
    )

    # Categorize whether market ends in long or short
    df = df.with_columns(
        pl.when(pl.col("odds_duration") > 30).then(pl.lit("long"))
        .when(pl.col("odds_duration") <= 30).then(pl.lit("short"))
        .otherwise(pl.lit("unknown"))
        .alias("odds_long_short")
    )

    df = df.with_columns(
        pl.when(pl.col("odds") > 0.5).then(1)
        .otherwise(0)
        .alias("odds_is_bullish")
    )

    # Long
    long = df.filter(
        pl.col("odds_long_short") == "long"
    ).group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        pl.col("market_id").n_unique().alias("odds_long_market"),
        pl.col("odds").mean().alias("odds_long_avg"),
    )

    short = df.filter(
        pl.col("odds_long_short") == "short"
    ).group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        pl.col("market_id").n_unique().alias("odds_short_market"),
        pl.col("odds").mean().alias("odds_short_avg"),
    )

    temp = df.group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        # No of markets
        pl.col("market_id").n_unique().alias("odds_markets"),
        pl.col("odds").mean().alias("odds_avg"),
        pl.col("odds").std().alias("odds_dispersion"),

        # Percentage of markets bullish
        pl.col("odds_is_bullish").mean().alias("odds_breadth")
    )

    temp = temp.sort("time").with_columns(
        pl.col("odds_avg").diff().alias("odds_momentum"),
    )

    temp = temp.join(long, on="time", how="left")
    temp = temp.join(short, on="time", how="left")


    return temp