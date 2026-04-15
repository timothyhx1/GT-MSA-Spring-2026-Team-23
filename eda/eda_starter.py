"""
Exploratory Data Analysis (EDA) Starter Template

This template demonstrates how to perform EDA on Bitcoin and Polymarket data
using Polars with lazy evaluation for efficient data processing.
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

# --- Configuration ---
# Robustly determine the project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = SCRIPT_DIR / "plots"
COINMETRICS_PATH = DATA_DIR / "Coin Metrics" / "coinmetrics_btc.csv"
POLYMARKET_DIR = DATA_DIR / "Polymarket"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)


# --- Memory Tracking Utilities ---


def get_memory_usage_mb() -> float:
    """
    Get current memory usage of the process in MB.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_memory(mb: float) -> str:
    """
    Format memory value in MB to human-readable string.

    Args:
        mb: Memory value in megabytes

    Returns:
        Formatted string (e.g., "123.45 MB" or "1.23 GB")
    """
    if mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb / 1024:.2f} GB"


@contextmanager
def track_memory(operation_name: str):
    """
    Context manager to track memory usage before and after an operation.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        None
    """
    memory_before = get_memory_usage_mb()
    print(f"[Memory] Before {operation_name}: {format_memory(memory_before)}")

    try:
        yield
    finally:
        memory_after = get_memory_usage_mb()
        memory_delta = memory_after - memory_before
        print(
            f"[Memory] After {operation_name}: {format_memory(memory_after)} "
            f"(Δ {format_memory(memory_delta)})"
        )


# --- Data Loading Functions ---
def load_bitcoin_data(filepath: Path) -> Optional[pl.DataFrame]:
    """
    Load Bitcoin data from CSV using Polars lazy scan.

    Args:
        filepath: Path to the Coin Metrics CSV file

    Returns:
        Polars DataFrame with parsed datetime column, or None if loading fails
    """
    print(f"Loading Bitcoin data from {filepath}...")
    try:
        with track_memory("loading Bitcoin data"):
            df = (
                pl.scan_csv(filepath, infer_schema_length=10000)
                .with_columns(pl.col("time").str.to_datetime())
                .collect()
            )
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading Bitcoin data: {e}")
        return None

def load_polymarket_data(datadir: Path) -> Optional[dict[str, pl.DataFrame]]:
    """
    Load Polymarket data from parquet files using Polars lazy scan.

    Args:
        datadir: Directory containing Polymarket parquet files

    Returns:
        Dictionary mapping data type names to Polars DataFrames, or None if loading fails
    """
    print(f"Loading Polymarket data from {datadir}...")
    markets_path = datadir / "finance_politics_markets.parquet"
    tokens_path = datadir / "finance_politics_tokens.parquet"
    trades_path = datadir / "finance_politics_trades.parquet"
    odds_path = datadir / "finance_politics_odds_history.parquet"
    event_path = datadir / "finance_politics_event_stats.parquet"
    summary_path = datadir / "finance_politics_summary.parquet"

    data: dict[str, pl.DataFrame] = {}

    try:
        with track_memory("loading Polymarket data"):
            if markets_path.exists():
                # Load with lazy scan, then collect and handle datetime columns
                markets_df = pl.scan_parquet(markets_path).collect()
                
                # Convert datetime columns only if they exist and are strings
                # (parquet files may already have proper datetime types)
                datetime_cols = []
                for col_name in ["created_at", "end_date"]:
                    if col_name in markets_df.columns:
                        col_dtype = markets_df[col_name].dtype
                        if col_dtype == pl.String or col_dtype == pl.Utf8:
                            datetime_cols.append(pl.col(col_name).str.to_datetime())
                
                if datetime_cols:
                    markets_df = markets_df.with_columns(datetime_cols)
                
                # Fix timestamp corruption
                for col in markets_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if markets_df[col].dtype == pl.Datetime or markets_df[col].dtype == pl.Date:
                            if not markets_df[col].is_empty() and markets_df[col].max() < datetime(2020, 1, 1):
                                markets_df = markets_df.with_columns((pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))
                                
                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if markets_df[col].dtype == pl.Datetime or markets_df[col].dtype == pl.Date:
                             markets_df = markets_df.with_columns(
                                 pl.when(pl.col(col) < datetime(2020, 1, 1))
                                 .then(None)
                                 .otherwise(pl.col(col))
                                 .alias(col)
                             )
                
                data["markets"] = markets_df
                print(f"Loaded {len(markets_df)} markets.")

            if odds_path.exists():
                odds_df = pl.scan_parquet(odds_path).collect()
                
                # Fix timestamp corruption
                for col in odds_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if odds_df[col].dtype == pl.Datetime or odds_df[col].dtype == pl.Date:
                            if not odds_df[col].is_empty() and odds_df[col].max() < datetime(2020, 1, 1):
                                odds_df = odds_df.with_columns((pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))
                                
                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if odds_df[col].dtype == pl.Datetime or odds_df[col].dtype == pl.Date:
                             odds_df = odds_df.with_columns(
                                 pl.when(pl.col(col) < datetime(2020, 1, 1))
                                 .then(None)
                                 .otherwise(pl.col(col))
                                 .alias(col)
                             )
                            
                data["odds"] = odds_df
                print(f"Loaded {len(odds_df)} odds history records.")

            if summary_path.exists():
                summary_df = pl.scan_parquet(summary_path).collect()
                
                # Fix timestamp corruption
                for col in summary_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if summary_df[col].dtype == pl.Datetime or summary_df[col].dtype == pl.Date:
                            if not summary_df[col].is_empty() and summary_df[col].max() < datetime(2020, 1, 1):
                                summary_df = summary_df.with_columns((pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))
                                
                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if summary_df[col].dtype == pl.Datetime or summary_df[col].dtype == pl.Date:
                             summary_df = summary_df.with_columns(
                                 pl.when(pl.col(col) < datetime(2020, 1, 1))
                                 .then(None)
                                 .otherwise(pl.col(col))
                                 .alias(col)
                             )
                            
                data["summary"] = summary_df
                print(f"Loaded {len(summary_df)} summary records.")

            if tokens_path.exists():
                tokens_df = pl.scan_parquet(tokens_path).collect()
                data["tokens"] = tokens_df
                print(f"Loaded {len(tokens_df)} tokens records.")

            if trades_path.exists():
                trades_df = pl.scan_parquet(trades_path).collect()

                # Fix timestamp corruption
                for col in trades_df.columns:
                    if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                        if trades_df[col].dtype == pl.Datetime or trades_df[col].dtype == pl.Date:
                            if not trades_df[col].is_empty() and trades_df[col].max() < datetime(2020, 1, 1):
                                trades_df = trades_df.with_columns(
                                    (pl.col(col).cast(pl.Int64) * 1000).cast(pl.Datetime))

                        # Enforce 2020+ constraint (replace placeholders/zeros with null)
                        if trades_df[col].dtype == pl.Datetime or trades_df[col].dtype == pl.Date:
                            trades_df = trades_df.with_columns(
                                pl.when(pl.col(col) < datetime(2020, 1, 1))
                                .then(None)
                                .otherwise(pl.col(col))
                                .alias(col)
                            )

                data["trades"] = trades_df
                print(f"Loaded {len(trades_df)} trades records.")

            if event_path.exists():
                event_df = pl.scan_parquet(event_path).collect()
                data["events"] = event_df
                print(f"Loaded {len(event_df)} events records.")

        return data if data else None
    except Exception as e:
        print(f"Error loading Polymarket data: {e}")
        return None


# --- Bitcoin Analysis Functions ---


def analyze_btc_metrics(df: pl.DataFrame) -> None:
    """
    Analyze Bitcoin metrics and generate summary statistics.

    Args:
        df: Polars DataFrame containing Bitcoin data
    """
    print("\n--- Bitcoin Data Summary ---")

    # Select relevant columns and compute descriptive statistics
    metrics = ["PriceUSD", "CapMrktCurUSD", "HashRate"]
    available_metrics = [col for col in metrics if col in df.columns]

    if available_metrics:
        summary = df.select(available_metrics).describe()
        print(summary)

    # Correlation analysis
    correlation_cols = ["PriceUSD", "CapMrktCurUSD", "HashRate", "TxCnt"]
    available_corr_cols = [col for col in correlation_cols if col in df.columns]

    if len(available_corr_cols) >= 2:
        corr_df = df.select(available_corr_cols).to_pandas()
        corr = corr_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation of Bitcoin Metrics")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "btc_correlation_matrix.png")
        print("Saved btc_correlation_matrix.png")
        plt.close()


# --- Polymarket Analysis Functions ---


def analyze_polymarket_summary(data: dict[str, pl.DataFrame]) -> None:
    """
    Analyze Polymarket data and generate summary statistics.

    Args:
        data: Dictionary containing Polymarket DataFrames
    """
    print("\n--- Polymarket Data Summary ---")

    markets_df = data.get("markets")
    if markets_df is not None:
        print(f"Total Markets: {len(markets_df)}")

        if "active" in markets_df.columns:
            active_count = markets_df["active"].sum()
            print(f"Active Markets: {active_count}")
            print(f"Closed Markets: {len(markets_df) - active_count}")

        if "volume" in markets_df.columns:
            total_volume = markets_df["volume"].sum()
            avg_volume = markets_df["volume"].mean()
            print(f"Total Volume: ${total_volume:,.2f}")
            print(f"Average Volume per Market: ${avg_volume:,.2f}")

    tokens_df = data.get("tokens")
    if tokens_df is not None:
        print(f"Total Tokens: {len(tokens_df)}")
        print(tokens_df.head(5))

    trades_df = data.get("trades")
    if trades_df is not None:
        print(f"Total Trades: {len(trades_df)}")
        print(trades_df.head(5))

    odds_df = data.get("odds")
    if odds_df is not None:
        print(f"Total Odds History Records: {len(odds_df):,}")
        print(odds_df.head(5))

    event_df = data.get("events")
    if event_df is not None:
        print(f"Total Events: {len(event_df)}")
        print(event_df.head(5))

    summary_df = data.get("summary")
    if summary_df is not None and "trade_count" in summary_df.columns:
        total_trades = summary_df["trade_count"].sum()
        print(f"Total Trades: {total_trades:,}")


# --- Visualization Functions ---


def plot_btc_price(df: pl.DataFrame) -> None:
    """
    Plot Bitcoin price history over time.

    Args:
        df: Polars DataFrame containing Bitcoin data with 'time' and 'PriceUSD' columns
    """
    if "time" not in df.columns or "PriceUSD" not in df.columns:
        print("Required columns 'time' or 'PriceUSD' not found in Bitcoin data.")
        return

    # Convert to pandas for plotting (Polars doesn't have direct matplotlib integration)
    plot_df = df.select(["time", "PriceUSD"]).to_pandas()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["time"], plot_df["PriceUSD"], label="BTC Price (USD)")
    plt.title("Bitcoin Price History")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "btc_price_history.png")
    print("Saved btc_price_history.png")
    plt.close()


def plot_polymarket_volume(df: pl.DataFrame) -> None:
    """
    Plot top 10 Polymarket categories by volume.

    Args:
        df: Polars DataFrame containing Polymarket markets data
    """
    if "volume" not in df.columns or "category" not in df.columns:
        print("Columns 'volume' or 'category' not found in Polymarket data.")
        return

    # Use Polars to compute top categories
    top_cats = (
        df.group_by("category")
        .agg(pl.col("volume").sum())
        .sort("volume", descending=True)
        .head(10)
    )

    if len(top_cats) == 0:
        print("No data available for volume by category plot.")
        return

    # Convert to pandas for seaborn plotting
    plot_df = top_cats.to_pandas()
    print(plot_df["category"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=plot_df["volume"], y=plot_df["category"])
    plt.title("Top 10 Polymarket Categories by Volume")
    plt.xlabel("Total Volume")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "polymarket_volume_by_category.png")
    print("Saved polymarket_volume_by_category.png")
    plt.close()


# --- Main Execution ---

def btc_features (btc_df: pl.DataFrame) -> pl.DataFrame:
    btc_df = btc_df.filter(pl.col("PriceUSD").is_not_null())
    btc_df = btc_df.with_columns(
        pl.col("PriceUSD").shift(-1).alias("lag_1d"),
        pl.col("PriceUSD").shift(-2).alias("lag_2d"),
        pl.col("PriceUSD").shift(-3).alias("lag_3d"),
        pl.col("PriceUSD").shift(-7).alias("lag_7d"),
        pl.col("PriceUSD").shift(-30).alias("lag_30d"),
    )
    btc_df = btc_df.with_columns(
        (pl.col("lag_1d") - pl.col("PriceUSD")).alias("delta_1d"),
        (pl.col("lag_2d") - pl.col("PriceUSD")).alias("delta_2d"),
        (pl.col("lag_3d") - pl.col("PriceUSD")).alias("delta_3d"),
        (pl.col("lag_3d") - pl.col("PriceUSD")).alias("delta_7d"),
        (pl.col("lag_30d") - pl.col("PriceUSD")).alias("delta_30d"),
        ((pl.col("lag_1d") - pl.col("PriceUSD")) / pl.col("PriceUSD")).alias("change_1d"),
        ((pl.col("lag_2d") - pl.col("PriceUSD")) / pl.col("PriceUSD")).alias("change_2d"),
        ((pl.col("lag_3d") - pl.col("PriceUSD")) / pl.col("PriceUSD")).alias("change_3d"),
        ((pl.col("lag_7d") - pl.col("PriceUSD")) / pl.col("PriceUSD")).alias("change_7d"),
        ((pl.col("lag_30d") - pl.col("PriceUSD")) / pl.col("PriceUSD")).alias("change_30d")
    )

    return btc_df

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

def categorize_market_sentiment(markets_df_bitcoin: pl.DataFrame)-> pl.DataFrame:
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

    markets_df_bitcoin_new = markets_df_bitcoin[:600].with_columns(
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
        .alias("stance")
    )

    texts = markets_df_bitcoin_new["question"]
    labels = markets_df_bitcoin_new["stance"]
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
    markets_df_bitcoin = markets_df_bitcoin.join(markets_df_bitcoin_new[["market_id", "stance"]], on="market_id",
                                                 how="left")
    markets_df_bitcoin_new = categorize_blank(markets_df_bitcoin, sentiment_model, "question", "stance", "neutral")
    markets_df_bitcoin_new = markets_df_bitcoin_new.with_columns(
        pl.when(pl.col("stance") == "neutral").then(0)
        .when(pl.col("stance") == "bullish").then(1)
        .when(pl.col("stance") == "bearish").then(-1)
        .alias("stance_val")
    )
    return markets_df_bitcoin_new

def categorize_token_sentiment(tokens_df_bitcoin:pl.DataFrame)->pl.DataFrame:
    positive_kw = ["Yes", "Long", "Bitcoin", "BTC", "$BITCOIN", "Up"]
    negative_kw = ["No", "Short", "Down"]

    q = pl.col("outcome")

    tokens_df_bitcoin = tokens_df_bitcoin.with_columns(
        pl.when(any_contains(q, negative_kw))
        .then(-1)
        .when(any_contains(q, positive_kw))
        .then(1)
        .otherwise(0)  # default
        .alias("outcome_val")
    )

    return tokens_df_bitcoin

def merge_tokens_trades(tokens_df_bitcoin:pl.DataFrame, trades_df:pl.DataFrame)->pl.DataFrame:
    trades_df_bitcoin = trades_df.join(tokens_df_bitcoin, on="token_id", how="left").filter(
        pl.col("question").is_not_null())
    trades_df_bitcoin = trades_df_bitcoin.with_columns(
        pl.when(pl.col("side") == "BUY").then(1)
        .when(pl.col("side") == "SELL").then(-1)
        .otherwise(0)
        .alias("side_val")
    )
    trades_df_bitcoin = trades_df_bitcoin.with_columns(
        (pl.col("side_val") * pl.col("stance_val")).alias("trade_val"))
    trades_df_bitcoin = trades_df_bitcoin.with_columns(
        (pl.col("price") * pl.col("size")).alias("transaction"),
        (pl.col("size") * pl.col("trade_val")).alias("size_trade_val"),
        (pl.col("price") * pl.col("size") * pl.col("trade_val")).alias("transaction_trade_val")
    )
    return trades_df_bitcoin

def any_contains(col: pl.Expr, kws: list[str]) -> pl.Expr:
    # case-insensitive substring match
    return pl.any_horizontal([col.str.contains(rf"(?i){k}") for k in kws])

def trades_daily_agg(trades_df_bitcoin:pl.DataFrame)->pl.DataFrame:
    trades_df_bitcoin_day = trades_df_bitcoin.group_by(
        pl.col("timestamp").dt.date().alias("time")
    ).agg(
        pl.col("size").sum().alias("daily_volume"),
        pl.col("transaction").sum().alias("daily_transaction_value"),
        pl.col("size_trade_val").sum().alias("daily_netstance_volume"),
        pl.col("transaction_trade_val").sum().alias("daily_netstance_transaction_value"),
    ).sort("time")

    trades_df_bitcoin_day = trades_df_bitcoin_day.with_columns(
        (pl.col("daily_netstance_volume") / pl.col("daily_volume")).alias("daily_sentiment_by_volume"),
        (pl.col("daily_netstance_transaction_value") / pl.col("daily_transaction_value")).alias(
            "daily_sentiment_by_transaction_value")
    )

    return trades_df_bitcoin_day

def main() -> None:
    """Main execution function for EDA workflow."""
    # Track overall memory usage
    initial_memory = get_memory_usage_mb()
    print(f"\n[Memory] Initial memory usage: {format_memory(initial_memory)}\n")

    # Load data using lazy evaluation
    btc_df = load_bitcoin_data(COINMETRICS_PATH)
    poly_data = load_polymarket_data(POLYMARKET_DIR)

    # Analyze Bitcoin data
    # if btc_df is not None:
    #     with track_memory("analyzing Bitcoin metrics"):
    #         analyze_btc_metrics(btc_df)
    #     with track_memory("plotting Bitcoin price"):
    #         plot_btc_price(btc_df)

    # Analyze Polymarket data
    if poly_data is not None:
        with track_memory("analyzing Polymarket summary"):
            analyze_polymarket_summary(poly_data)
        # if "markets" in poly_data:
        #     with track_memory("plotting Polymarket volume"):
        #         plot_polymarket_volume(poly_data["markets"])

    # Final memory summary
    final_memory = get_memory_usage_mb()
    total_delta = final_memory - initial_memory
    print(
        f"\n[Memory] Final memory usage: {format_memory(final_memory)} "
        f"(Total Δ: {format_memory(total_delta)})"
    )
    print("\nEDA Layout Complete. Check the 'plots' directory for visualizations.")


if __name__ == "__main__":
    main()
