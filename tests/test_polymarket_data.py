import pytest
import pandas as pd
import polars as pl
from datetime import datetime
from pathlib import Path
from template.prelude_template import load_polymarket_data as load_pandas
from eda.eda_starter_template import load_polymarket_data as load_polars

@pytest.fixture
def polymarket_dir():
    return Path("data/Polymarket")

def test_pandas_timestamps():
    """Ensure all Polymarket timestamps loaded via Pandas are correct (post-2000)."""
    data = load_pandas()
    assert data, "No Polymarket data found for Pandas"
    
    for key, df in data.items():
        for col in df.columns:
            if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    if not df[col].empty:
                        # Ensure all dates are reasonably modern (Polymarket started ~2020)
                        min_date = df[col].min()
                        assert min_date >= pd.Timestamp("2020-01-01"), f"Pandas: Column '{col}' in '{key}' contains pre-2020 dates (min: {min_date})"

def test_polars_timestamps(polymarket_dir):
    """Ensure all Polymarket timestamps loaded via Polars are correct (post-2000)."""
    data = load_polars(polymarket_dir)
    assert data, "No Polymarket data found for Polars"
    
    for key, df in data.items():
        for col in df.columns:
            if any(x in col.lower() for x in ["timestamp", "trade", "created_at", "end_date"]):
                if df[col].dtype == pl.Datetime or df[col].dtype == pl.Date:
                    if not df[col].is_empty():
                        min_date = df[col].min()
                        assert min_date >= datetime(2020, 1, 1), f"Polars: Column '{col}' in '{key}' contains pre-2020 dates (min: {min_date})"
