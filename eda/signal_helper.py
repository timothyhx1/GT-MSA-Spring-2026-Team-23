# Dependencies
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

import eda_starter as eda

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor, XGBClassifier
import shap

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]



def ROI_avg_quantile(ROI, rank, min_pctl, max_pctl = 1):
    return ROI.filter(((rank[0] <= max_pctl) & (rank[0] > min_pctl)).to_numpy()).mean()

def get_rank(model, X, len_train, purge_len = 0):
    pred_prob = model.predict_proba(X)[purge_len:, 1]
    rank = pd.DataFrame(pred_prob).rank(pct=True)
    rank = rank.rename(columns={0: 'percentile'})
    rank['train_test'] = 'test'
    # rank.iloc[len_train:(len_train+purge_len), rank.columns.get_loc('train_test')] = 'purge'
    # rank.iloc[(len_train+purge_len):, rank.columns.get_loc('train_test')] = 'test'
    return rank



def merge_ROI_rank(ROI, rank, purge_len = 0):
    ROI_rank = pl.concat([ROI[purge_len:], pl.from_pandas(rank)], how="horizontal")
    p = pl.col('percentile')
    ROI_rank = ROI_rank.with_columns(
        pl.when(p > 0.8).then(pl.lit('0.8-1.0'))
        .when(p > 0.6).then(pl.lit('0.6-0.8'))
        .when(p > 0.4).then(pl.lit('0.4-0.6'))
        .when(p > 0.2).then(pl.lit('0.2-0.4'))
        .when(p > 0).then(pl.lit('0.0-0.2'))
        .otherwise(pl.lit('unknown'))
        .alias('percentile_bucket')
    )
    return ROI_rank

def plot_roi_summary_by_bucket(ROI_rank: pl.DataFrame, threshold: 0):
    roi_cols = ROI_rank.select(pl.col("^ROI.*$")).columns
    if not roi_cols:
        raise ValueError("No ROI column found matching '^ROI.*$'")
    ROI_colname = roi_cols[0]

    # Mean ROI by bucket
    mean_grouped = (
        ROI_rank
        .filter(pl.col(ROI_colname).is_not_null())
        .group_by(["percentile_bucket", "train_test"])
        .agg(pl.col(ROI_colname).mean().alias("mean_roi"))
        .sort("percentile_bucket")
        .pivot(
            values="mean_roi",
            index="percentile_bucket",
            on="train_test"
        )
    )

    # Fraction above threshold by bucket
    frac_grouped = (
        ROI_rank
        .filter(pl.col(ROI_colname).is_not_null())
        .group_by(["percentile_bucket", "train_test"])
        .agg((pl.col(ROI_colname) > threshold).mean().alias("frac_above_threshold"))
        .sort("percentile_bucket")
        .pivot(
            values="frac_above_threshold",
            index="percentile_bucket",
            on="train_test"
        )
    )

    # display(mean_grouped)
    # display(frac_grouped)

    mean_pd = mean_grouped.to_pandas()
    frac_pd = frac_grouped.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_mean = mean_pd["percentile_bucket"]
    x_frac = frac_pd["percentile_bucket"]

    # Left plot: mean ROI
    if "test" in mean_pd.columns:
        axes[0].plot(x_mean, mean_pd["test"], marker="o", label="test")
    if "train" in mean_pd.columns:
        axes[0].plot(x_mean, mean_pd["train"], marker="o", label="train")

    axes[0].set_xlabel("percentile_bucket of model probability prediction")
    axes[0].set_ylabel(f"{ROI_colname} mean")
    axes[0].set_title("Mean ROI by Bucket")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    # Right plot: fraction above threshold
    if "test" in frac_pd.columns:
        axes[1].plot(x_frac, frac_pd["test"], marker="o", label="test")
    if "train" in frac_pd.columns:
        axes[1].plot(x_frac, frac_pd["train"], marker="o", label="train")

    axes[1].set_xlabel("percentile_bucket of model probability prediction")
    axes[1].set_ylabel(f"Fraction {ROI_colname} > {threshold}")
    axes[1].set_title(f"Fraction Above Threshold ({threshold})")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_ROI_by_bucket(ROI_rank):
    ROI_colname = ROI_rank.select(pl.col('^ROI.*$')).columns[0]
    grouped = (
        ROI_rank.filter(pl.col(ROI_colname).is_not_null())
        .group_by(['percentile_bucket', 'train_test'])
        .agg(pl.col(ROI_colname).mean())
        .sort('percentile_bucket')
    ).pivot("train_test", index="percentile_bucket", values=ROI_colname)
    display(grouped)
    # plt.plot(grouped['percentile_bucket'], grouped['train'])
    plt.plot(grouped['percentile_bucket'], grouped['test'])
    plt.legend(['test'])
    plt.xlabel('percentile_bucket of model probability prediction')
    plt.ylabel(f"{ROI_colname} mean")
    plt.show()


def plot_ROI_threshold(ROI_rank, threshold=0):
    ROI_colname = ROI_rank.select(pl.col('^ROI.*$')).columns[0]
    grouped2 = (
        ROI_rank.filter(pl.col(ROI_colname).is_not_null())
        .group_by(['percentile_bucket', 'train_test'])
        .agg((pl.col(ROI_colname) > threshold).mean())
        .sort('percentile_bucket')
    ).pivot("train_test", index="percentile_bucket", values=ROI_colname)
    display(grouped2)
    # plt.plot(grouped2['percentile_bucket'], grouped2['train'])
    plt.plot(grouped2['percentile_bucket'], grouped2['test'])
    plt.legend(['test'])
    plt.xlabel('percentile_bucket of model probability prediction')
    plt.ylabel(f"Fraction {ROI_colname} is more than threshold")
    plt.show()



def plot_ROI_threshold_test(ROI_rank, threshold=0):
    ROI_colname = ROI_rank.select(pl.col('^ROI.*$')).columns[0]
    grouped2 = (
        ROI_rank.filter(pl.col(ROI_colname).is_not_null())
        .group_by(['percentile_bucket'])
        .agg((pl.col(ROI_colname) > threshold).mean())
        .sort('percentile_bucket')
    )
    display(grouped2)
    plt.plot(grouped2['percentile_bucket'], grouped2[ROI_colname])
    plt.xlabel('percentile_bucket of model probability prediction')
    plt.ylabel(f"Fraction {ROI_colname} is more than threshold")
    plt.show()

def get_rank_reg(model, X, len_train, purge_len = 0):
    rank = model.predict(X)
    rank = pd.DataFrame(rank)
    rank = rank.rename(columns={0: 'rank'})
    rank['train_test'] = 'train'
    rank.iloc[len_train+purge_len:, rank.columns.get_loc('train_test')] = 'test'
    return rank

def merge_ROI_rank_reg(ROI, rank):
    ROI_rank = pl.concat([ROI, pl.from_pandas(rank)], how="horizontal")
    p = pl.col('rank')
    ROI_rank = ROI_rank.with_columns(
        pl.when(p > 4.5).then(pl.lit('5'))
        .when(p > 3.5).then(pl.lit('4'))
        .when(p > 2.5).then(pl.lit('3'))
        .when(p > 1.5).then(pl.lit('2'))
        .when(p > 0).then(pl.lit('1'))
        .otherwise(pl.lit('unknown'))
        .alias('rank_bucket')
    )
    return ROI_rank

def plot_ROI_by_bucket_reg(ROI_rank):
    ROI_colname = ROI_rank.select(pl.col('^ROI.*$')).columns[0]
    grouped = (
        ROI_rank.filter(pl.col(ROI_colname).is_not_null())
        .group_by(['rank_bucket', 'train_test'])
        .agg(pl.col(ROI_colname).mean())
        .sort('rank_bucket')
    ).pivot("train_test", index="rank_bucket", values=ROI_colname)
    display(grouped)
    plt.plot(grouped['rank_bucket'], grouped['train'])
    plt.plot(grouped['rank_bucket'], grouped['test'])
    plt.legend(['train', 'test'])
    plt.xlabel('rank_bucket of model probability prediction')
    plt.ylabel(f"{ROI_colname} mean")
    plt.show()

def plot_ROI_threshold_reg(ROI_rank, threshold=0):
    ROI_colname = ROI_rank.select(pl.col('^ROI.*$')).columns[0]
    grouped2 = (
        ROI_rank.filter(pl.col(ROI_colname).is_not_null())
        .group_by(['rank_bucket', 'train_test'])
        .agg((pl.col(ROI_colname) > threshold).mean())
        .sort('rank_bucket')
    ).pivot("train_test", index="rank_bucket", values=ROI_colname)
    display(grouped2)
    plt.plot(grouped2['rank_bucket'], grouped2['train'])
    plt.plot(grouped2['rank_bucket'], grouped2['test'])
    plt.legend(['train', 'test'])
    plt.xlabel('rank_bucket of model probability prediction')
    plt.ylabel(f"Fraction {ROI_colname} is more than threshold")
    plt.show()