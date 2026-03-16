# src/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

from .config import (
    PASS_RESULT_COL,
    ROUTE_COL,
    COVERAGE_COL,
    ID_COLS,
    FRAME_COL,
)

sns.set(style="whitegrid")


def plot_pass_result_distribution(supp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Sort categories for stable order
    order = sorted(supp_df[PASS_RESULT_COL].dropna().unique())
    sns.countplot(data=supp_df, x=PASS_RESULT_COL, order=order, ax=ax)
    ax.set_title("Pass Result Distribution")
    ax.set_xlabel("Pass Result")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_route_type_distribution(supp_df: pd.DataFrame, top_n: Optional[int] = None):
    counts = supp_df[ROUTE_COL].value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)

    df = counts.reset_index()
    df.columns = [ROUTE_COL, "count"]

    fig_height = max(4, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    sns.barplot(data=df, y=ROUTE_COL, x="count", ax=ax)
    ax.set_title("Route Type Frequency")
    ax.set_xlabel("Count")
    ax.set_ylabel("Route Type")
    fig.tight_layout()
    return fig


def plot_coverage_distribution(supp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=supp_df, x=COVERAGE_COL, ax=ax)
    ax.set_title("Coverage Type Distribution (Man vs Zone)")
    ax.set_xlabel("Coverage Type")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_separation_at_throw_by_result(play_feats: pd.DataFrame):
    df = play_feats.dropna(subset=["sep_at_throw"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=PASS_RESULT_COL, y="sep_at_throw", ax=ax)
    ax.set_title("Separation at Throw by Pass Result")
    ax.set_xlabel("Pass Result")
    ax.set_ylabel("Separation at Throw (yards)")
    fig.tight_layout()
    return fig


def plot_sample_separation_timeseries(
    sep_df: pd.DataFrame,
    play_feats: pd.DataFrame,
    n_samples: int = 5,
):
    """
    Plot separation over time for a few random plays, using filtered play_feats
    to define which (game_id, play_id, nfl_id_wr, pass_result) combinations are eligible.
    """
    # Minimal mapping (play-level + WR) with pass_result
    minimal_feats = play_feats[ID_COLS + ["nfl_id_wr", PASS_RESULT_COL]].drop_duplicates()
    sep_with_result = sep_df.merge(
        minimal_feats, on=ID_COLS + ["nfl_id_wr"], how="inner"
    )

    # Sample keys from the filtered subset
    unique_keys = sep_with_result[ID_COLS + ["nfl_id_wr"]].drop_duplicates()
    if len(unique_keys) > n_samples:
        sample_keys = unique_keys.sample(n=n_samples, random_state=42)
    else:
        sample_keys = unique_keys

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in sample_keys.iterrows():
        mask = (
            (sep_with_result["game_id"] == row["game_id"])
            & (sep_with_result["play_id"] == row["play_id"])
            & (sep_with_result["nfl_id_wr"] == row["nfl_id_wr"])
        )
        sub = sep_with_result[mask].sort_values(FRAME_COL)
        if sub.empty:
            continue
        label = f"{int(row['game_id'])}-{int(row['play_id'])} ({sub[PASS_RESULT_COL].iloc[0]})"
        ax.plot(sub[FRAME_COL], sub["separation"], marker="o", alpha=0.7, label=label)

    ax.set_title("Separation Over Time for Sample Plays")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Separation to Nearest Defender (yards)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ---------- EXTRA VISUALS FOR EDA / RQ2 / RQ3 ----------

def plot_completion_rate_by_route(
    play_feats: pd.DataFrame,
    min_samples: int = 30,
    top_n: int = 15,
):
    """
    Bar plot of completion rate by route type, filtered to routes
    with at least min_samples occurrences, showing top_n by count.
    """
    df = play_feats.dropna(subset=[ROUTE_COL, PASS_RESULT_COL])
    grouped = (
        df.groupby(ROUTE_COL)[PASS_RESULT_COL]
        .agg(
            attempts="count",
            completions=lambda x: (x == "C").sum(),
        )
        .reset_index()
    )
    grouped["completion_rate"] = grouped["completions"] / grouped["attempts"]

    # Filter by min_samples
    grouped = grouped[grouped["attempts"] >= min_samples]

    # Sort by attempts and take top_n
    grouped = grouped.sort_values("attempts", ascending=False).head(top_n)

    fig_height = max(4, 0.35 * len(grouped))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    sns.barplot(data=grouped, y=ROUTE_COL, x="completion_rate", ax=ax)
    ax.set_title(f"Completion Rate by Route Type\n(min {min_samples} targets, top {top_n} by volume)")
    ax.set_xlabel("Completion Rate")
    ax.set_ylabel("Route Type")
    fig.tight_layout()
    return fig


def plot_completion_rate_by_coverage(play_feats: pd.DataFrame, min_samples: int = 30):
    """
    Bar plot of completion rate by coverage type (man / zone).
    """
    df = play_feats.dropna(subset=[COVERAGE_COL, PASS_RESULT_COL])
    grouped = (
        df.groupby(COVERAGE_COL)[PASS_RESULT_COL]
        .agg(
            attempts="count",
            completions=lambda x: (x == "C").sum(),
        )
        .reset_index()
    )
    grouped = grouped[grouped["attempts"] >= min_samples]
    grouped["completion_rate"] = grouped["completions"] / grouped["attempts"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=grouped, x=COVERAGE_COL, y="completion_rate", ax=ax)
    ax.set_title(f"Completion Rate by Coverage Type\n(min {min_samples} targets)")
    ax.set_xlabel("Coverage Type")
    ax.set_ylabel("Completion Rate")
    fig.tight_layout()
    return fig


def plot_separation_by_coverage(play_feats: pd.DataFrame):
    """
    Boxplot of separation at throw by coverage type.
    """
    df = play_feats.dropna(subset=["sep_at_throw", COVERAGE_COL])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=COVERAGE_COL, y="sep_at_throw", ax=ax)
    ax.set_title("Separation at Throw by Coverage Type")
    ax.set_xlabel("Coverage Type")
    ax.set_ylabel("Separation at Throw (yards)")
    fig.tight_layout()
    return fig


def plot_sep_at_throw_by_route(
    play_feats: pd.DataFrame,
    top_n: int = 10,
    min_samples: int = 30,
):
    """
    Boxplot of separation at throw by most common route types.
    """
    df = play_feats.dropna(subset=["sep_at_throw", ROUTE_COL])

    # Find top routes by count with at least min_samples
    counts = df[ROUTE_COL].value_counts()
    valid_routes = counts[counts >= min_samples].head(top_n).index
    df = df[df[ROUTE_COL].isin(valid_routes)]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(valid_routes)), 5))
    sns.boxplot(data=df, x=ROUTE_COL, y="sep_at_throw", ax=ax)
    ax.set_title(f"Separation at Throw by Route Type\n(top {top_n} routes, min {min_samples} targets)")
    ax.set_xlabel("Route Type")
    ax.set_ylabel("Separation at Throw (yards)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_result_coverage_heatmap(play_feats: pd.DataFrame):
    """
    Heatmap of normalized counts: coverage type vs pass result.
    """
    df = play_feats.dropna(subset=[COVERAGE_COL, PASS_RESULT_COL])
    ct = pd.crosstab(df[COVERAGE_COL], df[PASS_RESULT_COL], normalize="index")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(ct, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Pass Result Distribution by Coverage Type (row-normalized)")
    ax.set_xlabel("Pass Result")
    ax.set_ylabel("Coverage Type")
    fig.tight_layout()
    return fig