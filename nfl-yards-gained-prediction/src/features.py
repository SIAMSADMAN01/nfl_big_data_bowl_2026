# src/features.py
import numpy as np
import pandas as pd
from typing import Tuple

from .config import (
    PLAYER_ROLE_COL,
    PLAYER_SIDE_COL,
    ID_COLS,
    FRAME_COL,
    X_COL,
    Y_COL,
)

def compute_wr_and_defenders(input_with_supp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into targeted receiver rows and defender rows.
    """
    wr = input_with_supp[input_with_supp[PLAYER_ROLE_COL] == "Targeted Receiver"].copy()
    defenders = input_with_supp[input_with_supp[PLAYER_SIDE_COL] == "Defense"].copy()

    # Keep only needed cols + IDs for efficiency
    wr = wr[
        ID_COLS
        + [FRAME_COL, "nfl_id", X_COL, Y_COL, "player_name", "player_position"]
    ].rename(columns={"nfl_id": "nfl_id_wr"})

    defenders = defenders[
        ID_COLS
        + [FRAME_COL, "nfl_id", X_COL, Y_COL, "player_name", "player_position"]
    ].rename(columns={"nfl_id": "nfl_id_db"})

    return wr, defenders


def compute_frame_level_separation(input_with_supp: pd.DataFrame) -> pd.DataFrame:
    """
    For each frame, compute distance between the targeted receiver and
    every defender, then keep the minimum separation (nearest defender).
    Returns a DataFrame with per-frame separation.
    """
    wr, defenders = compute_wr_and_defenders(input_with_supp)

    merged = wr.merge(
        defenders,
        on=ID_COLS + [FRAME_COL],
        suffixes=("_wr", "_db"),
        how="inner",
    )

    merged["separation"] = np.sqrt(
        (merged[f"{X_COL}_wr"] - merged[f"{X_COL}_db"]) ** 2
        + (merged[f"{Y_COL}_wr"] - merged[f"{Y_COL}_db"]) ** 2
    )

    # For each WR/frame, keep the nearest defender
    sep_df = (
        merged.groupby(ID_COLS + ["nfl_id_wr", FRAME_COL], as_index=False)["separation"]
        .min()
    )
    return sep_df


def compute_separation_at_throw(
    sep_df: pd.DataFrame, input_with_supp: pd.DataFrame
) -> pd.DataFrame:
    """
    Approximate 'throw frame' as the first frame in the input for that play/WR.
    Then take separation at that frame.
    """
    # Find first frame for each WR in input
    first_frames = (
        input_with_supp[input_with_supp[PLAYER_ROLE_COL] == "Targeted Receiver"]
        .groupby(ID_COLS + ["nfl_id"], as_index=False)[FRAME_COL]
        .min()
        .rename(columns={"nfl_id": "nfl_id_wr", FRAME_COL: "throw_frame"})
    )

    sep_with_throw = sep_df.merge(
        first_frames, on=ID_COLS + ["nfl_id_wr"], how="inner"
    )

    # Keep only rows at throw_frame
    sep_at_throw = sep_with_throw[
        sep_with_throw[FRAME_COL] == sep_with_throw["throw_frame"]
    ].copy()

    sep_at_throw = sep_at_throw.drop(columns=["throw_frame"])
    return sep_at_throw


def aggregate_play_features(
    sep_df: pd.DataFrame, input_with_supp: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-play aggregated features from the frame-level separations.
    Example: min separation, mean separation, separation at throw.
    """
    # Separation at throw
    sep_at_throw = compute_separation_at_throw(sep_df, input_with_supp)
    sep_at_throw = sep_at_throw.rename(columns={"separation": "sep_at_throw"})

    # ✅ Min and mean separation per play/WR using DataFrameGroupBy
    agg = (
        sep_df
        .groupby(ID_COLS + ["nfl_id_wr"], as_index=False)
        .agg(
            sep_min=("separation", "min"),
            sep_mean=("separation", "mean"),
        )
    )

    play_feats = agg.merge(
        sep_at_throw[ID_COLS + ["nfl_id_wr", "sep_at_throw"]],
        on=ID_COLS + ["nfl_id_wr"],
        how="left",
    )

    context_cols = [
        "pass_result",
        "route_of_targeted_receiver",
        "team_coverage_man_zone",
        "pass_length",
        "yards_gained",
    ]
    context = (
        input_with_supp[input_with_supp[PLAYER_ROLE_COL] == "Targeted Receiver"]
        .groupby(ID_COLS + ["nfl_id"], as_index=False)[context_cols]
        .first()
        .rename(columns={"nfl_id": "nfl_id_wr"})
    )

    play_feats = play_feats.merge(
        context, on=ID_COLS + ["nfl_id_wr"], how="left",
    )

    return play_feats