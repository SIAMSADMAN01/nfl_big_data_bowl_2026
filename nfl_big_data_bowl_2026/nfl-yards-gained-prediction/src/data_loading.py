import pandas as pd
from pathlib import Path
from .config import TRAIN_DIR, SUPPLEMENTARY_PATH


def load_tracking_inputs():

    """
    Load all input tracking files from train folder.
    """

    files = sorted(TRAIN_DIR.glob("input_*.csv"))

    dfs = []

    for f in files:
        print("Loading:", f.name)
        df = pd.read_csv(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    print("Tracking input shape:", data.shape)

    return data


def load_supplementary():

    """
    Load supplementary play-level data.
    """

    df = pd.read_csv(SUPPLEMENTARY_PATH)

    print("Supplementary shape:", df.shape)

    return df


def load_merged_input_with_supp():

    """
    Merge tracking input data with supplementary play-level data.
    """

    tracking = load_tracking_inputs()
    supp = load_supplementary()

    merged = tracking.merge(
        supp,
        on=["game_id", "play_id"],
        how="left"
    )

    print("Merged dataset shape:", merged.shape)

    return merged