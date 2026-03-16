# src/eda.py
import pandas as pd
from typing import Dict

from .config import PASS_RESULT_COL, ROUTE_COL, COVERAGE_COL

def basic_pass_result_counts(supp_df: pd.DataFrame) -> pd.Series:
    return supp_df[PASS_RESULT_COL].value_counts(dropna=False)


def basic_route_counts(supp_df: pd.DataFrame) -> pd.Series:
    return supp_df[ROUTE_COL].value_counts(dropna=False)


def basic_coverage_counts(supp_df: pd.DataFrame) -> pd.Series:
    return supp_df[COVERAGE_COL].value_counts(dropna=False)


def describe_play_features(play_feats: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Quick descriptive stats on the main separation features.
    """
    desc = {
        "sep_min": play_feats["sep_min"].describe(),
        "sep_mean": play_feats["sep_mean"].describe(),
        "sep_at_throw": play_feats["sep_at_throw"].describe(),
    }
    return desc