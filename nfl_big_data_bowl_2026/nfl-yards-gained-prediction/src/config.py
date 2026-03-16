from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --------------------------------------------------
# NFL Big Data Bowl raw dataset location
# --------------------------------------------------

NFL_DATA_DIR = RAW_DATA_DIR / "nfl_competition_data"

TRAIN_DIR = NFL_DATA_DIR / "train"

SUPPLEMENTARY_PATH = NFL_DATA_DIR / "supplementary_data.csv"

# --------------------------------------------------
# Processed outputs
# --------------------------------------------------

MASTER_PLAY_FEATURES = PROCESSED_DATA_DIR / "master_play_features.csv"
MASTER_SEPARATION_FRAME = PROCESSED_DATA_DIR / "master_separation_frame_level.csv"

MODEL_TABLE_PATH = PROCESSED_DATA_DIR / "model_table_yards_v2.parquet"

# --------------------------------------------------

def check_paths():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("TRAIN_DIR:", TRAIN_DIR)
    print("SUPPLEMENTARY_PATH:", SUPPLEMENTARY_PATH)



# Columns
ID_COLS = ["game_id", "play_id"]
FRAME_COL = "frame_id"
X_COL = "x"
Y_COL = "y"
PLAYER_ROLE_COL = "player_role"
PLAYER_SIDE_COL = "player_side"
PASS_RESULT_COL = "pass_result"
ROUTE_COL = "route_of_targeted_receiver"
COVERAGE_COL = "team_coverage_man_zone"

# For sampling in EDA to keep things light
DEFAULT_SAMPLE_PLAYS = 500