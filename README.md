# Predicting Yards Gained in NFL Passing Plays Using Pre-Throw Player Tracking Data

This project analyzes **NFL Big Data Bowl player tracking data** to
predict **yards gained on passing plays** using only information
available **before the throw**.

The goal is to understand how **route design, defensive coverage, and
spatial positioning** influence offensive performance in passing plays.

------------------------------------------------------------------------

# Project Overview

Modern NFL tracking systems record the position, speed, and direction of
every player on the field multiple times per second. These
high-resolution spatial data enable detailed analysis of passing plays
and player interactions.

In this project, we build a **predictive modeling framework** using only
**pre-throw tracking information**, meaning that no information
occurring after the ball is thrown is used. This avoids data leakage and
ensures that predictions reflect information that would realistically be
available before the play outcome is known.

The study focuses on three main research questions:

1.  **Separation vs Catch Outcome**\
    How does the separation between the targeted receiver and the
    nearest defender influence pass completion?

2.  **Route Design and Play Effectiveness**\
    How do different receiver routes influence yards gained?

3.  **Defensive Coverage and Play Effectiveness**\
    How do defensive coverage schemes (man vs zone) influence yardage
    outcomes?

To answer these questions we:

-   construct spatial features from player tracking data\
-   engineer receiver movement and defender proximity metrics\
-   validate separation as a predictor of pass completion\
-   train machine learning models to predict yards gained

------------------------------------------------------------------------

# Data Source

The data used in this project come from the **NFL Big Data Bowl 2026
Analytics Competition**.

The dataset is publicly available through Kaggle:

https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data

The competition provides **player tracking data for NFL games**,
including:

-   player positions (`x`, `y`)
-   player speed and acceleration
-   orientation and movement direction
-   player roles (offense, defense, targeted receiver)
-   contextual play information (down, yards to go, route, coverage)

The repository includes the **raw tracking files** used in the analysis
so that the project can be fully reproduced.

------------------------------------------------------------------------

# Dataset Setup

Download the dataset from Kaggle:

https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data

After downloading the competition archive, extract the files and place them in the following directory:

```
data/raw/nfl_competition_data/
```

The final folder structure should look like this:

```
data/raw/nfl_competition_data/

supplementary_data.csv

train/
├── input_2023_w01.csv
├── input_2023_w02.csv
├── input_2023_w03.csv
...
├── input_2023_w18.csv
├── output_2023_w01.csv
├── output_2023_w02.csv
...
└── output_2023_w18.csv
```

The notebooks expect the raw data to be located at:

```
data/raw/nfl_competition_data/
```

If the files are placed in a different location, update the paths defined in:

```
src/config.py
```
------------------------------------------------------------------------

# Project Structure

    nfl-yards-gained-prediction/

    data/
    ├── raw/
    │   └── nfl_competition_data/
    │       ├── supplementary_data.csv
    │       └── train/
    │           ├── input_2023_w01.csv ... input_2023_w18.csv
    │           └── output_2023_w01.csv ... output_2023_w18.csv
    │
    ├── processed/
    │   ├── master_play_features.csv
    │   ├── master_separation_frame_level.csv
    │   ├── model_table_yards.csv
    │   ├── model_table_yards.parquet
    │   ├── model_table_yards_data_dictionary.csv
    │   └── model_table_yards_data_dictionary_full.csv
    │
    └── model_table_yards_v2.parquet

    notebooks/
    ├── 01_data_overview.ipynb
    ├── 02_rq1_separation_vs_catch.ipynb
    ├── 03_feature_engineering_yards_model.ipynb
    └── 04_predictive_modeling_yards_gained.ipynb

    src/
    ├── config.py
    ├── data_loading.py
    ├── eda.py
    ├── features.py
    └── plotting.py

    figures/
    tables/

    report/
    └── latex/

    requirements.txt
    README.md

------------------------------------------------------------------------

# Reproducibility Workflow

The analysis is organized into four notebooks.

## 1. Data Overview and Exploratory Analysis

Run:

`notebooks/01_data_overview.ipynb`

Tasks:

-   inspect the raw NFL tracking datasets\
-   explore pass outcomes and route distributions\
-   analyze yards gained and coverage structure\
-   construct intermediate datasets

Outputs:

-   `data/processed/master_play_features.csv`
-   `data/processed/master_separation_frame_level.csv`

------------------------------------------------------------------------

## 2. Separation vs Catch Outcome (RQ1)

Run:

`notebooks/02_rq1_separation_vs_catch.ipynb`

Tasks:

-   analyze receiver--defender separation\
-   compare completions vs incompletions\
-   estimate logistic regression models

Outputs:

-   `figures/separation_completion.png`
-   `tables/logit_sep_only_summary.tex`

------------------------------------------------------------------------

## 3. Feature Engineering for Yards Prediction

Run:

`notebooks/03_feature_engineering_yards_model.ipynb`

Tasks:

-   construct play-level modeling dataset\
-   compute spatial features from tracking data\
-   generate receiver--defender proximity metrics

Outputs:

-   `data/processed/model_table_yards.parquet`
-   `data/processed/model_table_yards_data_dictionary_full.csv`

------------------------------------------------------------------------

## 4. Predictive Modeling of Yards Gained

Run:

`notebooks/04_predictive_modeling_yards_gained.ipynb`

Tasks:

-   train predictive models\
-   evaluate model performance\
-   analyze feature importance

Models evaluated:

-   Linear Regression\
-   Random Forest\
-   Gradient Boosting

Evaluation metrics:

-   MAE\
-   RMSE\
-   R²

Validation strategy:

-   chronological train/test split\
-   time-series cross-validation

------------------------------------------------------------------------

# Modeling Approach

The modeling pipeline focuses on predicting **yards gained on a pass
play** using only **pre-throw information**.

Key feature groups include:

-   receiver--defender separation metrics\
-   top-K nearest defender distances\
-   defender density around the receiver\
-   route type indicators\
-   defensive coverage indicators\
-   contextual play variables (down, yards to go)

Tree-based models such as Random Forest and Gradient Boosting capture
**nonlinear spatial interactions** between players.

------------------------------------------------------------------------

# Notes

-   Only **pre-throw information** is used to avoid data leakage.\
-   Post-throw tracking frames are excluded from feature construction.\
-   The project demonstrates how spatial tracking data can be used to
    understand passing play effectiveness.

------------------------------------------------------------------------

# Authors

Siam Sadman\
Ergys Korsita
