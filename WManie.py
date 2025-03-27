#%%
import pandas as pd
import os
import numpy as np
import re
import itertools
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from xgboost import XGBClassifier

#%%
BASE_DIR = "data"


def load_csv(filename):
    return pd.read_csv(os.path.join(BASE_DIR, filename))


teams = load_csv("WTeams.csv")
seasons = load_csv("WSeasons.csv")
tourney_seeds = load_csv("WNCAATourneySeeds.csv")
regular_season_results = load_csv("WRegularSeasonCompactResults.csv")
tourney_results = load_csv("WNCAATourneyCompactResults.csv")
regular_season_detailed = load_csv("WRegularSeasonDetailedResults.csv")
tourney_detailed = load_csv("WNCAATourneyDetailedResults.csv")

#%%
datasets = {
    "Teams": teams,
    "Seasons": seasons,
    "Tourney Seeds": tourney_seeds,
    "Regular Season Results": regular_season_results,
    "Tourney Results": tourney_results,
    "Regular Season Detailed": regular_season_detailed,
    "Tourney Detailed": tourney_detailed
}

for name, df in datasets.items():
    print(f"\n{name} Dataset Sample:")
    print(df.head(), "\n")

#%%
tourney_results["HomeWinBonus"] = (tourney_results["WLoc"] == "H").astype(int)

#%%
tourney_results = tourney_results.merge(
    tourney_seeds, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"Seed": "WSeed"})

tourney_results = tourney_results.merge(
    tourney_seeds, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"Seed": "LSeed"})

# Drop duplicate TeamID columns
tourney_results.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

print("✅ Tournament Seeds Merged")

#%%
regular_season_summary = regular_season_results.groupby(["Season", "WTeamID"]).agg({
    "WScore": "mean",
    "LScore": "mean",
    "NumOT": "mean"
}).reset_index().rename(
    columns={"WTeamID": "TeamID", "WScore": "AvgPointsScored", "LScore": "AvgPointsConceded", "NumOT": "AvgOT"})

tourney_results = tourney_results.merge(
    regular_season_summary, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"AvgPointsScored": "WAvgPointsScored", "AvgPointsConceded": "WAvgPointsConceded", "AvgOT": "WAvgOT"})

tourney_results = tourney_results.merge(
    regular_season_summary, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"AvgPointsScored": "LAvgPointsScored", "AvgPointsConceded": "LAvgPointsConceded", "AvgOT": "LAvgOT"})

# Drop duplicate TeamID columns
tourney_results.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

print("✅ Regular Season Performance Merged")

#%%
# Convert seed values from string to numeric
tourney_results["WSeed"] = tourney_results["WSeed"].astype(str).str.extract("(\d+)").astype(float)
tourney_results["LSeed"] = tourney_results["LSeed"].astype(str).str.extract("(\d+)").astype(float)

# Compute Seed Difference
tourney_results["SeedDiff"] = tourney_results["WSeed"] - tourney_results["LSeed"]

#%%
tourney_results["W_OffensiveRating"] = tourney_results["WAvgPointsScored"] - tourney_results["LAvgPointsConceded"]
tourney_results["L_OffensiveRating"] = tourney_results["LAvgPointsScored"] - tourney_results["WAvgPointsConceded"]

tourney_results["W_DefensiveRating"] = tourney_results["WAvgPointsConceded"] - tourney_results["LAvgPointsScored"]
tourney_results["L_DefensiveRating"] = tourney_results["LAvgPointsConceded"] - tourney_results["WAvgPointsScored"]

#%%
tourney_results_flipped = tourney_results.copy()

# Swap winning and losing team stats
tourney_results_flipped["SeedDiff"] *= -1
tourney_results_flipped["W_OffensiveRating"], tourney_results_flipped["L_OffensiveRating"] = (
    tourney_results_flipped["L_OffensiveRating"],
    tourney_results_flipped["W_OffensiveRating"],
)
tourney_results_flipped["W_DefensiveRating"], tourney_results_flipped["L_DefensiveRating"] = (
    tourney_results_flipped["L_DefensiveRating"],
    tourney_results_flipped["W_DefensiveRating"],
)

# Change the target label (flipping the match outcome)
tourney_results_flipped["WinLabel"] = 0

# Combine the original and flipped datasets
final_dataset = pd.concat([tourney_results, tourney_results_flipped], ignore_index=True)

# Shuffle the dataset
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Check if we now have both classes
print(f"✅ Class Distribution: \n{final_dataset['WinLabel'].value_counts()}")

#%%
print("Missing values in y:", y.isnull().sum())

#%%
print("Missing values in y after cleaning:", y.isnull().sum())

#%%
print("Class distribution in y_train:\n", y_train.value_counts())

#%%
print("Class distribution in full dataset:\n", final_dataset["WinLabel"].value_counts())

#%%
# Compute Win/Loss Streaks
regular_season_results["WinStreak"] = regular_season_results.groupby(["Season", "WTeamID"]).cumcount() + 1
regular_season_results["LoseStreak"] = regular_season_results.groupby(["Season", "LTeamID"]).cumcount() + 1

# Merge Win/Loss streaks with tournament results
final_dataset = final_dataset.merge(
    regular_season_results[["Season", "WTeamID", "WinStreak"]],
    how="left", left_on=["Season", "WTeamID"], right_on=["Season", "WTeamID"]
).rename(columns={"WinStreak": "WWinStreak"})

final_dataset = final_dataset.merge(
    regular_season_results[["Season", "LTeamID", "LoseStreak"]],
    how="left", left_on=["Season", "LTeamID"], right_on=["Season", "LTeamID"]
).rename(columns={"LoseStreak": "LLoseStreak"})

# Compute Point Differential
final_dataset["PointDiff"] = final_dataset["WAvgPointsScored"] - final_dataset["LAvgPointsScored"]

# Fill missing values with 0
final_dataset["WWinStreak"].fillna(0, inplace=True)
final_dataset["LLoseStreak"].fillna(0, inplace=True)
final_dataset["PointDiff"].fillna(0, inplace=True)

print("✅ Features computed before balancing!")

#%%
print("Missing values in y:", y.isnull().sum())

#%%
# Display a few rows where `WinLabel` is missing
missing_rows = final_dataset[final_dataset["WinLabel"].isnull()]
print("Rows with missing WinLabel:\n", missing_rows.head())

# Count missing values in the entire dataset
print("Missing values per column:\n", final_dataset.isnull().sum())

#%%
# If WTeamID exists, set WinLabel = 1 (Winning Team), otherwise 0
final_dataset["WinLabel"] = final_dataset["WTeamID"].notnull().astype(int)

print("✅ Recomputed `WinLabel` successfully!")

#%%
print("Missing values in WinLabel after fixing:", final_dataset["WinLabel"].isnull().sum())

#%%
print("Available columns in final_dataset:", final_dataset.columns.tolist())

#%%
# Print class distribution
print("Class distribution in WinLabel:\n", final_dataset["WinLabel"].value_counts())

#%%
# Duplicate and flip matches to create losses
tourney_results_flipped = final_dataset.copy()

# Swap team-related stats for losing team as winning team
tourney_results_flipped["SeedDiff"] *= -1
tourney_results_flipped["W_OffensiveRating"], tourney_results_flipped["L_OffensiveRating"] = (
    tourney_results_flipped["L_OffensiveRating"],
    tourney_results_flipped["W_OffensiveRating"],
)
tourney_results_flipped["W_DefensiveRating"], tourney_results_flipped["L_DefensiveRating"] = (
    tourney_results_flipped["L_DefensiveRating"],
    tourney_results_flipped["W_DefensiveRating"],
)

# Swap Win/Loss Streaks
tourney_results_flipped["WWinStreak"], tourney_results_flipped["LLoseStreak"] = (
    tourney_results_flipped["LLoseStreak"],
    tourney_results_flipped["WWinStreak"],
)

# Flip Point Differential
tourney_results_flipped["PointDiff"] *= -1

# Flip WinLabel (change 1 → 0 and 0 → 1)
tourney_results_flipped["WinLabel"] = 1 - tourney_results_flipped["WinLabel"]

# Combine original and flipped datasets
final_dataset = pd.concat([final_dataset, tourney_results_flipped], ignore_index=True)

# Shuffle the dataset
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new class distribution
print("✅ Class distribution after balancing:\n", final_dataset["WinLabel"].value_counts())

#%%
# Print column names from the original DataFrame before scaling
print("Training features:", list(X.columns))  # Use X, not X_train
print("New matchups features:", list(X_new.columns))  # X_new should be a DataFrame

#%%
features = ["SeedDiff", "W_OffensiveRating", "L_OffensiveRating",
            "W_DefensiveRating", "L_DefensiveRating", "HomeWinBonus","WWinStreak", "LLoseStreak", "PointDiff"]

X = final_dataset[features]  # Use balanced dataset
y = final_dataset["WinLabel"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)

#%%
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

y_pred_log = log_reg.predict(X_test)
y_pred_proba_log = log_reg.predict_proba(X_test)[:, 1]

log_accuracy = accuracy_score(y_test, y_pred_log)
log_loss_score = log_loss(y_test, y_pred_proba_log)

print(f"✅ Logistic Regression Accuracy: {log_accuracy:.4f}")
print(f"✅ Logistic Regression Log Loss: {log_loss_score:.4f}")

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_loss_score = log_loss(y_test, y_pred_proba_xgb)

print(f"✅ XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"✅ XGBoost Log Loss: {xgb_loss_score:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix (XGBoost):\n", conf_matrix)

#%%
import joblib

# Save the Logistic Regression model
joblib.dump(log_reg, "logistic_regression_model.pkl")

# Save the XGBoost model
joblib.dump(xgb_model, "xgboost_model.pkl")

# Save the Scaler
joblib.dump(scaler, "Predictions/scaler.pkl")

print("✅ Models and scaler saved successfully!")

#%%
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],  # Reduce from [100, 200, 300] → [100, 200]
    "learning_rate": [0.01, 0.1],  # Reduce from [0.01, 0.05, 0.1] → [0.01, 0.1]
    "max_depth": [3, 5],  # Reduce from [3, 5, 7] → [3, 5]
    "subsample": [0.8, 1.0],  # Reduce from [0.7, 0.8, 1.0] → [0.8, 1.0]
    "colsample_bytree": [0.8, 1.0],  # Reduce from [0.7, 0.8, 1.0] → [0.8, 1.0]
    "gamma": [0, 0.1],  # Reduce from [0, 0.1, 0.2] → [0, 0.1]
}

xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("✅ Best Parameters:", grid_search.best_params_)

xgb_best = XGBClassifier(**grid_search.best_params_, objective="binary:logistic", eval_metric="logloss",
                         random_state=42)
xgb_best.fit(X_train, y_train)

y_pred_xgb_tuned = xgb_best.predict(X_test)
y_pred_proba_xgb_tuned = xgb_best.predict_proba(X_test)[:, 1]

xgb_tuned_accuracy = accuracy_score(y_test, y_pred_xgb_tuned)
xgb_tuned_logloss = log_loss(y_test, y_pred_proba_xgb_tuned)

print(f"✅ XGBoost Tuned Accuracy: {xgb_tuned_accuracy:.4f}")
print(f"✅ XGBoost Tuned Log Loss: {xgb_tuned_logloss:.4f}")

joblib.dump(xgb_best, "xgboost_women_march_madness_tuned.pkl")
print("✅ Tuned model saved as xgboost_women_march_madness_tuned.pkl")
