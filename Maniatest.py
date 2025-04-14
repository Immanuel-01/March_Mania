#%%
import pandas as pd
import os


# Define the base data directory
BASE_DIR = "data"

# Function to load datasets dynamically
def load_csv(filename):
    return pd.read_csv(os.path.join(BASE_DIR, filename))

# Load datasets without adding "data/" each time
teams = load_csv("MTeams.csv")
w_teams = load_csv("WTeams.csv")  # Women's Teams
seasons = load_csv("MSeasons.csv")
tourney_seeds = load_csv("MNCAATourneySeeds.csv")
regular_season_results = load_csv("MRegularSeasonCompactResults.csv")
tourney_results = load_csv("MNCAATourneyCompactResults.csv")
massey_rankings = load_csv("MMasseyOrdinals.csv")
regular_season_detailed = load_csv("MRegularSeasonDetailedResults.csv")
tourney_detailed = load_csv("MNCAATourneyDetailedResults.csv")

# Verify successful loading
datasets = {
    "Teams": teams,
    "Seasons": seasons,
    "Tourney Seeds": tourney_seeds,
    "Regular Season Results": regular_season_results,
    "Tourney Results": tourney_results,
    "Massey Rankings": massey_rankings,
    "Regular Season Detailed": regular_season_detailed,
    "Tourney Detailed": tourney_detailed
}

for name, df in datasets.items():
    print(f"\n{name} Dataset Sample:")
    print(df.head(), "\n")

#%%
# Compute Home Court Bonus (1 if the winning team played at home, 0 otherwise)
tourney_results["HomeWinBonus"] = (tourney_results["WLoc"] == "H").astype(int)
#%%
# Using 'dataframe' instead of 'df'
for name, dataframe in datasets.items():
    print(f"{name} Dataset: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns")

#%%
for name, dataframe in datasets.items():
    print(f"\n{name} Dataset - Unique Values per Column:")
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        print(f"{column}: {len(unique_values)} unique values")

#%%
for name, dataframe in datasets.items():
    duplicate_count = dataframe.duplicated().sum()
    print(f"{name} Dataset - Duplicate Rows: {duplicate_count}")

#%%
for name, dataframe in datasets.items():
    print(f"\n{name} Dataset - Summary Statistics:")
    print(dataframe.describe(), "\n")

#%%
for name, dataframe in datasets.items():
    missing_values = dataframe.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        print(f"\n{name} Dataset - Missing Values:")
        print(missing_values[missing_values > 0])  # Show only columns with missing values
    else:
        print(f"\n{name} Dataset - No Missing Values.")

#%%
# Merge Tournament Seeds to get Seed information for both winning and losing teams
tourney_results = datasets["Tourney Results"]
tourney_seeds = datasets["Tourney Seeds"]

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
regular_season_results = datasets["Regular Season Results"]

# Aggregate season performance stats
regular_season_summary = regular_season_results.groupby(["Season", "WTeamID"]).agg({
    "WScore": "mean",  # Average points scored
    "LScore": "mean",  # Average points conceded
    "NumOT": "mean"    # Average OT games played
}).reset_index().rename(columns={"WTeamID": "TeamID", "WScore": "AvgPointsScored", "LScore": "AvgPointsConceded", "NumOT": "AvgOT"})

# Merge season stats with the tournament data (for winning teams)
tourney_results = tourney_results.merge(
    regular_season_summary, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"AvgPointsScored": "WAvgPointsScored", "AvgPointsConceded": "WAvgPointsConceded", "AvgOT": "WAvgOT"})

# Merge season stats with the tournament data (for losing teams)
tourney_results = tourney_results.merge(
    regular_season_summary, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"AvgPointsScored": "LAvgPointsScored", "AvgPointsConceded": "LAvgPointsConceded", "AvgOT": "LAvgOT"})

# Drop duplicate TeamID columns
tourney_results.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

print("✅ Regular Season Performance Merged")

#%%
massey_rankings = datasets["Massey Rankings"]

# Select the last ranking before the tournament (DayNum = 133)
latest_rankings = massey_rankings[massey_rankings["RankingDayNum"] == 133]

# Merge rankings for winning teams
tourney_results = tourney_results.merge(
    latest_rankings, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"OrdinalRank": "WTeamRank"})

# Merge rankings for losing teams
tourney_results = tourney_results.merge(
    latest_rankings, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"OrdinalRank": "LTeamRank"})

# Drop duplicate TeamID columns
tourney_results.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

print("✅ Massey Rankings Merged")

#%%
# Display merged dataset structure
print("\nMerged Dataset Sample:")
print(tourney_results.head())

# Check for missing values after merging
missing_values = tourney_results.isnull().sum().sum()
print(f"\nTotal Missing Values After Merging: {missing_values}")

#%%
missing_summary = tourney_results.isnull().sum()
print(missing_summary[missing_summary > 0])

#%%
tourney_results.drop(columns=["RankingDayNum_x", "SystemName_x", "RankingDayNum_y", "SystemName_y"], inplace=True)

#%%
print("\nTotal Missing Values After Fixes:", tourney_results.isnull().sum().sum())

#%%
missing_summary = tourney_results.isnull().sum()
print(missing_summary[missing_summary > 0])
#%%
# Display rows where WTeamRank or LTeamRank is missing
missing_ranks = tourney_results[tourney_results["WTeamRank"].isnull() | tourney_results["LTeamRank"].isnull()]
print(missing_ranks[["Season", "WTeamID", "WTeamRank", "LTeamID", "LTeamRank"]].head(10))
#%%
# Show unique values in team rankings
print("Unique Winning Team Ranks:", tourney_results["WTeamRank"].dropna().unique()[:15])  # Show first 15 unique values
print("Unique Losing Team Ranks:", tourney_results["LTeamRank"].dropna().unique()[:15])  # Show first 15 unique values

#%%
print("Winning Team Rank Range: ", tourney_results["WTeamRank"].min(), "-", tourney_results["WTeamRank"].max())
print("Losing Team Rank Range: ", tourney_results["LTeamRank"].min(), "-", tourney_results["LTeamRank"].max())

#%%
import matplotlib.pyplot as plt

# Plot histogram for Winning Team Rank
plt.figure(figsize=(12, 5))
plt.hist(tourney_results["WTeamRank"].dropna(), bins=30, alpha=0.7, color="blue", label="WTeamRank")
plt.hist(tourney_results["LTeamRank"].dropna(), bins=30, alpha=0.7, color="red", label="LTeamRank")
plt.xlabel("Team Rank")
plt.ylabel("Frequency")
plt.title("Distribution of Team Rankings")
plt.legend()
plt.show()

#%%
# Fill missing team rankings with 350 to indicate unranked teams
tourney_results["WTeamRank"].fillna(350, inplace=True)
tourney_results["LTeamRank"].fillna(350, inplace=True)

# Verify no missing values remain
print("Total Missing Values After Fix:", tourney_results.isnull().sum().sum())

#%%
import re

# Function to clean seed values
def clean_seed(seed):
    if pd.isna(seed):  # If value is NaN, return NaN
        return None
    seed_num = re.sub("[^0-9]", "", seed)  # Remove non-numeric characters
    return int(seed_num)  # Convert to integer

# Apply function to clean both columns
tourney_results["WSeed"] = tourney_results["WSeed"].apply(clean_seed)
tourney_results["LSeed"] = tourney_results["LSeed"].apply(clean_seed)

# Now compute Seed Difference
tourney_results["SeedDiff"] = tourney_results["WSeed"] - tourney_results["LSeed"]

# Display sample values to confirm fix
print(tourney_results[["WSeed", "LSeed", "SeedDiff"]].head())

#%%
tourney_results["RankDiff"] = tourney_results["WTeamRank"] - tourney_results["LTeamRank"]
#%%
tourney_results["W_OffensiveRating"] = tourney_results["WAvgPointsScored"] - tourney_results["LAvgPointsConceded"]
tourney_results["L_OffensiveRating"] = tourney_results["LAvgPointsScored"] - tourney_results["WAvgPointsConceded"]

tourney_results["W_DefensiveRating"] = tourney_results["WAvgPointsConceded"] - tourney_results["LAvgPointsScored"]
tourney_results["L_DefensiveRating"] = tourney_results["LAvgPointsConceded"] - tourney_results["WAvgPointsScored"]
#%%
tourney_results["WinLabel"] = 1  # Since this dataset contains only winning teams

#%%
# Select final features for modeling
features = [
    "SeedDiff", "RankDiff",
    "W_OffensiveRating", "L_OffensiveRating",
    "W_DefensiveRating", "L_DefensiveRating", "HomeWinBonus"
]

X = tourney_results[features]  # Independent variables
y = tourney_results["WinLabel"]  # Target variable (always 1 in this dataset)

#%%
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the feature matrix
X_scaled = scaler.fit_transform(X)

#%%
import numpy as np
import pandas as pd

# Create a flipped version of the dataset
tourney_results_flipped = tourney_results.copy()

# Swap winning and losing team features
tourney_results_flipped["SeedDiff"] *= -1
tourney_results_flipped["RankDiff"] *= -1
tourney_results_flipped["W_OffensiveRating"], tourney_results_flipped["L_OffensiveRating"] = \
    tourney_results_flipped["L_OffensiveRating"], tourney_results_flipped["W_OffensiveRating"]

tourney_results_flipped["W_DefensiveRating"], tourney_results_flipped["L_DefensiveRating"] = \
    tourney_results_flipped["L_DefensiveRating"], tourney_results_flipped["W_DefensiveRating"]

# Change target variable (flip winners)
tourney_results_flipped["WinLabel"] = 0  # These are now losing team matchups

# Combine original and flipped datasets
final_dataset = pd.concat([tourney_results, tourney_results_flipped], ignore_index=True)

# Shuffle data
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Final Dataset Balanced: {final_dataset['WinLabel'].value_counts()}")

#%%
from sklearn.model_selection import train_test_split

# Select final features again after balancing dataset
X = final_dataset[features]
y = final_dataset["WinLabel"]

# Normalize again
X_scaled = scaler.fit_transform(X)

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"✅ Training Set Size: {X_train.shape[0]}")
print(f"✅ Testing Set Size: {X_test.shape[0]}")

#%%
from sklearn.linear_model import LogisticRegression

# Initialize the model
log_reg = LogisticRegression(random_state=42)

# Train the model on the training data
log_reg.fit(X_train, y_train)

print("✅ Logistic Regression Model Trained Successfully!")

#%%
# Predict the probabilities for the test set
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Probability of WinLabel = 1

# Convert to binary predictions (0 or 1)
y_pred = log_reg.predict(X_test)

#%%
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

#%%
from sklearn.metrics import log_loss

# Calculate log loss (lower is better)
logloss = log_loss(y_test, y_pred_proba)
print(f"✅ Log Loss: {logloss:.4f}")

#%%
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

#%%
import numpy as np

# Get feature importance (Logistic Regression Coefficients)
feature_importance = np.abs(log_reg.coef_[0])  # Use absolute values

# Map feature names to importance
feature_importance_dict = dict(zip(features, feature_importance))

# Sort by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Display feature importance
print("🔍 Feature Importance (Most Impactful Features First):")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

#%%
import os
os.system("pip install xgboost")


#%%
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


#%%
# ✅ Step 1: Initialize XGBoost Classifier
xgb_model = XGBClassifier(
    objective="binary:logistic",  # Binary classification task
    eval_metric="logloss",        # Log loss is better for probability-based evaluation
    use_label_encoder=False,      # Avoids unnecessary warnings
    random_state=42
)

# ✅ Step 2: Train the Model
xgb_model.fit(X_train, y_train)

print("✅ XGBoost Model Trained Successfully!")

#%%
# ✅ Step 3: Make Predictions
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probability of WinLabel = 1
y_pred_xgb = xgb_model.predict(X_test)
#%%
# ✅ Step 4: Evaluate Model Performance
# Accuracy Score
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"✅ XGBoost Model Accuracy: {xgb_accuracy:.4f}")
#%%
# Log Loss Score (lower is better)
xgb_logloss = log_loss(y_test, y_pred_proba_xgb)
print(f"✅ XGBoost Log Loss: {xgb_logloss:.4f}")

#%%
# Confusion Matrix
xgb_conf_matrix = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix (XGBoost):\n", xgb_conf_matrix)
#%%
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Assume 'teams' DataFrame is already loaded with columns "TeamID" and "TeamName"
# Create a mapping of team IDs to team names if needed
team_id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

# Generate matchups
num_rows = 134510
team_ids = teams["TeamID"].unique()
matchups = np.random.choice(team_ids, size=(num_rows, 2), replace=True)

# Ensure no duplicate matchups (no team plays itself)
for i in range(num_rows):
    while matchups[i, 0] == matchups[i, 1]:
        matchups[i, 1] = np.random.choice(team_ids)

# Create DataFrame for matchups
matchup_df = pd.DataFrame(matchups, columns=["TeamA", "TeamB"])

# Generate feature placeholders (update these with your actual feature computations)
matchup_df["SeedDiff"] = np.random.randint(-16, 16, num_rows)
matchup_df["RankDiff"] = np.random.randint(-50, 50, num_rows)
matchup_df["W_OffensiveRating"] = np.random.uniform(50, 100, num_rows)
matchup_df["L_OffensiveRating"] = np.random.uniform(50, 100, num_rows)
matchup_df["W_DefensiveRating"] = np.random.uniform(-50, 50, num_rows)
matchup_df["L_DefensiveRating"] = np.random.uniform(-50, 50, num_rows)
matchup_df["HomeWinBonus"] = np.random.choice([0, 1], num_rows)

# Normalize features using your trained scaler
features = ["SeedDiff", "RankDiff", "W_OffensiveRating", "L_OffensiveRating", "W_DefensiveRating", "L_DefensiveRating", "HomeWinBonus"]
scaler = StandardScaler()
matchup_df_scaled = scaler.fit_transform(matchup_df[features])

# Load trained XGBoost model
xgb_model = joblib.load("Predictions/xgb_best.pkl")  # Ensure this path is correct

# Predict winner probabilities
# Here, predictions represent the probability that TeamA wins.
predictions = xgb_model.predict_proba(matchup_df_scaled)[:, 1]

# For the submission, we'll use a constant season value.
season = 2025  # Change as needed

# Create the submission ID column in the format: Season_TeamA_TeamB
matchup_df["ID"] = matchup_df.apply(lambda row: f"{season}_{int(row['TeamA'])}_{int(row['TeamB'])}", axis=1)

# Create the Pred column: the predicted probability that TeamA wins.
matchup_df["Pred"] = predictions

# Create the submission DataFrame with only the required columns
submission_df = matchup_df[["ID", "Pred"]]

# Save to CSV without the index
submission_df.to_csv("submission.csv", index=False)

print("✅ Submission file saved as 'submission.csv'")

#%%
import os

# Check if the file exists in the current directory
print("Files in current directory:", os.listdir())

# Verify if submission.csv is in the list
print("File Exists:", os.path.exists("submission.csv"))

#%%
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Create a mapping of team IDs to team names from the already loaded `teams` DataFrame
team_id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

# Generate matchups
num_rows = 134510
team_ids = teams["TeamID"].unique()
matchups = np.random.choice(team_ids, size=(num_rows, 2), replace=True)

# Ensure no duplicate matchups (no team plays itself)
for i in range(num_rows):
    while matchups[i, 0] == matchups[i, 1]:
        matchups[i, 1] = np.random.choice(team_ids)

# Create DataFrame for matchups
matchup_df = pd.DataFrame(matchups, columns=["TeamA", "TeamB"])

# Generate feature placeholders (Update if you have real feature computation)
matchup_df["SeedDiff"] = np.random.randint(-16, 16, num_rows)
matchup_df["RankDiff"] = np.random.randint(-50, 50, num_rows)
matchup_df["W_OffensiveRating"] = np.random.uniform(50, 100, num_rows)
matchup_df["L_OffensiveRating"] = np.random.uniform(50, 100, num_rows)
matchup_df["W_DefensiveRating"] = np.random.uniform(-50, 50, num_rows)
matchup_df["L_DefensiveRating"] = np.random.uniform(-50, 50, num_rows)
matchup_df["HomeWinBonus"] = np.random.choice([0, 1], num_rows)

# Normalize features using your trained scaler
features = ["SeedDiff", "RankDiff", "W_OffensiveRating", "L_OffensiveRating", "W_DefensiveRating", "L_DefensiveRating", "HomeWinBonus"]
scaler = StandardScaler()
matchup_df_scaled = scaler.fit_transform(matchup_df[features])

# Load trained XGBoost model
xgb_model = joblib.load("Predictions/xgb_best.pkl")  # Ensure this path is correct

# Predict winner probabilities
predictions = xgb_model.predict_proba(matchup_df_scaled)[:, 1]  # Probability of TeamA winning

# Assign winners and underdogs based on probabilities
matchup_df["Predicted Winner"] = np.where(predictions >= 0.5, matchup_df["TeamA"], matchup_df["TeamB"])
matchup_df["Underdog"] = np.where(predictions < 0.5, matchup_df["TeamA"], matchup_df["TeamB"])

# Map team IDs to team names
matchup_df["Predicted Winner"] = matchup_df["Predicted Winner"].map(team_id_to_name)
matchup_df["Underdog"] = matchup_df["Underdog"].map(team_id_to_name)

# Add win probability to the DataFrame
matchup_df["Win Probability"] = np.where(predictions >= 0.5, predictions, 1 - predictions)  # Assign correct probability

# Save results to CSV
output_df = matchup_df[["Predicted Winner", "Underdog", "Win Probability"]]
output_df.to_csv("predicted_matchups_with_probabilities.csv", index=False)

print("✅ CSV file with matchups and predicted winners saved as 'predicted_matchups_with_probabilities.csv'")

#%%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

# Initialize model
xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)

# Perform grid search
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_log_loss', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use best parameters to train the final model
xgb_best = XGBClassifier(**grid_search.best_params_, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
xgb_best.fit(X_train, y_train)

# Save the optimized model
import joblib
joblib.dump(xgb_best, "xgboost_optimized.pkl")

#%%

# Save the trained model
joblib.dump(xgb_best, "Predictions/xgb_best.pkl")

print("✅ XGBoost model trained and saved as 'xgboost_model.pkl'")
#%%
from sklearn.metrics import accuracy_score, log_loss

# Predict on test data
y_pred_proba = xgb_best.predict_proba(X_test)[:, 1]  # Probability of winning
y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

print(f"✅ Optimized Model Accuracy: {accuracy:.4f}")
print(f"✅ Optimized Model Log Loss: {logloss:.4f}")

#%%
print("Scaler expects:", scaler.n_features_in_)

#%%
import pandas as pd
df = pd.read_csv("predicted_matchups_with_names.csv")
print(df.head())

#%%
import pandas as pd
df = pd.read_csv("predicted_matchups_with_probabilities.csv")
print(df.head())

