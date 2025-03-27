```python
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

```


```python
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

```


```python
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

```

    
    Teams Dataset Sample:
       TeamID     TeamName
    0    3101  Abilene Chr
    1    3102    Air Force
    2    3103        Akron
    3    3104      Alabama
    4    3105  Alabama A&M 
    
    
    Seasons Dataset Sample:
       Season     DayZero RegionW  RegionX  RegionY  RegionZ
    0    1998  10/27/1997    East  Midwest  Mideast     West
    1    1999  10/26/1998    East  Mideast  Midwest     West
    2    2000  11/01/1999    East  Midwest  Mideast     West
    3    2001  10/30/2000    East  Midwest  Mideast     West
    4    2002  10/29/2001    East     West  Mideast  Midwest 
    
    
    Tourney Seeds Dataset Sample:
       Season Seed  TeamID
    0    1998  W01    3330
    1    1998  W02    3163
    2    1998  W03    3112
    3    1998  W04    3301
    4    1998  W05    3272 
    
    
    Regular Season Results Dataset Sample:
       Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT
    0    1998      18     3104      91     3202      41    H      0
    1    1998      18     3163      87     3221      76    H      0
    2    1998      18     3222      66     3261      59    H      0
    3    1998      18     3307      69     3365      62    H      0
    4    1998      18     3349     115     3411      35    H      0 
    
    
    Tourney Results Dataset Sample:
       Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT
    0    1998     137     3104      94     3422      46    H      0
    1    1998     137     3112      75     3365      63    H      0
    2    1998     137     3163      93     3193      52    H      0
    3    1998     137     3198      59     3266      45    H      0
    4    1998     137     3203      74     3208      72    A      0 
    
    
    Regular Season Detailed Dataset Sample:
       Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT  WFGM  WFGA  \
    0    2010      11     3103      63     3237      49    H      0    23    54   
    1    2010      11     3104      73     3399      68    N      0    26    62   
    2    2010      11     3110      71     3224      59    A      0    29    62   
    3    2010      11     3111      63     3267      58    A      0    27    52   
    4    2010      11     3119      74     3447      70    H      1    30    74   
    
       ...  LFGA3  LFTM  LFTA  LOR  LDR  LAst  LTO  LStl  LBlk  LPF  
    0  ...     13     6    10   11   27    11   23     7     6   19  
    1  ...     21    14    27   14   26     7   20     4     2   27  
    2  ...     14    19    23   17   23     8   15     6     0   15  
    3  ...     26    16    25   22   22    15   11    14     5   14  
    4  ...     17    11    21   21   32    12   14     4     2   14  
    
    [5 rows x 34 columns] 
    
    
    Tourney Detailed Dataset Sample:
       Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT  WFGM  WFGA  \
    0    2010     138     3124      69     3201      55    N      0    28    57   
    1    2010     138     3173      67     3395      66    N      0    23    59   
    2    2010     138     3181      72     3214      37    H      0    26    57   
    3    2010     138     3199      75     3256      61    H      0    25    63   
    4    2010     138     3207      62     3265      42    N      0    24    68   
    
       ...  LFGA3  LFTM  LFTA  LOR  LDR  LAst  LTO  LStl  LBlk  LPF  
    0  ...     34     3     5   17   19    12   18     4     1   18  
    1  ...     27    14    15   18   26     8    8     8     6   22  
    2  ...     15     3     8   10   21     4   16     6     4   20  
    3  ...     20    17    22   16   21    13   16     5     4   24  
    4  ...     26    11    17   16   22     9   10     3     4   12  
    
    [5 rows x 34 columns] 
    
    


```python
tourney_results["HomeWinBonus"] = (tourney_results["WLoc"] == "H").astype(int)

```


```python
tourney_results = tourney_results.merge(
    tourney_seeds, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"Seed": "WSeed"})

tourney_results = tourney_results.merge(
    tourney_seeds, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
).rename(columns={"Seed": "LSeed"})

# Drop duplicate TeamID columns
tourney_results.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

print("✅ Tournament Seeds Merged")

```

    ✅ Tournament Seeds Merged
    


```python
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

```

    ✅ Regular Season Performance Merged
    


```python
# Convert seed values from string to numeric
tourney_results["WSeed"] = tourney_results["WSeed"].astype(str).str.extract("(\d+)").astype(float)
tourney_results["LSeed"] = tourney_results["LSeed"].astype(str).str.extract("(\d+)").astype(float)

# Compute Seed Difference
tourney_results["SeedDiff"] = tourney_results["WSeed"] - tourney_results["LSeed"]

```

    <>:2: SyntaxWarning: invalid escape sequence '\d'
    <>:3: SyntaxWarning: invalid escape sequence '\d'
    <>:2: SyntaxWarning: invalid escape sequence '\d'
    <>:3: SyntaxWarning: invalid escape sequence '\d'
    C:\Users\Emmanuel Okunfolami\AppData\Local\Temp\ipykernel_9508\803557826.py:2: SyntaxWarning: invalid escape sequence '\d'
      tourney_results["WSeed"] = tourney_results["WSeed"].astype(str).str.extract("(\d+)").astype(float)
    C:\Users\Emmanuel Okunfolami\AppData\Local\Temp\ipykernel_9508\803557826.py:3: SyntaxWarning: invalid escape sequence '\d'
      tourney_results["LSeed"] = tourney_results["LSeed"].astype(str).str.extract("(\d+)").astype(float)
    


```python
tourney_results["W_OffensiveRating"] = tourney_results["WAvgPointsScored"] - tourney_results["LAvgPointsConceded"]
tourney_results["L_OffensiveRating"] = tourney_results["LAvgPointsScored"] - tourney_results["WAvgPointsConceded"]

tourney_results["W_DefensiveRating"] = tourney_results["WAvgPointsConceded"] - tourney_results["LAvgPointsScored"]
tourney_results["L_DefensiveRating"] = tourney_results["LAvgPointsConceded"] - tourney_results["WAvgPointsScored"]

```


```python
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

```

    ✅ Class Distribution: 
    WinLabel
    0.0    1650
    Name: count, dtype: int64
    


```python
print("Missing values in y:", y.isnull().sum())

```

    Missing values in y: 0
    


```python
print("Missing values in y after cleaning:", y.isnull().sum())

```

    Missing values in y after cleaning: 0
    


```python
print("Class distribution in y_train:\n", y_train.value_counts())

```

    Class distribution in y_train:
     WinLabel
    1    508655
    0    508359
    Name: count, dtype: int64
    


```python
print("Class distribution in full dataset:\n", final_dataset["WinLabel"].value_counts())

```

    Class distribution in full dataset:
     WinLabel
    0.0    1650
    Name: count, dtype: int64
    


```python
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

```

    ✅ Features computed before balancing!
    

    C:\Users\Emmanuel Okunfolami\AppData\Local\Temp\ipykernel_9508\1574382924.py:20: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      final_dataset["WWinStreak"].fillna(0, inplace=True)
    C:\Users\Emmanuel Okunfolami\AppData\Local\Temp\ipykernel_9508\1574382924.py:21: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      final_dataset["LLoseStreak"].fillna(0, inplace=True)
    C:\Users\Emmanuel Okunfolami\AppData\Local\Temp\ipykernel_9508\1574382924.py:22: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      final_dataset["PointDiff"].fillna(0, inplace=True)
    


```python
print("Missing values in y:", y.isnull().sum())

```

    Missing values in y: 0
    


```python
# Display a few rows where `WinLabel` is missing
missing_rows = final_dataset[final_dataset["WinLabel"].isnull()]
print("Rows with missing WinLabel:\n", missing_rows.head())

# Count missing values in the entire dataset
print("Missing values per column:\n", final_dataset.isnull().sum())

```

    Rows with missing WinLabel:
        Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT  HomeWinBonus  \
    0    1998     145     3301      55     3330      54    A      0             0   
    1    1998     145     3301      55     3330      54    A      0             0   
    2    1998     145     3301      55     3330      54    A      0             0   
    3    1998     145     3301      55     3330      54    A      0             0   
    4    1998     145     3301      55     3330      54    A      0             0   
    
       WSeed  ...  LAvgOT  SeedDiff  W_OffensiveRating  L_OffensiveRating  \
    0    4.0  ...     0.0       3.0          19.704762          23.110476   
    1    4.0  ...     0.0       3.0          19.704762          23.110476   
    2    4.0  ...     0.0       3.0          19.704762          23.110476   
    3    4.0  ...     0.0       3.0          19.704762          23.110476   
    4    4.0  ...     0.0       3.0          19.704762          23.110476   
    
       W_DefensiveRating  L_DefensiveRating  WinLabel  WWinStreak  LLoseStreak  \
    0         -23.110476         -19.704762       NaN           1          1.0   
    1         -23.110476         -19.704762       NaN           1          2.0   
    2         -23.110476         -19.704762       NaN           2          1.0   
    3         -23.110476         -19.704762       NaN           2          2.0   
    4         -23.110476         -19.704762       NaN           3          1.0   
    
       PointDiff  
    0  -6.015238  
    1  -6.015238  
    2  -6.015238  
    3  -6.015238  
    4  -6.015238  
    
    [5 rows x 26 columns]
    Missing values per column:
     Season                     0
    DayNum                     0
    WTeamID                    0
    WScore                     0
    LTeamID                    0
    LScore                     0
    WLoc                       0
    NumOT                      0
    HomeWinBonus               0
    WSeed                      0
    LSeed                      0
    WAvgPointsScored           0
    WAvgPointsConceded         0
    WAvgOT                     0
    LAvgPointsScored           0
    LAvgPointsConceded         0
    LAvgOT                     0
    SeedDiff                   0
    W_OffensiveRating          0
    L_OffensiveRating          0
    W_DefensiveRating          0
    L_DefensiveRating          0
    WinLabel              317817
    WWinStreak                 0
    LLoseStreak                0
    PointDiff                  0
    dtype: int64
    


```python
# If WTeamID exists, set WinLabel = 1 (Winning Team), otherwise 0
final_dataset["WinLabel"] = final_dataset["WTeamID"].notnull().astype(int)

print("✅ Recomputed `WinLabel` successfully!")

```

    ✅ Recomputed `WinLabel` successfully!
    


```python
print("Missing values in WinLabel after fixing:", final_dataset["WinLabel"].isnull().sum())

```

    Missing values in WinLabel after fixing: 0
    


```python
print("Available columns in final_dataset:", final_dataset.columns.tolist())

```

    Available columns in final_dataset: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'HomeWinBonus', 'WSeed', 'LSeed', 'WAvgPointsScored', 'WAvgPointsConceded', 'WAvgOT', 'LAvgPointsScored', 'LAvgPointsConceded', 'LAvgOT', 'SeedDiff', 'W_OffensiveRating', 'L_OffensiveRating', 'W_DefensiveRating', 'L_DefensiveRating', 'WinLabel', 'WWinStreak', 'LLoseStreak', 'PointDiff']
    


```python
# Print class distribution
print("Class distribution in WinLabel:\n", final_dataset["WinLabel"].value_counts())

```

    Class distribution in WinLabel:
     WinLabel
    1    635634
    Name: count, dtype: int64
    


```python
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

```

    ✅ Class distribution after balancing:
     WinLabel
    0    635634
    1    635634
    Name: count, dtype: int64
    


```python
# Print column names from the original DataFrame before scaling
print("Training features:", list(X.columns))  # Use X, not X_train
print("New matchups features:", list(X_new.columns))  # X_new should be a DataFrame

```

    Training features: ['SeedDiff', 'W_OffensiveRating', 'L_OffensiveRating', 'W_DefensiveRating', 'L_DefensiveRating', 'HomeWinBonus', 'WWinStreak', 'LLoseStreak', 'PointDiff']
    New matchups features: ['SeedDiff', 'W_OffensiveRating', 'L_OffensiveRating', 'W_DefensiveRating', 'L_DefensiveRating', 'HomeWinBonus', 'WWinStreak', 'LLoseStreak', 'PointDiff']
    


```python
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

```




<style>#sk-container-id-9 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-9 {
  color: var(--sklearn-color-text);
}

#sk-container-id-9 pre {
  padding: 0;
}

#sk-container-id-9 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-9 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-9 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-9 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-9 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-9 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-9 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-9 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-9 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-9 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-9 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-9 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-9 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-9 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-9 div.sk-label label.sk-toggleable__label,
#sk-container-id-9 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-9 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-9 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-9 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-9 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-9 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-9 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-9 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-9 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-9 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=&#x27;logloss&#x27;,
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=42, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" checked><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>XGBClassifier</div></div><div><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=&#x27;logloss&#x27;,
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=42, ...)</pre></div> </div></div></div></div>




```python
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

```

    ✅ Logistic Regression Accuracy: 0.8617
    ✅ Logistic Regression Log Loss: 0.2916
    ✅ XGBoost Accuracy: 0.9605
    ✅ XGBoost Log Loss: 0.1029
    
    Confusion Matrix (XGBoost):
     [[122473   4802]
     [  5251 121728]]

```python
import joblib

# Save the Logistic Regression model
joblib.dump(log_reg, "logistic_regression_model.pkl")

# Save the XGBoost model
joblib.dump(xgb_model, "xgboost_model.pkl")

# Save the Scaler
joblib.dump(scaler, "Predictions/scaler.pkl")

print("✅ Models and scaler saved successfully!")

```

    ✅ Models and scaler saved successfully!
    


```python
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

```

    Fitting 3 folds for each of 64 candidates, totalling 192 fits
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[317], line 15
         12 xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
         14 grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
    ---> 15 grid_search.fit(X_train, y_train)
         17 print("✅ Best Parameters:", grid_search.best_params_)
         19 xgb_best = XGBClassifier(**grid_search.best_params_, objective="binary:logistic", eval_metric="logloss",
         20                          random_state=42)
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\sklearn\base.py:1389, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1382     estimator._validate_params()
       1384 with config_context(
       1385     skip_parameter_validation=(
       1386         prefer_skip_nested_validation or global_skip_validation
       1387     )
       1388 ):
    -> 1389     return fit_method(estimator, *args, **kwargs)
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\sklearn\model_selection\_search.py:1024, in BaseSearchCV.fit(self, X, y, **params)
       1018     results = self._format_results(
       1019         all_candidate_params, n_splits, all_out, all_more_results
       1020     )
       1022     return results
    -> 1024 self._run_search(evaluate_candidates)
       1026 # multimetric is determined here because in the case of a callable
       1027 # self.scoring the return type is only known after calling
       1028 first_test_score = all_out[0]["test_scores"]
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\sklearn\model_selection\_search.py:1571, in GridSearchCV._run_search(self, evaluate_candidates)
       1569 def _run_search(self, evaluate_candidates):
       1570     """Search all candidates in param_grid"""
    -> 1571     evaluate_candidates(ParameterGrid(self.param_grid))
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\sklearn\model_selection\_search.py:970, in BaseSearchCV.fit.<locals>.evaluate_candidates(candidate_params, cv, more_results)
        962 if self.verbose > 0:
        963     print(
        964         "Fitting {0} folds for each of {1} candidates,"
        965         " totalling {2} fits".format(
        966             n_splits, n_candidates, n_candidates * n_splits
        967         )
        968     )
    --> 970 out = parallel(
        971     delayed(_fit_and_score)(
        972         clone(base_estimator),
        973         X,
        974         y,
        975         train=train,
        976         test=test,
        977         parameters=parameters,
        978         split_progress=(split_idx, n_splits),
        979         candidate_progress=(cand_idx, n_candidates),
        980         **fit_and_score_kwargs,
        981     )
        982     for (cand_idx, parameters), (split_idx, (train, test)) in product(
        983         enumerate(candidate_params),
        984         enumerate(cv.split(X, y, **routed_params.splitter.split)),
        985     )
        986 )
        988 if len(out) < 1:
        989     raise ValueError(
        990         "No fits were performed. "
        991         "Was the CV iterator empty? "
        992         "Were there no candidates?"
        993     )
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\sklearn\utils\parallel.py:77, in Parallel.__call__(self, iterable)
         72 config = get_config()
         73 iterable_with_config = (
         74     (_with_config(delayed_func, config), args, kwargs)
         75     for delayed_func, args, kwargs in iterable
         76 )
    ---> 77 return super().__call__(iterable_with_config)
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\joblib\parallel.py:2007, in Parallel.__call__(self, iterable)
       2001 # The first item from the output is blank, but it makes the interpreter
       2002 # progress until it enters the Try/Except block of the generator and
       2003 # reaches the first `yield` statement. This starts the asynchronous
       2004 # dispatch of the tasks to the workers.
       2005 next(output)
    -> 2007 return output if self.return_generator else list(output)
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\joblib\parallel.py:1650, in Parallel._get_outputs(self, iterator, pre_dispatch)
       1647     yield
       1649     with self._backend.retrieval_context():
    -> 1650         yield from self._retrieve()
       1652 except GeneratorExit:
       1653     # The generator has been garbage collected before being fully
       1654     # consumed. This aborts the remaining tasks if possible and warn
       1655     # the user if necessary.
       1656     self._exception = True
    

    File ~\PycharmProjects\March Mania\.venv\Lib\site-packages\joblib\parallel.py:1762, in Parallel._retrieve(self)
       1757 # If the next job is not ready for retrieval yet, we just wait for
       1758 # async callbacks to progress.
       1759 if ((len(self._jobs) == 0) or
       1760     (self._jobs[0].get_status(
       1761         timeout=self.timeout) == TASK_PENDING)):
    -> 1762     time.sleep(0.01)
       1763     continue
       1765 # We need to be careful: the job list can be filling up as
       1766 # we empty it and Python list are not thread-safe by
       1767 # default hence the use of the lock
    

    KeyboardInterrupt: 

