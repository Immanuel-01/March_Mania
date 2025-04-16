🏀 MarchMania: Predicting NCAA March Madness 2025 Outcomes with Machine Learning
🎯 Project Overview
This project predicts outcomes for the 2025 NCAA Men’s Basketball Tournament using historical data and machine learning. By analyzing team performance, seedings, rankings, and other competitive factors, we estimate the probability of a team winning any given matchup.

🔮 Live App: Try out the interactive predictor powered by Streamlit on [Hugging Face Spaces (update this with your actual link)](https://huggingface.co/spaces/EOEOkunfolami/Predictions)

📊 Features Used
Engineered features include:

SeedDiff: Difference in tournament seed

RankDiff: Massey rankings gap

Offensive & Defensive Ratings: Based on opponent-adjusted season stats

HomeWinBonus: Whether the winning team played at home

(Optionally included) Win/Loss streaks, team averages, OT frequency

🧠 Models Used

Model	Accuracy	Log Loss
Logistic Regression (baseline)	~70.14%	0.5598
XGBoost (optimized)	~66.27%	0.6373
XGBoost was selected as the final model after hyperparameter tuning via GridSearchCV.

🗂️ Dataset Sources
MTeams.csv, MSeasons.csv – Team and season metadata

MNCAATourneySeeds.csv, MNCAATourneyCompactResults.csv – Tournament history

MRegularSeasonCompactResults.csv – Regular season game results

MMasseyOrdinals.csv – Pre-tournament team rankings

📁 All datasets sourced from Kaggle’s NCAA March Madness Database.

🔧 Data Processing & Feature Engineering
Cleaned, merged, and augmented data from multiple sources

Created custom matchup features (SeedDiff, RankDiff, ratings)

Balanced the dataset by flipping winning/losing matchups

Scaled data using StandardScaler

🚀 Deployment
Deployed as an interactive Streamlit app on Hugging Face Spaces

Users select any two teams and instantly see:

Predicted winner

Underdog

Win probability


📈 Example Output

Predicted Winner	Underdog	Win Probability
Connecticut	Longwood	99.39%
Iowa State	WKU	99.17%
Illinois	Oakland	98.88%
Predictions were generated for over 134,000 hypothetical matchups using the trained XGBoost model.

🔍 Feature Importance (XGBoost)
SeedDiff

W_DefensiveRating

L_DefensiveRating

RankDiff

OffensiveRatings

📌 Future Enhancements
Integrate player-level stats (injuries, recent form)

Add live tournament updates during March Madness

Explore deep learning models (LSTM, attention)

Improve probability calibration for betting use-cases

📁 Files
submission.csv – Tournament prediction format for Kaggle

predicted_matchups_with_probabilities.csv – Final probabilities

xgb_best.pkl – Trained XGBoost model

scaler.pkl – StandardScaler used for input normalization

app.py – Streamlit frontend code

🤝 Acknowledgements
NCAA Data via Kaggle

Streamlit & Hugging Face for hosting

XGBoost for model power 💥
