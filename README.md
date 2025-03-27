# Predicting March Madness 2025 Outcomes Using Machine Learning

## 1. Introduction
This project aims to predict the outcomes of **March Madness 2025** NCAA basketball games for the men's tournament. Using **historical data** and **machine learning models**, we estimate the probability of a team winning a given matchup. 

Our pipeline leverages **Logistic Regression** and **XGBoost** to generate predictions based on features such as:
- **Seed Difference**
- **Massey Rankings**
- **Offensive & Defensive Ratings**
- **Win/Loss Streaks**
- **Home Court Advantage**

---

## 2. Dataset Overview
We use multiple datasets containing historical NCAA tournament data:

- **Teams (`MTeams.csv`)** – Contains details of NCAA teams.
- **Seasons (`MSeasons.csv`)** – Lists all NCAA seasons.
- **Tournament Seeds (`MNCAATourneySeeds.csv`)** – Lists team seedings per year.
- **Regular Season Results (`MRegularSeasonCompactResults.csv`)** – Includes game-by-game results.
- **Tournament Results (`MNCAATourneyCompactResults.csv`)** – Contains tournament match outcomes.
- **Massey Rankings (`MMasseyOrdinals.csv`)** – External ranking data from various sources.

---

## 3. Data Preprocessing & Feature Engineering

To enhance model performance, we applied the following preprocessing steps:

### 3.1 Handling Missing Values
- Replaced missing **team rankings** with a neutral value (350 for unranked teams).
- Filled missing statistics with league-wide averages.

### 3.2 Feature Engineering
We engineered new features to improve predictive accuracy:
- **Seed Difference** – Difference in seeding between two competing teams.
- **Rank Difference** – Difference in Massey rankings before the tournament.
- **Offensive Rating** – Avg. points scored per game vs. opponent's avg. points allowed.
- **Defensive Rating** – Avg. points conceded per game vs. opponent's avg. points scored.
- **Win/Loss Streaks** – Count of consecutive wins/losses before the tournament.
- **Home Court Advantage** – Boolean flag indicating whether the team played at home.

---

## 4. Model Selection & Training

### 4.1 Baseline Model – Logistic Regression
Our baseline **Logistic Regression** model achieved:
- **Accuracy:** ~70.14%
- **Log Loss:** 0.5598

### 4.2 Final Model – XGBoost
We trained an **XGBoost classifier**, which outperformed Logistic Regression after hyperparameter tuning:

- **Accuracy:** ~66.27%
- **Log Loss:** 0.6373

Hyperparameter tuning was performed using **GridSearchCV** to optimize:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

The optimized XGBoost model was saved as **`xgboost_optimized.pkl`**.

---

## 5. Model Evaluation

### 5.1 Feature Importance
Analyzing **XGBoost feature importance**, we found the most impactful factors:
- **Seed Difference**
- **Defensive Ratings**
- **Offensive Ratings**

![image](https://github.com/user-attachments/assets/1190a0d7-fae8-4f82-a93a-874d0c7e248d)

![image](https://github.com/user-attachments/assets/44f69564-da4a-4c17-9941-8f1498fd2881)


### 5.2 Confusion Matrix
The confusion matrix illustrates correct vs. incorrect classifications:

![image](https://github.com/user-attachments/assets/da8c5419-314f-47f0-9095-e74337365964)


### 5.3 Win Probability Analysis
We generated a **win probability distribution** based on seed differences and historical data.

![image](https://github.com/user-attachments/assets/1837f9fd-316e-4d10-9ab6-c05d3216f904)


---

## 6. Predictions for March Madness 2025

Using our trained **XGBoost model**, we evaluated **134,510 possible matchups** to predict winners and win probabilities.

### Example Predictions:
| **Predicted Winner** | **Underdog** | **Win Probability** |
|----------------------|-------------|---------------------|
| Connecticut         | Longwood    | 99.39%             |
| Iowa State         | WKU         | 99.17%             |
| Illinois           | Oakland     | 98.88%             |

The final predictions are stored in **`predicted_matchups_with_probabilities.csv`**.

---

## 7. Conclusion & Future Improvements

Our **March Madness 2025 predictor** successfully utilizes machine learning to estimate tournament outcomes. 

### Future Improvements:
- **Incorporate Player-Level Data** – Individual player stats could improve accuracy.
- **Dynamic In-Tournament Updates** – Adjust predictions based on live tournament performance.
- **Deep Learning Approaches** – Investigate neural networks for improved pattern recognition.
- **Better Time-Series Modeling** – Utilize **LSTMs/RNNs** to capture momentum trends.

This project demonstrates the power of **machine learning in sports analytics** and provides a **scalable framework** for March Madness predictions.

![image](https://github.com/user-attachments/assets/6610db9a-641a-49d9-8a68-dbd58b20f686)


