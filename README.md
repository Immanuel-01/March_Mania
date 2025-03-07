# Predicting March Madness 2024 Outcomes Using Machine Learning

## 1. Introduction
March Madness is one of the most exciting and unpredictable events in sports, where NCAA basketball teams compete in a knockout-style tournament. The goal of this project is to build a machine learning model to predict the outcomes of March Madness games for both the men's and women's tournaments.

Leveraging historical data and advanced predictive modeling techniques, we designed a robust framework using Logistic Regression and XGBoost to estimate the probability of a team winning a given matchup. Our final model incorporates features like seed differences, offensive and defensive ratings, Massey rankings, and win/loss streaks.

## 2. Dataset Overview
To build an effective prediction model, we utilized multiple datasets from historical NCAA tournament data:
- **Regular Season Results** (`MRegularSeasonCompactResults.csv`, `WRegularSeasonCompactResults.csv`) – Contains game-by-game results from the regular season.
- **NCAA Tournament Results** (`MNCAATourneyCompactResults.csv`, `WNCAATourneyCompactResults.csv`) – Provides tournament game outcomes.
- **Tournament Seeds** (`MNCAATourneySeeds.csv`, `WNCAATourneySeeds.csv`) – Lists the seeding of each team per year.
- **Massey Rankings** (`MMasseyOrdinals.csv`) – Provides ranking data from various external rating systems.

## 3. Data Preprocessing and Feature Engineering
To enhance model performance, we conducted the following preprocessing steps:

### **3.1 Handling Missing Values**
- Replaced missing ranking values with a neutral value (e.g., 350 for unranked teams).
- Filled missing statistics with league-wide averages.

### **3.2 Feature Engineering**
- **Seed Difference**: Difference in seeding between two competing teams.
- **Rank Difference**: Difference in Massey ranking before the tournament.
- **Offensive Rating**: Average points scored per game vs. opponent's average points allowed.
- **Defensive Rating**: Average points conceded per game vs. opponent's average points scored.
- **Win/Loss Streaks**: Count of consecutive wins or losses leading up to the tournament.
- **Home Win Bonus**: Added a feature to indicate if a team played on its home court.

## 4. Model Selection and Training
### **4.1 Baseline Model – Logistic Regression**
We started with a simple **Logistic Regression** model as a baseline, achieving:
- **Accuracy**: ~86%
- **Log Loss**: 0.29

### **4.2 Final Model – XGBoost**
XGBoost, a tree-based ensemble model, was selected for its ability to handle structured data effectively. After hyperparameter tuning, it achieved different results for the men's and women's tournaments:

- **Men's Tournament:**
  - **Accuracy**: ~97.98%
  - **Log Loss**: 0.1564

- **Women's Tournament:**
  - **Accuracy**: ~96.05%
  - **Log Loss**: 0.1029
XGBoost, a tree-based ensemble model, was selected for its ability to handle structured data effectively. After hyperparameter tuning, it achieved:
- **Accuracy**: ~97.98%
- **Log Loss**: 0.1564

## 5. Model Evaluation

### **5.1 Feature Importance**
To determine which features had the most impact on predictions, we analyzed the feature importance scores from the XGBoost model.

![image](https://github.com/user-attachments/assets/5028cc31-21dd-4656-a731-156e93d0d6ba)


### **5.2 Confusion Matrix**
The confusion matrix below visualizes the accuracy of our predictions, showing the number of correct and incorrect classifications.

![image](https://github.com/user-attachments/assets/1d08eb41-ace7-44ad-b73c-782d34b8ef3f)


### **5.3 Win Probability Analysis**
The scatter plot below illustrates how seed differences correlate with win probabilities, indicating that higher-seeded teams generally have a higher chance of winning.

![image](https://github.com/user-attachments/assets/597e0ee4-11b9-41e0-a587-837358f24b41)



We evaluated model performance using:
- **Confusion Matrix**: Showed that XGBoost correctly classified a majority of game outcomes.
- **Feature Importance**: Seed difference and rank difference were the strongest predictors.
- **Win Probability Analysis**: Higher-seeded teams consistently had higher predicted probabilities of winning.

## 6. Predictions for March Madness 2024

To visualize the distribution of predicted win probabilities, we generated the following histogram:



Using the trained XGBoost model, we generated predictions for the upcoming 2024 tournament. The model evaluated **2,278 possible matchups**, ranking teams based on their likelihood of winning.
Using the trained XGBoost model, we generated predictions for the upcoming 2024 tournament. The model evaluated **2,278 possible matchups**, ranking teams based on their likelihood of winning.

**Example Predictions:**
| Predicted Winner  | Underdog         | Win Probability |
|------------------|-----------------|----------------|
| Connecticut      | Longwood        | 99.39%         |
| Iowa State      | WKU             | 99.17%         |
| Illinois        | Oakland         | 98.88%         |

## 7. Conclusion and Future Improvements
This project successfully developed a machine learning pipeline to predict NCAA tournament matchups. The XGBoost model demonstrated high accuracy, making it a valuable tool for March Madness forecasting.

### **Future Improvements**:
1. **Additional Features** – Incorporate player statistics, coaching trends, and game location factors.
2. **Deep Learning** – Explore neural networks for pattern recognition.
3. **Better Time-Series Modeling** – Utilize recurrent neural networks (RNNs) for sequence modeling.

Our final predictions for the 2024 tournament are stored in **MarchMadnessPredictions.csv**, providing a probabilistic ranking of potential matchups.

---

This project demonstrates the power of machine learning in sports analytics and provides a foundation for future enhancements in predictive modeling.

