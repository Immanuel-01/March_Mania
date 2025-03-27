import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
xgb_model = joblib.load("xgb_best.pkl")  # Ensure correct file path

# Load team data for dropdown selection
teams = pd.read_csv("data/MTeams.csv")
team_id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))
team_name_to_id = {v: k for k, v in team_id_to_name.items()}

# Load the trained scaler
scaler = joblib.load("scaler.pkl")  # Save & load your scaler if used during training

# Streamlit UI
st.title("🏀 NCAA Match Predictor")
st.write("Select two teams to predict the winner based on tournament and ranking data.")

# User selects teams
team1 = st.selectbox("Select Team 1", options=list(team_name_to_id.keys()))
team2 = st.selectbox("Select Team 2", options=list(team_name_to_id.keys()))

# Ensure different teams are selected
if team1 == team2:
    st.warning("Please select two different teams!")

if st.button("Predict Winner"):
    # Retrieve team IDs
    team1_id = team_name_to_id[team1]
    team2_id = team_name_to_id[team2]

    # Generate matchup features (fetch real values in production)
    seed_diff = np.random.randint(-16, 16)  # Replace with actual seed difference
    rank_diff = np.random.randint(-50, 50)  # Replace with actual rank difference
    w_off_rating = np.random.uniform(50, 100)  # Replace with actual offensive rating
    l_off_rating = np.random.uniform(50, 100)  # Replace with actual defensive rating
    w_def_rating = np.random.uniform(-50, 50)  # Replace with actual defensive rating
    l_def_rating = np.random.uniform(-50, 50)  # Replace with actual defensive rating
    home_win_bonus = np.random.choice([0, 1])  # Replace with actual home win bonus

    # Prepare feature array
    input_features = np.array(
        [[seed_diff, rank_diff, w_off_rating, l_off_rating, w_def_rating, l_def_rating, home_win_bonus]])

    # Normalize features
    input_scaled = scaler.transform(input_features)

    # Predict probability
    win_probability = xgb_model.predict_proba(input_scaled)[:, 1][0]

    # Determine predicted winner
    predicted_winner = team1 if win_probability >= 0.5 else team2
    underdog = team2 if win_probability >= 0.5 else team1
    win_prob_display = win_probability if win_probability >= 0.5 else 1 - win_probability

    # Display results
    st.success(f"🏆 **Predicted Winner:** {predicted_winner}")
    st.write(f"📉 **Underdog:** {underdog}")
    st.write(f"🔥 **Win Probability:** {win_prob_display:.2%}")
