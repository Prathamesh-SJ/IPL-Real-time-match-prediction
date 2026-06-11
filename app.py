import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set up page styling and layout
st.set_page_config(
    page_title="IPL Live Match Predictor", page_icon="🏏", layout="centered"
)

st.title("🏏 IPL Win Probability Predictor")
st.markdown(
    "Predict the second innings outcome of an IPL match dynamically using machine learning."
)
st.write("---")

# 1. Load the trained pipeline safely
MODEL_PATH = "ipl_win_predictor.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"**Model file not found!** Please ensure you have trained your model and saved it as `{MODEL_PATH}` in the same directory."
    )
    st.stop()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


pipe = load_model()

# 2. Define Options (Matching your cleaned training configuration)
teams = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

cities = [
    "Mumbai",
    "Chennai",
    "Bengaluru",
    "Kolkata",
    "Hyderabad",
    "Delhi",
    "Chandigarh",
    "Jaipur",
    "Ahmedabad",
    "Lucknow",
    "Pune",
    "Dubai",
    "Abu Dhabi",
    "Sharjah",
]

# 3. User Layout Interface Components
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting Team (Chasing)", sorted(teams))

with col2:
    # Filter out the selected batting team so a team cannot play against itself
    remaining_teams = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox("Select Bowling Team (Defending)", sorted(remaining_teams))

selected_city = st.selectbox("Select Host City / Venue", sorted(cities))

st.write("---")
st.subheader("Live Match Context")

col3, col4, col5 = st.columns(3)

with col3:
    target_runs = st.number_input(
        "Target Score", min_value=1, max_value=350, value=180, step=1
    )

with col4:
    current_score = st.number_input(
        "Current Score", min_value=0, max_value=350, value=100, step=1
    )

with col5:
    wickets_lost = st.slider("Wickets Down", min_value=0, max_value=9, value=3)

# Ball tracking configurations
col6, col7 = st.columns(2)

with col6:
    overs_completed = st.number_input(
        "Overs Completed", min_value=0.1, max_value=19.5, value=10.0, step=0.1
    )
    # Convert traditional overs system (e.g., 10.3 overs) into raw ball count
    completed_overs_int = int(overs_completed)
    extra_balls = int(round((overs_completed - completed_overs_int) * 10))
    total_balls_bowled = (completed_overs_int * 6) + extra_balls

with col7:
    # Set fallback parameters to make sure text UI is always intuitive
    balls_remaining = 120 - total_balls_bowled
    st.info(f"Balls Remaining: **{int(balls_remaining)}**")

# 4. Calculation and Match Logic Engine
if st.button("Predict Winning Probabilities", type="primary"):

    # Derived Feature Calculations
    runs_remaining = target_runs - current_score
    wickets_remaining = 10 - wickets_lost

    # Guardrails against invalid or chemically broken match parameters
    if current_score >= target_runs:
        st.success(f"🎉 **{batting_team}** has already won the match!")
    elif total_balls_bowled >= 120:
        st.error("⏳ Match Over! Maximum delivery limits reached (20 Overs).")
    elif total_balls_bowled <= 0:
        st.warning("Please enter a valid over count above 0 to calculate run rates.")
    else:
        # Generate the standard features required by our model pipeline
        current_run_rate = (current_score * 6) / total_balls_bowled
        required_run_rate = (runs_remaining * 6) / balls_remaining
        rr_difference = required_run_rate - current_run_rate
        runs_per_wicket = runs_remaining / wickets_remaining

        # Assemble individual variables into a payload DataFrame
        input_data = pd.DataFrame(
            [
                {
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "city": selected_city,
                    "runs_remaining": runs_remaining,
                    "balls_remaining": balls_remaining,
                    "wickets_remaining": wickets_remaining,
                    "current_run_rate": current_run_rate,
                    "required_run_rate": required_run_rate,
                    "rr_difference": rr_difference,
                    "runs_per_wicket": runs_per_wicket,
                }
            ]
        )

        # 5. Extract Probabilities out of the Model Pipeline
        result_probabilities = pipe.predict_proba(input_data)

        loss_percentage = result_probabilities[0][0] * 100
        win_percentage = result_probabilities[0][1] * 100

        # Display Metrics Header Dashboard
        st.write("---")
        st.subheader("Match Projections Summary")

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric(label="Runs Needed", value=f"{runs_remaining}")
        stat_col2.metric(label="Required Run Rate", value=f"{required_run_rate:.2f}")
        stat_col3.metric(label="Current Run Rate", value=f"{current_run_rate:.2f}")

        # 6. Visual Output Distribution
        st.subheader("Win / Loss Probabilities")

        st.markdown(f"**{batting_team} (Chasing):** {win_percentage:.1f}%")
        st.progress(int(win_percentage))

        st.markdown(f"**{bowling_team} (Defending):** {loss_percentage:.1f}%")
        st.progress(int(loss_percentage))
