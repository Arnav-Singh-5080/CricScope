import streamlit as st
import pickle
import pandas as pd

# Load pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("🏏 IPL Match Win Predictor")

teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians',
    'Rajasthan Royals', 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

cities = [
    'Mumbai', 'Chennai', 'Kolkata', 'Delhi',
    'Bangalore', 'Hyderabad', 'Jaipur'
]

# Inputs
batting_team = st.selectbox('Batting Team', teams)
bowling_team = st.selectbox('Bowling Team', teams)
city = st.selectbox('Match City', cities)

target = st.number_input('Target')
score = st.number_input('Current Score')
overs = st.number_input('Overs Completed')
wickets = st.number_input('Wickets Fallen')

# Derived features
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_remaining = 10 - wickets
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# Prediction
if st.button('Predict Probability'):
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    st.header(batting_team + " Win Probability: " + str(round(win * 100)) + "%")
    st.header(bowling_team + " Win Probability: " + str(round(loss * 100)) + "%")