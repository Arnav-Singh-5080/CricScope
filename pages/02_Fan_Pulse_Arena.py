import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="Fan Pulse Arena", page_icon="🔥", layout="wide")

# Custom CSS for stadium night theme
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        font-size: 24px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .mood-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(251,191,36,0.2);
        margin: 10px 0;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f59e0b20, #fbbf2420);
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .feature-card {
        background: rgba(15,23,42,0.5);
        border: 1px solid rgba(251,191,36,0.15);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Fan Pulse Arena")
st.markdown("*Real-time community mood, polls, and AI-powered sentiment prediction*")

# ============================================
# SECTION 1: INITIALIZE SESSION STATES
# ============================================
if 'mood_votes' not in st.session_state:
    st.session_state.mood_votes = {'😡': 0, '😐': 0, '🔥': 0}
if 'poll_votes' not in st.session_state:
    st.session_state.poll_votes = {}
if 'current_poll' not in st.session_state:
    st.session_state.current_poll = "Best Powerplay Opener"
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'ml_predictions' not in st.session_state:
    st.session_state.ml_predictions = []

# ============================================
# SECTION 2: XGBOOST MODEL FUNCTIONS
# ============================================
@st.cache_data
def generate_training_data():
    """Generate synthetic cricket mood data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'over_number': np.random.randint(1, 21, n_samples),
        'runs_in_over': np.random.randint(0, 36, n_samples),
        'wickets_in_over': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
        'run_rate': np.random.uniform(4, 12, n_samples),
        'required_rate': np.random.uniform(5, 15, n_samples),
        'time_of_match': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
        'match_importance': np.random.choice(['league', 'playoff', 'final'], n_samples, p=[0.6, 0.3, 0.1]),
    }
    
    moods = []
    for i in range(n_samples):
        if data['wickets_in_over'][i] >= 2:
            moods.append('angry')
        elif data['runs_in_over'][i] > 15 and data['wickets_in_over'][i] == 0:
            moods.append('excited')
        elif data['run_rate'][i] > data['required_rate'][i]:
            moods.append('excited')
        elif data['wickets_in_over'][i] == 1:
            moods.append('neutral')
        else:
            moods.append(np.random.choice(['angry', 'neutral', 'excited'], p=[0.2, 0.5, 0.3]))
    
    data['mood'] = moods
    return pd.DataFrame(data)

def train_mood_predictor(df):
    """Train XGBoost model to predict fan mood"""
    with st.spinner("🧠 Training AI model on fan sentiment data..."):
        le_time = LabelEncoder()
        le_importance = LabelEncoder()
        le_mood = LabelEncoder()
        
        df['time_encoded'] = le_time.fit_transform(df['time_of_match'])
        df['importance_encoded'] = le_importance.fit_transform(df['match_importance'])
        df['mood_encoded'] = le_mood.fit_transform(df['mood'])
        
        features = ['over_number', 'runs_in_over', 'wickets_in_over', 
                   'run_rate', 'required_rate', 'time_encoded', 'importance_encoded']
        X = df[features]
        y = df['mood_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        joblib.dump(model, 'mood_predictor.pkl')
        joblib.dump(le_time, 'le_time.pkl')
        joblib.dump(le_importance, 'le_importance.pkl')
        joblib.dump(le_mood, 'le_mood.pkl')
        
        return model, le_time, le_importance, le_mood, accuracy

def predict_current_mood(over, runs, wickets, run_rate, req_rate, time, importance, encoders):
    """Use trained model to predict fan sentiment"""
    model, le_time, le_importance, le_mood = encoders
    
    input_data = pd.DataFrame([{
        'over_number': over,
        'runs_in_over': runs,
        'wickets_in_over': wickets,
        'run_rate': run_rate,
        'required_rate': req_rate,
        'time_encoded': le_time.transform([time])[0],
        'importance_encoded': le_importance.transform([importance])[0]
    }])
    
    pred_encoded = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]
    
    mood = le_mood.inverse_transform([pred_encoded])[0]
    confidence = max(pred_proba) * 100
    
    return mood, confidence

# ============================================
# SECTION 3: MAIN LAYOUT - 3 COLUMNS
# ============================================

# Create 3 columns for the three features
col1, col2, col3 = st.columns(3, gap="large")

# ========== COLUMN 1: MOOD METER ==========
with col1:
    st.markdown("""
        <div class="feature-card">
            <div style="font-size: 14px; letter-spacing: 3px; color: rgba(251,191,36,0.8); margin-bottom: 15px;">
                📊 LIVE MOOD METER
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**How are you feeling?**")
    
    # Mood buttons
    mood_b1, mood_b2, mood_b3 = st.columns(3)
with mood_b1:
    if st.button("😡", key="mood_a", use_container_width=True):
        st.session_state.mood_votes['😡'] += 1
        st.toast("😡 ANGRY vote recorded!", icon="😡")
        st.rerun()
with mood_b2:
    if st.button("😐", key="mood_n", use_container_width=True):
        st.session_state.mood_votes['😐'] += 1
        st.toast("😐 NEUTRAL vote recorded!", icon="😐")
        st.rerun()
with mood_b3:
    if st.button("🔥", key="mood_e", use_container_width=True):
        st.session_state.mood_votes['🔥'] += 1
        st.toast("🔥 EXCITED vote recorded!", icon="🔥")
        st.rerun()
    
    # Display mood stats
    total_votes = sum(st.session_state.mood_votes.values())
    if total_votes > 0:
        for mood, emoji in [('😡', 'ANGRY'), ('😐', 'NEUTRAL'), ('🔥', 'EXCITED')]:
            count = st.session_state.mood_votes[mood]
            pct = (count / total_votes) * 100
            st.markdown(f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{emoji}</span>
                        <span style="color: #fbbf24;">{pct:.0f}%</span>
                    </div>
                    <div style="background: #334155; border-radius: 10px; height: 25px; overflow: hidden;">
                        <div style="width: {pct}%; background: linear-gradient(90deg, #f59e0b, #fbbf24); height: 100%;"></div>
                    </div>
                    <div style="font-size: 11px; color: #94a3b8;">{count} votes</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Dominant mood
        dominant = max(st.session_state.mood_votes, key=st.session_state.mood_votes.get)
        st.info(f"💡 Crowd is feeling {dominant}")
    else:
        st.info("Click a mood to start!")
    
    st.metric("Total Votes", total_votes)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== COLUMN 2: ROLE POLLS ==========
with col2:
    st.markdown("""
        <div class="feature-card">
            <div style="font-size: 14px; letter-spacing: 3px; color: rgba(251,191,36,0.8); margin-bottom: 15px;">
                🏆 BEST ROLE POLLS
            </div>
    """, unsafe_allow_html=True)
    
    poll_type = st.selectbox(
        "Select Category",
        ["Best Powerplay Opener", "Most Clutch Finisher", "Best Death Bowler", "MVP of the Tournament"],
        key="poll_type"
    )
    
    player_options = {
        "Best Powerplay Opener": ["Rohit Sharma", "Jos Buttler", "Quinton de Kock", "David Warner"],
        "Most Clutch Finisher": ["MS Dhoni", "Hardik Pandya", "Andre Russell", "Kieron Pollard"],
        "Best Death Bowler": ["Jasprit Bumrah", "Kagiso Rabada", "Trent Boult", "Pat Cummins"],
        "MVP of the Tournament": ["Virat Kohli", "Rashid Khan", "Jos Buttler", "Jasprit Bumrah"]
    }
    
    players = player_options.get(poll_type, ["Player 1", "Player 2", "Player 3", "Player 4"])
    
    if st.session_state.current_poll != poll_type:
        st.session_state.current_poll = poll_type
        st.session_state.poll_votes = {player: 0 for player in players}
    
    st.markdown("**Cast your vote**")
    
    # 2-column layout for vote buttons
    vc1, vc2 = st.columns(2)
    for idx, player in enumerate(players):
        with vc1 if idx % 2 == 0 else vc2:
            if st.button(f"🗳️ {player}", key=f"poll_{player}", use_container_width=True):
                st.session_state.poll_votes[player] = st.session_state.poll_votes.get(player, 0) + 1
                st.success(f"Voted for {player}!")
                st.rerun()
    
    # Results
    st.markdown("**📊 Results**")
    if st.session_state.poll_votes and sum(st.session_state.poll_votes.values()) > 0:
        df_votes = pd.DataFrame({
            'Player': list(st.session_state.poll_votes.keys()),
            'Votes': list(st.session_state.poll_votes.values())
        }).sort_values('Votes', ascending=True)
        st.bar_chart(df_votes.set_index('Player'))
        
        winner = max(st.session_state.poll_votes, key=st.session_state.poll_votes.get)
        st.success(f"🏆 Leader: {winner}")
    else:
        st.info("No votes yet")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== COLUMN 3: ML SENTIMENT PREDICTION ==========
with col3:
    st.markdown("""
        <div class="feature-card">
            <div style="font-size: 14px; letter-spacing: 3px; color: rgba(251,191,36,0.8); margin-bottom: 15px;">
                🤖 AI SENTIMENT PREDICTION
            </div>
    """, unsafe_allow_html=True)
    
    # Match context inputs
    over = st.slider("Current Over", 1, 20, 10, key="ml_over")
    runs_in_over = st.number_input("Runs this over", 0, 36, 8, key="ml_runs")
    wickets = st.selectbox("Wickets this over", [0, 1, 2, 3], key="ml_wickets")
    run_rate = st.slider("Current Run Rate", 4.0, 12.0, 7.5, 0.1, key="ml_crr")
    req_rate = st.slider("Required Run Rate", 5.0, 15.0, 9.0, 0.1, key="ml_rrr")
    time_of_day = st.selectbox("Time", ["morning", "afternoon", "evening", "night"], key="ml_time")
    match_imp = st.selectbox("Importance", ["league", "playoff", "final"], key="ml_imp")
    
    # Load or train model
    model_files_exist = all(os.path.exists(f) for f in ['mood_predictor.pkl', 'le_time.pkl', 'le_importance.pkl', 'le_mood.pkl'])
    
    if not model_files_exist or not st.session_state.model_trained:
        with st.spinner("Training XGBoost model..."):
            df = generate_training_data()
            model, le_time, le_importance, le_mood, accuracy = train_mood_predictor(df)
            st.session_state.model_trained = True
            st.session_state.ml_accuracy = accuracy
            st.session_state.encoders = (model, le_time, le_importance, le_mood)
    else:
        model = joblib.load('mood_predictor.pkl')
        le_time = joblib.load('le_time.pkl')
        le_importance = joblib.load('le_importance.pkl')
        le_mood = joblib.load('le_mood.pkl')
        st.session_state.encoders = (model, le_time, le_importance, le_mood)
    
    # Prediction button
    if st.button("🔮 Predict Mood", use_container_width=True, type="primary"):
        mood, confidence = predict_current_mood(
            over, runs_in_over, wickets, run_rate, req_rate,
            time_of_day, match_imp, st.session_state.encoders
        )
        
        mood_emoji = {'angry': '😡', 'neutral': '😐', 'excited': '🔥'}.get(mood, '😐')
        
        st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 48px;">{mood_emoji}</div>
                <div style="font-size: 20px; font-weight: bold; color: #fbbf24;">{mood.upper()}</div>
                <div style="font-size: 12px; color: #94a3b8;">Confidence: {confidence:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.session_state.ml_predictions.append({
            'timestamp': datetime.now(),
            'mood': mood,
            'confidence': confidence
        })
    
    if hasattr(st.session_state, 'ml_accuracy'):
        st.metric("Model Accuracy", f"{st.session_state.ml_accuracy:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# SECTION 4: RECENT PREDICTIONS & ACTIVITY
# ============================================
st.divider()

col_act, col_hist = st.columns(2)

with col_act:
    st.subheader("🔄 Live Activity Feed")
    if st.button("🔄 Refresh Feed", use_container_width=True):
        activities = [
            f"⚡ {random.choice(['🔥', '😡', '😐'])} New mood vote!",
            f"🗳️ New vote in {poll_type}",
            f"📊 {total_votes} total fans engaged"
        ]
        for act in activities:
            st.markdown(f"> {act}")

with col_hist:
    st.subheader("📜 Recent AI Predictions")
    if st.session_state.ml_predictions:
        df_hist = pd.DataFrame(st.session_state.ml_predictions[-5:])
        df_hist['timestamp'] = df_hist['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(df_hist[['timestamp', 'mood', 'confidence']], use_container_width=True)
    else:
        st.info("Click 'Predict Mood' to see predictions")

# ============================================
# SECTION 5: FOOTER
# ============================================
st.divider()
st.caption("⚡ Real-time updates | 🤖 XGBoost Powered | 🎯 GSSoC 2025 Contribution")