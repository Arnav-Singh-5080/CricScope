import streamlit as st
import streamlit.components.v1 as components
import random
import time
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Fan Pulse Arena", page_icon="🔥", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        font-size: 24px;
        padding: 20px;
    }
    .mood-stats {
        font-size: 18px;
        font-weight: bold;
    }
    .vote-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Fan Pulse Arena")
st.markdown("*Real-time community mood and polls*")

# Initialize session state for local demo (no Firebase needed)
if 'mood_votes' not in st.session_state:
    st.session_state.mood_votes = {'😡': 0, '😐': 0, '🔥': 0}
if 'poll_votes' not in st.session_state:
    st.session_state.poll_votes = {
        'Virat Kohli': 0,
        'Rohit Sharma': 0,
        'KL Rahul': 0,
        'Shubman Gill': 0
    }
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Live Mood Meter")
    st.markdown("**How are you feeling right now?**")
    
    # Create 3 columns for mood buttons
    mood_col1, mood_col2, mood_col3 = st.columns(3)
    
    with mood_col1:
        if st.button("😡 ANGRY", use_container_width=True):
            st.session_state.mood_votes['😡'] += 1
            st.rerun()
    
    with mood_col2:
        if st.button("😐 NEUTRAL", use_container_width=True):
            st.session_state.mood_votes['😐'] += 1
            st.rerun()
    
    with mood_col3:
        if st.button("🔥 EXCITED", use_container_width=True):
            st.session_state.mood_votes['🔥'] += 1
            st.rerun()
    
    # Calculate percentages
    total_votes = sum(st.session_state.mood_votes.values())
    if total_votes > 0:
        angry_pct = (st.session_state.mood_votes['😡'] / total_votes) * 100
        neutral_pct = (st.session_state.mood_votes['😐'] / total_votes) * 100
        excited_pct = (st.session_state.mood_votes['🔥'] / total_votes) * 100
        
        # Display progress bars
        st.markdown("### Current Mood Distribution")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**😡 ANGRY**")
            st.progress(int(angry_pct), text=f"{angry_pct:.1f}%")
            st.caption(f"{st.session_state.mood_votes['😡']} votes")
        
        with col_b:
            st.markdown(f"**😐 NEUTRAL**")
            st.progress(int(neutral_pct), text=f"{neutral_pct:.1f}%")
            st.caption(f"{st.session_state.mood_votes['😐']} votes")
        
        with col_c:
            st.markdown(f"**🔥 EXCITED**")
            st.progress(int(excited_pct), text=f"{excited_pct:.1f}%")
            st.caption(f"{st.session_state.mood_votes['🔥']} votes")
        
        # Determine dominant mood
        moods = {'😡': angry_pct, '😐': neutral_pct, '🔥': excited_pct}
        dominant = max(moods, key=moods.get)
        
        st.info(f"💡 **Fan Sentiment:** The crowd is feeling {dominant} right now!")
    else:
        st.info("👆 Click on a mood button above to start the voting!")
    
    # Display total votes
    st.metric("Total Mood Votes", total_votes)

with col2:
    st.subheader("🏆 Best Role Polls")
    
    # Poll type selector
    poll_type = st.selectbox(
        "Select Poll Category",
        ["Best Powerplay Opener", "Most Clutch Finisher", "Best Death Bowler", "MVP of the Tournament"]
    )
    
    st.markdown("### Cast your vote")
    
    # Dynamic player options based on poll type
    player_options = {
        "Best Powerplay Opener": ["Rohit Sharma", "Jos Buttler", "Quinton de Kock", "David Warner"],
        "Most Clutch Finisher": ["MS Dhoni", "Hardik Pandya", "Andre Russell", "Kieron Pollard"],
        "Best Death Bowler": ["Jasprit Bumrah", "Kagiso Rabada", "Trent Boult", "Pat Cummins"],
        "MVP of the Tournament": ["Virat Kohli", "Rashid Khan", "Jos Buttler", "Jasprit Bumrah"]
    }
    
    players = player_options.get(poll_type, ["Player 1", "Player 2", "Player 3", "Player 4"])
    
    # Create vote buttons for each player
    vote_cols = st.columns(2)
    vote_results = {}
    
    # Reset poll votes when poll type changes
    if 'current_poll' not in st.session_state or st.session_state.current_poll != poll_type:
        st.session_state.current_poll = poll_type
        st.session_state.poll_votes = {player: 0 for player in players}
    
    for idx, player in enumerate(players):
        col_idx = idx % 2
        with vote_cols[col_idx]:
            if st.button(f"🗳️ Vote for {player}", key=f"poll_{player}"):
                st.session_state.poll_votes[player] = st.session_state.poll_votes.get(player, 0) + 1
                st.success(f"✓ You voted for {player}!")
                st.rerun()
    
    st.markdown("### 📊 Poll Results")
    
    # Display results as bar chart using native Streamlit
    if st.session_state.poll_votes:
        votes_df = pd.DataFrame({
            'Player': list(st.session_state.poll_votes.keys()),
            'Votes': list(st.session_state.poll_votes.values())
        }).sort_values('Votes', ascending=True)
        
        st.bar_chart(votes_df.set_index('Player'))
        
        # Show winner
        if votes_df['Votes'].sum() > 0:
            winner = votes_df.loc[votes_df['Votes'].idxmax(), 'Player']
            st.success(f"🏆 Current Leader: **{winner}** with {max(st.session_state.poll_votes.values())} votes!")
    else:
        st.info("No votes yet. Be the first to vote!")

# Add a third section for live updates simulation
st.divider()

# Real-time update simulation
col_updates, col_insights = st.columns(2)

with col_updates:
    st.subheader("🔄 Live Activity Feed")
    update_placeholder = st.empty()
    
    # Simulate live updates
    if st.button("🔄 Refresh Feed", use_container_width=True):
        activities = [
            f"⚡ {random.choice(['🔥', '😡', '😐'])} New mood vote recorded!",
            f"🗳️ Someone just voted in '{poll_type}'",
            f"📊 Mood shifted to {dominant if total_votes > 0 else 'neutral'}",
            f"💬 {total_votes} total fans engaged so far!"
        ]
        for activity in activities:
            update_placeholder.markdown(f"> {activity}")
            
with col_insights:
    st.subheader("📈 Sentiment Insights")
    if total_votes > 0:
        st.metric("Fan Engagement", f"{total_votes} votes")
        if 'excited_pct' in locals():
            st.metric("Excitement Level", f"{excited_pct:.0f}%")
        st.caption("🤖 *AI-Powered insights coming soon with XGBoost forecasting!*")
    else:
        st.caption("Start voting to see insights!")

# Footer
st.divider()
st.caption("⚡ Real-time updates | 🔥 Powered by Streamlit | 🎯 GSSoC 2025 Contribution")