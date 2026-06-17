import streamlit as st

st.set_page_config(
    page_title="404 - CricScope",
    page_icon="🏏",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
}

.error-card {
    max-width: 750px;
    margin: 80px auto;
    padding: 50px;
    text-align: center;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.brand {
    font-size: 3rem;
    font-weight: 800;
    color: #fbbf24;
    margin-bottom: 10px;
}

.error-code {
    font-size: 7rem;
    font-weight: 900;
    color: #fbbf24;
    line-height: 1;
    margin-bottom: 10px;
}

.title {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
}

.description {
    font-size: 1.1rem;
    color: #d1d5db;
    line-height: 1.8;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    margin-top: 30px;
    color: #9ca3af;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# Main Card
st.markdown("""
<div class="error-card">

<div class="brand">
🏏 CricScope
</div>

<div class="error-code">
404
</div>

<div class="title">
Page Not Found
</div>

<div class="description">
Oops! The page you're looking for doesn't exist,
may have been moved, or the URL is incorrect.
<br><br>
Return to the dashboard to continue exploring
IPL analytics, match insights, and real-time
win probability predictions.
</div>

</div>
""", unsafe_allow_html=True)

# Dashboard Button
if st.button("🏠 Go to Dashboard", use_container_width=True):
    st.switch_page("application.py")

# Footer
st.markdown("""
<div class="footer">
CricScope • IPL Match Intelligence Dashboard
</div>
""", unsafe_allow_html=True)