import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from model_evaluation import (
    calculate_model_summary,
    calculate_phase_wise_metrics,
    calculate_pressure_situation_metrics,
    calculate_season_wise_metrics,
    create_probability_buckets,
    prepare_match_state_data,
    run_historical_backtest,
)


st.set_page_config(
    page_title="Model Reliability & Backtesting",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88));
        border: 1px solid rgba(250, 204, 21, 0.25);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28);
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: #facc15;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .subtle-text {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_datasets():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries


@st.cache_data(show_spinner=False)
def build_backtest(sample_size: int):
    matches, deliveries = load_datasets()
    match_states = prepare_match_state_data(matches, deliveries)

    if sample_size and len(match_states) > sample_size:
        match_states = (
            match_states.sample(sample_size, random_state=42)
            .sort_values(["season", "match_id"])
            .reset_index(drop=True)
        )

    predictions = run_historical_backtest(match_states)
    return match_states, predictions


def format_percent(value):
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def format_number(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.3f}"


st.markdown("# Model Reliability & Backtesting")
st.markdown(
    """
    <p class="subtle-text">
    This dashboard evaluates whether CricScope's win probability predictions are historically reliable.
    It checks calibration, Brier Score, season-wise stability, innings-phase performance,
    and pressure chase behavior.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Backtest Settings")
    sample_size = st.slider(
        "Historical match states to evaluate",
        min_value=2000,
        max_value=30000,
        value=12000,
        step=1000,
        help="Higher values are more complete but may take longer to run.",
    )

    st.info(
        "Backtesting trains on earlier seasons and evaluates on later seasons wherever possible."
    )

with st.spinner("Running historical model backtest..."):
    match_states, predictions = build_backtest(sample_size)

if predictions.empty:
    st.error(
        "Backtesting could not generate predictions. Please check matches.csv and deliveries.csv."
    )
    st.stop()

summary = calculate_model_summary(predictions)
calibration_table = create_probability_buckets(predictions)
season_metrics = calculate_season_wise_metrics(predictions)
phase_metrics = calculate_phase_wise_metrics(predictions)
pressure_metrics = calculate_pressure_situation_metrics(predictions)

st.markdown('<div class="section-title">Overall Reliability Summary</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Evaluated States</div>
            <div class="metric-value">{summary["samples"]:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{format_percent(summary["accuracy"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Brier Score</div>
            <div class="metric-value">{format_number(summary["brier_score"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{format_percent(summary["average_confidence"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Brier Score measures probability reliability. Lower is better. "
    "A perfectly calibrated model should have actual win rates close to predicted probabilities."
)

st.markdown('<div class="section-title">Calibration Curve</div>', unsafe_allow_html=True)

curve_df = calibration_table.dropna(
    subset=["average_predicted_probability", "actual_win_rate"]
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(dash="dash", color="#94a3b8"),
    )
)

fig.add_trace(
    go.Scatter(
        x=curve_df["average_predicted_probability"],
        y=curve_df["actual_win_rate"],
        mode="lines+markers",
        name="CricScope Calibration",
        line=dict(color="#facc15", width=3),
        marker=dict(size=9),
    )
)

fig.update_layout(
    xaxis_title="Average Predicted Win Probability",
    yaxis_title="Actual Win Rate",
    template="plotly_dark",
    height=460,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">Probability Bucket Table</div>', unsafe_allow_html=True)

display_calibration = calibration_table.copy()
for col in [
    "average_predicted_probability",
    "actual_win_rate",
    "calibration_gap",
]:
    display_calibration[col] = display_calibration[col].apply(format_percent)

st.dataframe(display_calibration, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">Season-wise Model Stability</div>', unsafe_allow_html=True)

if not season_metrics.empty:
    season_chart = px.bar(
        season_metrics,
        x="season",
        y="brier_score",
        hover_data=[
            "samples",
            "accuracy",
            "average_predicted_probability",
            "actual_win_rate",
            "calibration_gap",
        ],
        title="Brier Score by Season",
        template="plotly_dark",
    )
    season_chart.update_traces(marker_color="#facc15")
    st.plotly_chart(season_chart, use_container_width=True)

    season_display = season_metrics.copy()
    for col in [
        "accuracy",
        "average_predicted_probability",
        "actual_win_rate",
        "calibration_gap",
    ]:
        season_display[col] = season_display[col].apply(format_percent)

    season_display["brier_score"] = season_display["brier_score"].apply(format_number)
    st.dataframe(season_display, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">Innings Phase Performance</div>', unsafe_allow_html=True)

if not phase_metrics.empty:
    phase_chart = px.bar(
        phase_metrics,
        x="phase",
        y="accuracy",
        color="phase",
        hover_data=[
            "samples",
            "brier_score",
            "average_predicted_probability",
            "actual_win_rate",
            "calibration_gap",
        ],
        title="Accuracy by Match Phase",
        template="plotly_dark",
    )
    st.plotly_chart(phase_chart, use_container_width=True)

    phase_display = phase_metrics.copy()
    for col in [
        "accuracy",
        "average_predicted_probability",
        "actual_win_rate",
        "calibration_gap",
    ]:
        phase_display[col] = phase_display[col].apply(format_percent)

    phase_display["brier_score"] = phase_display["brier_score"].apply(format_number)
    st.dataframe(phase_display, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">Pressure Situation Analysis</div>', unsafe_allow_html=True)

st.markdown(
    """
    <p class="subtle-text">
    Pressure chase states are detected when required run rate is above 10,
    balls left are below 30, wickets in hand are 4 or fewer, and runs left are above 40.
    </p>
    """,
    unsafe_allow_html=True,
)

if not pressure_metrics.empty:
    pressure_display = pressure_metrics.copy()

    for col in [
        "accuracy",
        "average_predicted_probability",
        "actual_win_rate",
        "calibration_gap",
    ]:
        pressure_display[col] = pressure_display[col].apply(format_percent)

    pressure_display["brier_score"] = pressure_display["brier_score"].apply(format_number)

    st.dataframe(pressure_display, use_container_width=True, hide_index=True)

with st.expander("What do these metrics mean?"):
    st.markdown(
        """
        - **Brier Score:** Measures how close predicted probabilities are to actual outcomes. Lower is better.
        - **Calibration Gap:** Actual win rate minus average predicted probability.
        - **Probability Buckets:** Groups predictions into 0–10%, 10–20%, etc. to check reliability.
        - **Season Stability:** Shows whether model quality changes across IPL seasons.
        - **Phase Analysis:** Compares reliability in Powerplay, Middle Overs, and Death Overs.
        - **Pressure Analysis:** Evaluates model behavior in difficult chase situations.
        """
    )
