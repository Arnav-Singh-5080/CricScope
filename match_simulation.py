import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

# ─────────────────────────────────────────────
#  CONSTANTS & THEME
# ─────────────────────────────────────────────

GOLD       = "#d4af37"
GOLD_LIGHT = "#f0d060"
RED        = "#e05555"
TEAL       = "#4ecdc4"
PAPER_BG   = "rgba(15,15,20,0.0)"
GRID_COL   = "rgba(212,175,55,0.12)"
FONT_COL   = "#e8d5a3"

PHASE_COLORS = {
    "Powerplay (1-6)"  : "rgba(212,175,55,0.18)",
    "Middle (7-15)"    : "rgba(78,205,196,0.10)",
    "Death (16-20)"    : "rgba(224,85,85,0.13)",
}

IPL_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings",
]

# ─────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_deliveries():
    return pd.read_csv("deliveries.csv")

@st.cache_data
def load_matches():
    return pd.read_csv("matches.csv")

@st.cache_resource
def load_model():
    with open("pipe.pkl", "rb") as f:
        return pickle.load(f)

# ─────────────────────────────────────────────
#  THEME HELPER
# ─────────────────────────────────────────────

def apply_gold_theme(fig, title=""):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=GOLD, size=16, family="Cormorant Garamond, serif"),
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(color=FONT_COL, family="DM Sans, sans-serif"),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=FONT_COL)),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=FONT_COL)),
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor=GOLD, borderwidth=1,
            font=dict(color=FONT_COL),
        ),
        margin=dict(l=20, r=20, t=55, b=30),
        hovermode="x unified",
    )
    return fig

# ─────────────────────────────────────────────
#  CORE: BUILD OVER-BY-OVER TIMELINE FROM MODEL
# ─────────────────────────────────────────────

def _get_phase(over: int) -> str:
    if over <= 6:
        return "Powerplay (1-6)"
    elif over <= 15:
        return "Middle (7-15)"
    return "Death (16-20)"


@st.cache_data
def build_simulation_timeline(
    batting_team: str,
    bowling_team: str,
    city: str,
    target: int,
    _model,           # leading _ so st.cache_data skips hashing it
) -> pd.DataFrame:
    """
    Simulate the probability trajectory over 20 overs using the
    existing pipe.pkl model with a 'typical' scoring curve derived
    from historical deliveries for the selected matchup.
    Falls back to a smooth synthetic curve if no history exists.
    """
    target = int(target)
    deliveries = load_deliveries()
    matches    = load_matches()

    # ── find historical matches between these two teams ──
    bat_col  = "batting_team" if "batting_team" in deliveries.columns else "bat_team"
    bowl_col = "bowling_team" if "bowling_team" in deliveries.columns else "bowl_team"

    # join to get team names
    if "team1" in matches.columns:
        id_col = "id" if "id" in matches.columns else matches.columns[0]
        team_matches = matches[
            ((matches["team1"] == batting_team) & (matches["team2"] == bowling_team)) |
            ((matches["team1"] == bowling_team) & (matches["team2"] == batting_team))
        ][id_col].tolist()
    else:
        team_matches = []

    # filter deliveries for inning 2 of those matches
    inn_col = "inning" if "inning" in deliveries.columns else "innings"
    hist = deliveries[
        (deliveries["match_id"].isin(team_matches)) &
        (deliveries[inn_col] == 2) &
        (deliveries[bat_col] == batting_team)
    ] if team_matches else pd.DataFrame()

    records = []

    if not hist.empty:
        # average cumulative score per over from history
        over_col = "over"
        runs_col = "total_runs" if "total_runs" in hist.columns else "batsman_runs"

        hist = hist.copy()
        if hist[over_col].min() == 0:
            hist[over_col] = hist[over_col] + 1

        if "player_dismissed" in hist.columns:
            hist["is_wicket"] = hist["player_dismissed"].notna().astype(int)
        else:
            hist["is_wicket"] = 0

        over_stats = (
            hist.groupby(["match_id", over_col])
            .agg(over_runs=( runs_col, "sum"), over_wkts=("is_wicket", "sum"))
            .reset_index()
            .groupby(over_col)
            .agg(avg_runs=("over_runs", "mean"), avg_wkts=("over_wkts", "mean"))
            .reset_index()
            .sort_values(over_col)
        )

        cum_runs    = 0
        cum_wickets = 0

        for over in range(1, 21):
            row = over_stats[over_stats[over_col] == over]
            if not row.empty:
                cum_runs    += row["avg_runs"].values[0]
                cum_wickets += row["avg_wkts"].values[0]
            else:
                cum_runs    += round(target / 20, 2)   # fallback: linear, float-safe
                cum_wickets += 0.45

            cum_wickets = min(cum_wickets, 10.0)
            # End-of-over simulation: count all six balls from the completed over.
            balls_bowled = over * 6
            balls_left   = max(120 - balls_bowled, 1)
            runs_left    = max(target - int(cum_runs), 0)
            wickets_left = max(10 - int(cum_wickets), 0)
            crr = cum_runs / over if over > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            try:
                input_df = pd.DataFrame([{
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "city"        : city,
                    "runs_left"   : runs_left,
                    "balls_left"  : balls_left,
                    "wickets"     : wickets_left,
                    "total_runs_x": target,
                    "crr"         : round(crr, 2),
                    "rrr"         : round(rrr, 2),
                }])
                prob = float(_model.predict_proba(input_df)[0][1])
            except Exception as e:
                import logging

                logging.warning(f"Model prediction failed at over {over}: {e}. Using fallback.")
                # if model fails (column mismatch etc.) use logistic approximation
                z    = 0.15 * (crr - rrr) + 0.05 * wickets_left - 0.02 * (runs_left / max(balls_left, 1))
                prob = float(np.clip(1 / (1 + np.exp(-z)), 0.0, 1.0))

            records.append({
                "over"        : over,
                "cum_runs"    : int(cum_runs),
                "cum_wickets" : int(cum_wickets),
                "runs_left"   : runs_left,
                "balls_left"  : balls_left,
                "wickets_left": wickets_left,
                "crr"         : round(crr, 2),
                "rrr"         : round(rrr, 2),
                "win_prob"    : round(prob * 100, 1),
                "phase"       : _get_phase(over),
            })

    else:
        # ── synthetic curve: smooth logistic ramp ──
        rng = np.random.default_rng(seed=42)   # deterministic, reproducible
        for over in range(1, 21):
            frac         = over / 20
            cum_runs     = int(target * frac * rng.uniform(0.82, 1.05))
            cum_wickets  = min(int(over * 0.4), 10)
            balls_bowled = over * 6
            balls_left   = max(120 - balls_bowled, 1)
            runs_left    = max(target - cum_runs, 0)
            wickets_left = max(10 - cum_wickets, 0)
            crr = cum_runs / over if over > 0 else 0
            rrr = (runs_left * 6) / balls_left

            try:
                input_df = pd.DataFrame([{
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "city"        : city,
                    "runs_left"   : runs_left,
                    "balls_left"  : balls_left,
                    "wickets"     : wickets_left,
                    "total_runs_x": target,
                    "crr"         : round(crr, 2),
                    "rrr"         : round(rrr, 2),
                }])
                prob = float(_model.predict_proba(input_df)[0][1])
            except Exception as e:
                import logging

                logging.warning(f"Model prediction failed at over {over}: {e}. Using fallback.")
                z    = 0.15 * (crr - rrr) + 0.05 * wickets_left - 0.02 * (runs_left / max(balls_left, 1))
                prob = float(np.clip(1 / (1 + np.exp(-z)), 0.0, 1.0))

            records.append({
                "over"        : over,
                "cum_runs"    : cum_runs,
                "cum_wickets" : cum_wickets,
                "runs_left"   : runs_left,
                "balls_left"  : balls_left,
                "wickets_left": wickets_left,
                "crr"         : round(crr, 2),
                "rrr"         : round(rrr, 2),
                "win_prob"    : round(prob * 100, 1),
                "phase"       : _get_phase(over),
            })
# ─────────────────────────────────────────────
#  WHAT-IF: recompute single probability
# ─────────────────────────────────────────────

def what_if_prob(model, batting_team, bowling_team, city, target,
                 current_over, current_score, current_wickets,
                 extra_runs, extra_wickets):
    new_over     = min(current_over + 2, 20)
    new_score    = current_score + extra_runs
    new_wickets  = min(current_wickets + extra_wickets, 10)
    balls_bowled = new_over * 6
    balls_left   = max(120 - balls_bowled, 1)
    runs_left    = max(target - new_score, 0)
    wickets_left = max(10 - new_wickets, 0)
    crr = new_score / new_over if new_over > 0 else 0
    rrr = (runs_left * 6) / balls_left

    try:
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "city"        : city,
            "runs_left"   : runs_left,
            "balls_left"  : balls_left,
            "wickets"     : wickets_left,
            "total_runs_x": target,
            "crr"         : round(crr, 2),
            "rrr"         : round(rrr, 2),
        }])
        return round(float(model.predict_proba(input_df)[0][1]) * 100, 1)
    except Exception as e:
        import logging

        logging.warning(f"Model prediction failed in what-if calculator: {e}. Using fallback.")
        z = 0.15 * (crr - rrr) + 0.05 * wickets_left - 0.02 * (runs_left / max(balls_left, 1))
        return round(float(np.clip(1 / (1 + np.exp(-z)), 0.0, 1.0)) * 100, 1)


# ─────────────────────────────────────────────
#  DETECT TOP 3 MOMENTUM SHIFTS
# ─────────────────────────────────────────────

def detect_momentum_shifts(df: pd.DataFrame, top_n: int = 3) -> list[dict]:
    shifts = []
    probs  = df["win_prob"].tolist()
    overs  = df["over"].tolist()

    for i in range(1, len(probs)):
        swing = abs(probs[i] - probs[i - 1])
        direction = "📈 Batting surge" if probs[i] > probs[i - 1] else "📉 Bowling dominance"
        shifts.append({
            "over"     : overs[i],
            "swing"    : round(swing, 1),
            "direction": direction,
            "prob_after": probs[i],
        })

    top = sorted(shifts, key=lambda x: x["swing"], reverse=True)[:top_n]
    return sorted(top, key=lambda x: x["over"])


# ─────────────────────────────────────────────
#  RENDER: PROBABILITY TIMELINE CHART
# ─────────────────────────────────────────────

def render_timeline_chart(df: pd.DataFrame, batting_team: str, bowling_team: str,
                          momentum_shifts: list[dict]):
    fig = go.Figure()

    # Phase background shading
    phase_bounds = {
        "Powerplay (1-6)"  : (1,  6),
        "Middle (7-15)"    : (7,  15),
        "Death (16-20)"    : (16, 20),
    }
    for phase, (x0, x1) in phase_bounds.items():
        fig.add_vrect(
            x0=x0 - 0.5, x1=x1 + 0.5,
            fillcolor=PHASE_COLORS[phase],
            layer="below", line_width=0,
            annotation_text=phase.split(" ")[0],
            annotation_position="top left",
            annotation_font=dict(color="#a08c50", size=10),
        )

    # 50% reference line
    fig.add_hline(
        y=50, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1,
        annotation_text="50%", annotation_font=dict(color="#666", size=10),
    )

    # Win probability line — batting team
    fig.add_trace(go.Scatter(
        x=df["over"], y=df["win_prob"],
        mode="lines+markers",
        name=f"{batting_team} Win %",
        line=dict(color=GOLD, width=3),
        marker=dict(size=7, color=GOLD, line=dict(color=GOLD_LIGHT, width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(212,175,55,0.07)",
        hovertemplate=(
            "<b>Over %{x}</b><br>"
            f"{batting_team}: " + "%{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Bowling team (inverse)
    fig.add_trace(go.Scatter(
        x=df["over"], y=100 - df["win_prob"],
        mode="lines",
        name=f"{bowling_team} Win %",
        line=dict(color=TEAL, width=2, dash="dot"),
        hovertemplate=(
            "<b>Over %{x}</b><br>"
            f"{bowling_team}: " + "%{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Cumulative runs secondary trace
    fig.add_trace(go.Bar(
        x=df["over"], y=df["cum_runs"],
        name="Score",
        marker_color="rgba(212,175,55,0.15)",
        marker_line=dict(color="rgba(212,175,55,0.3)", width=0.5),
        yaxis="y2",
        hovertemplate="<b>Over %{x}</b><br>Score: %{y}<extra></extra>",
    ))

    # Momentum shift annotations
    for shift in momentum_shifts:
        fig.add_vline(
            x=shift["over"],
            line_dash="dot",
            line_color=RED,
            line_width=1.5,
            annotation_text=f"Ov {shift['over']}: {shift['swing']}% shift",
            annotation_position="top right",
            annotation_font=dict(color=RED, size=10),
        )

    apply_gold_theme(fig, "📈 Win Probability Timeline — Over by Over")
    fig.update_layout(
        height=440,
        yaxis =dict(title="Win Probability (%)", range=[0, 105]),
        yaxis2=dict(title="Score", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color="#666", size=10)),
        xaxis =dict(title="Over", dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  RENDER: MOMENTUM SHIFT CARDS
# ─────────────────────────────────────────────

def render_momentum_cards(shifts: list[dict], batting_team: str):
    st.markdown("""
    <div style="font-family:'Cormorant Garamond',serif;color:#d4af37;
                font-size:15px;letter-spacing:2px;text-transform:uppercase;
                margin:20px 0 12px 0;border-bottom:1px solid rgba(212,175,55,0.3);padding-bottom:6px;">
        ⚡ Top 3 Momentum Shift Moments
    </div>""", unsafe_allow_html=True)

    cols = st.columns(3)
    for col, shift in zip(cols, shifts):
        border      = GOLD if "surge" in shift["direction"] else RED
        s_over      = int(shift["over"])
        s_direction = str(shift["direction"])
        s_swing     = float(shift["swing"])
        s_prob      = float(shift["prob_after"])
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(212,175,55,0.06),rgba(15,15,20,0.95));
                    border:1px solid {border};border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:24px;font-weight:700;color:{border};
                        font-family:'DM Mono',monospace;">Over {s_over}</div>
            <div style="font-size:13px;color:#e8d5a3;margin:6px 0;">{s_direction}</div>
            <div style="font-size:20px;color:{border};font-family:'DM Mono',monospace;font-weight:700;">
                {s_swing}% swing
            </div>
            <div style="font-size:11px;color:#a08c50;margin-top:4px;">
                {batting_team} at {s_prob}% after this over
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RENDER: WHAT-IF SLIDERS
# ─────────────────────────────────────────────

def render_what_if(model, batting_team, bowling_team, city, target, df: pd.DataFrame):
    st.markdown("""
    <div style="font-family:'Cormorant Garamond',serif;color:#d4af37;
                font-size:15px;letter-spacing:2px;text-transform:uppercase;
                margin:24px 0 12px 0;border-bottom:1px solid rgba(212,175,55,0.3);padding-bottom:6px;">
        🎛️ What-If Scenario Simulator
    </div>""", unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#a08c50;font-size:13px;'>Adjust the sliders to see how the next 2 overs "
        "could shift the win probability in real time.</p>",
        unsafe_allow_html=True
    )

    mid_over   = 10
    mid_row    = df[df["over"] == mid_over].iloc[0] if mid_over in df["over"].values else df.iloc[9]
    curr_score = int(mid_row["cum_runs"])
    curr_wkts  = int(mid_row["cum_wickets"])

    c1, c2 = st.columns(2)
    extra_runs = c1.slider(
        "Runs scored in next 2 overs", min_value=0, max_value=36,
        value=12, step=1, key="wi_runs"
    )
    extra_wickets = c2.slider(
        "Wickets lost in next 2 overs", min_value=0, max_value=4,
        value=1, step=1, key="wi_wkts"
    )

    baseline_prob = float(mid_row["win_prob"])
    new_prob      = what_if_prob(
        model, batting_team, bowling_team, city, target,
        mid_over, curr_score, curr_wkts, extra_runs, extra_wickets
    )
    delta         = round(new_prob - baseline_prob, 1)
    delta_str     = f"+{delta}%" if delta >= 0 else f"{delta}%"
    delta_color   = GOLD if delta >= 0 else RED

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(212,175,55,0.08),rgba(15,15,20,0.95));
                border:1px solid rgba(212,175,55,0.35);border-radius:12px;
                padding:20px;margin-top:12px;text-align:center;">
        <div style="font-size:13px;color:#a08c50;letter-spacing:1px;text-transform:uppercase;
                    margin-bottom:8px;">Projected Win Probability after Over 12</div>
        <div style="font-size:42px;font-weight:700;color:#f0d060;font-family:'DM Mono',monospace;">
            {new_prob}%
        </div>
        <div style="font-size:16px;color:{delta_color};margin-top:4px;font-family:'DM Mono',monospace;">
            {delta_str} vs baseline ({baseline_prob}%)
        </div>
        <div style="font-size:12px;color:#a08c50;margin-top:8px;">
            Scenario: +{extra_runs} runs, +{extra_wickets} wickets in overs 11–12
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION HEADER HELPER
# ─────────────────────────────────────────────

def _section_banner():
    st.markdown("""
    <div style="background:linear-gradient(90deg,rgba(212,175,55,0.15),rgba(212,175,55,0.02));
                border-left:3px solid #d4af37;padding:14px 20px;margin:24px 0 20px 0;
                border-radius:0 8px 8px 0;">
        <span style="font-family:'Cormorant Garamond',serif;font-size:22px;
                     color:#d4af37;letter-spacing:2px;text-transform:uppercase;">
            🎬 Match Simulation Mode
        </span><br>
        <span style="font-size:13px;color:#a08c50;">
            Simulate how win probability evolves ball-by-ball across 20 overs
        </span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT — call from application.py
# ─────────────────────────────────────────────

def render_match_simulation():
    """
    Call this from the Analysis / Simulation page in application.py.
    Renders all inputs independently — does not need batting_team passed in.
    """
    _section_banner()

    # ── Inputs ──
    col1, col2 = st.columns(2)
    batting_team = col1.selectbox("Batting Team", IPL_TEAMS, key="sim_bat")
    bowling_team = col2.selectbox("Bowling Team",
                                  [t for t in IPL_TEAMS if t != batting_team],
                                  key="sim_bowl")

    col3, col4 = st.columns(2)
    cities = [
        "Mumbai", "Kolkata", "Delhi", "Chennai", "Bangalore",
        "Hyderabad", "Jaipur", "Chandigarh", "Pune", "Dharamsala",
    ]
    city   = col3.selectbox("Venue City", cities, key="sim_city")
    target = col4.number_input("Target (Runs)", min_value=50, max_value=300,
                                value=175, step=1, key="sim_target")

    if st.button("▶ Run Simulation", use_container_width=True, type="primary"):
        model = load_model()

        with st.spinner("Simulating over-by-over probability trajectory…"):
            df = build_simulation_timeline(batting_team, bowling_team, city, int(target), model)

        if df.empty:
            st.error("Could not build simulation. Check that matches.csv and deliveries.csv are present.")
            return

        momentum_shifts = detect_momentum_shifts(df, top_n=3)

        # ── Tab layout ──
        tab1, tab2, tab3 = st.tabs([
            "📈 Probability Timeline",
            "⚡ Momentum Shifts",
            "🎛️ What-If Simulator",
        ])

        with tab1:
            render_timeline_chart(df, batting_team, bowling_team, momentum_shifts)

            # Summary stats row
            final_prob = df["win_prob"].iloc[-1]
            max_prob   = df["win_prob"].max()
            min_prob   = df["win_prob"].min()
            peak_over  = int(df.loc[df["win_prob"].idxmax(), "over"])
            trough_over= int(df.loc[df["win_prob"].idxmin(), "over"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Win Prob",    f"{final_prob}%")
            m2.metric("Peak Prob",         f"{max_prob}%",   f"Over {peak_over}")
            m3.metric("Lowest Prob",       f"{min_prob}%",   f"Over {trough_over}")
            m4.metric("Avg Prob",          f"{df['win_prob'].mean():.1f}%")

        with tab2:
            render_momentum_cards(momentum_shifts, batting_team)

            # Phase-wise avg probability table
            st.markdown("""
            <div style="font-family:'Cormorant Garamond',serif;color:#d4af37;
                        font-size:14px;letter-spacing:1.5px;text-transform:uppercase;
                        margin:20px 0 10px 0;">Phase-wise Average Win Probability</div>""",
                        unsafe_allow_html=True)

            phase_df = (
                df.groupby("phase")["win_prob"]
                .agg(["mean", "min", "max"])
                .round(1)
                .reset_index()
                .rename(columns={"phase":"Phase","mean":"Avg %","min":"Min %","max":"Max %"})
            )
            st.dataframe(
                phase_df,
                use_container_width=True,
                hide_index=True,
            )

        with tab3:
            render_what_if(model, batting_team, bowling_team, city, int(target), df)
