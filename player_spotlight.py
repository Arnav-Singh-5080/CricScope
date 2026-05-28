import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  DATA LOADERS  (cached so CSVs load once)
# ─────────────────────────────────────────────

@st.cache_data
def load_deliveries():
    return pd.read_csv("deliveries.csv")

@st.cache_data
def load_matches():
    return pd.read_csv("matches.csv")

# ─────────────────────────────────────────────
#  GOLD THEME HELPER
# ─────────────────────────────────────────────

GOLD       = "#d4af37"
GOLD_LIGHT = "#f0d060"
PAPER_BG   = "rgba(15,15,20,0.0)"
GRID_COL   = "rgba(212,175,55,0.12)"
FONT_COL   = "#e8d5a3"

def apply_gold_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=GOLD, size=16, family="Cormorant Garamond, serif")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_COL, family="DM Sans, sans-serif"),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=FONT_COL)),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=FONT_COL)),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GOLD, borderwidth=1, font=dict(color=FONT_COL)),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

# ─────────────────────────────────────────────
#  SECTION 1 — OVER-PHASE HEATMAP
# ─────────────────────────────────────────────

def get_phase(over):
    if over <= 6:
        return "Powerplay (1-6)"
    elif over <= 15:
        return "Middle (7-15)"
    else:
        return "Death (16-20)"

@st.cache_data
def compute_phase_stats(deliveries_key="deliveries"):
    df = load_deliveries()

    # Detect column names
    bat_col  = "batting_team" if "batting_team" in df.columns else "bat_team"
    runs_col = "total_runs"   if "total_runs"   in df.columns else "batsman_runs"
    over_col = "over"

    if "player_dismissed" in df.columns:
        df["is_wicket"] = df["player_dismissed"].notna().astype(int)
    else:
        df["is_wicket"] = 0

    # Normalise to 1-indexed overs
    if df[over_col].min() == 0:
        df = df.copy()
        df[over_col] = df[over_col] + 1

    df["phase"] = df[over_col].apply(get_phase)

    grp = (
        df.groupby(["match_id", bat_col, "phase"])
        .agg(phase_runs=("total_runs" if "total_runs" in df.columns else runs_col, "sum"),
             phase_wickets=("is_wicket", "sum"))
        .reset_index()
    )

    summary = (
        grp.groupby([bat_col, "phase"])
        .agg(avg_runs=("phase_runs", "mean"), avg_wickets=("phase_wickets", "mean"))
        .reset_index()
        .rename(columns={bat_col: "team"})
    )
    return summary


def render_phase_heatmap(batting_team, bowling_team):
    summary     = compute_phase_stats()
    phase_order = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
    teams       = [batting_team, bowling_team]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{batting_team}", f"{bowling_team}"],
        horizontal_spacing=0.14,
    )

    for col_idx, team in enumerate(teams, start=1):
        team_data = summary[summary["team"] == team].set_index("phase")

        runs_vals = [team_data.loc[p, "avg_runs"] if p in team_data.index else 0 for p in phase_order]
        wkt_vals  = [team_data.loc[p, "avg_wickets"] if p in team_data.index else 0 for p in phase_order]

        fig.add_trace(
            go.Bar(
                x=phase_order, y=runs_vals,
                name=f"{team} — Runs",
                marker=dict(
                    color=runs_vals,
                    colorscale=[[0,"rgba(212,175,55,0.2)"],[1,GOLD]],
                    showscale=False,
                    line=dict(color=GOLD_LIGHT, width=0.8),
                ),
                text=[f"{v:.1f}" for v in runs_vals],
                textposition="outside",
                textfont=dict(color=GOLD_LIGHT, size=11),
                hovertemplate="<b>%{x}</b><br>Avg Runs: %{y:.1f}<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=phase_order, y=wkt_vals,
                name=f"{team} — Wickets",
                mode="lines+markers",
                line=dict(color="#e05555", width=2, dash="dot"),
                marker=dict(size=8, color="#e05555", symbol="diamond"),
                hovertemplate="<b>%{x}</b><br>Avg Wickets: %{y:.2f}<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    apply_gold_theme(fig, "Over-Phase Performance Heatmap")
    fig.update_layout(height=380, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  SECTION 2 — TOP PERFORMER CARDS
# ─────────────────────────────────────────────

@st.cache_data
def get_top_batters(team, top_n=3):
    df       = load_deliveries()
    bat_col  = "batting_team" if "batting_team" in df.columns else "bat_team"
    btr_col  = "batsman"      if "batsman"      in df.columns else "batter"
    runs_col = "batsman_runs" if "batsman_runs" in df.columns else "runs_off_bat"

    df = df[df[bat_col] == team]
    stats = (
        df.groupby(btr_col)
        .agg(
            total_runs     = (runs_col, "sum"),
            balls_faced    = (runs_col, "count"),
            matches_played = ("match_id", "nunique"),
        )
        .reset_index()
    )
    stats["strike_rate"]    = (stats["total_runs"] / stats["balls_faced"] * 100).round(1)
    stats["avg_runs_match"] = (stats["total_runs"] / stats["matches_played"]).round(1)
    stats = stats[stats["balls_faced"] >= 80]
    return stats.nlargest(top_n, "total_runs").rename(columns={btr_col: "player"})


@st.cache_data
def get_top_bowlers(team, top_n=3):
    df       = load_deliveries().copy()
    bowl_col = "bowling_team" if "bowling_team" in df.columns else "bowl_team"
    runs_col = "total_runs"   if "total_runs"   in df.columns else "batsman_runs"

    df["is_wicket"] = df["player_dismissed"].notna().astype(int) if "player_dismissed" in df.columns else 0

    df = df[df[bowl_col] == team]
    stats = (
        df.groupby("bowler")
        .agg(
            wickets        = ("is_wicket",   "sum"),
            runs_conceded  = (runs_col,       "sum"),
            balls_bowled   = (runs_col,       "count"),
            matches_played = ("match_id",    "nunique"),
        )
        .reset_index()
    )
    stats["economy"] = (stats["runs_conceded"] / (stats["balls_bowled"] / 6)).round(2)
    stats = stats[stats["balls_bowled"] >= 60]
    return stats.nlargest(top_n, "wickets").rename(columns={"bowler": "player"})


def _card(player, l1, v1, l2, v2, l3, v3, icon="🏏"):
    return f"""
    <div style="background:linear-gradient(135deg,rgba(212,175,55,0.08),rgba(15,15,20,0.95));
                border:1px solid rgba(212,175,55,0.35);border-radius:12px;
                padding:18px 16px;margin-bottom:12px;
                box-shadow:0 4px 24px rgba(212,175,55,0.08);">
        <div style="font-size:13px;color:#d4af37;font-family:'Cormorant Garamond',serif;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">{icon} {player}</div>
        <div style="display:flex;gap:16px;margin-top:10px;">
            <div style="flex:1;text-align:center;">
                <div style="font-size:20px;font-weight:700;color:#f0d060;font-family:'DM Mono',monospace;">{v1}</div>
                <div style="font-size:10px;color:#a08c50;text-transform:uppercase;letter-spacing:0.5px;">{l1}</div>
            </div>
            <div style="width:1px;background:rgba(212,175,55,0.2);"></div>
            <div style="flex:1;text-align:center;">
                <div style="font-size:20px;font-weight:700;color:#f0d060;font-family:'DM Mono',monospace;">{v2}</div>
                <div style="font-size:10px;color:#a08c50;text-transform:uppercase;letter-spacing:0.5px;">{l2}</div>
            </div>
            <div style="width:1px;background:rgba(212,175,55,0.2);"></div>
            <div style="flex:1;text-align:center;">
                <div style="font-size:20px;font-weight:700;color:#f0d060;font-family:'DM Mono',monospace;">{v3}</div>
                <div style="font-size:10px;color:#a08c50;text-transform:uppercase;letter-spacing:0.5px;">{l3}</div>
            </div>
        </div>
    </div>"""

def _section_header(text):
    st.markdown(f"""
    <div style="font-family:'Cormorant Garamond',serif;color:#d4af37;
                font-size:15px;letter-spacing:2px;text-transform:uppercase;
                margin-bottom:12px;border-bottom:1px solid rgba(212,175,55,0.3);padding-bottom:6px;">
        {text}
    </div>""", unsafe_allow_html=True)


def render_performer_cards(batting_team, bowling_team):
    col1, col2 = st.columns(2)

    with col1:
        _section_header(f"🏏 Top Batters — {batting_team}")
        try:
            for _, r in get_top_batters(batting_team).iterrows():
                st.markdown(_card(r["player"],
                    "Total Runs",  f"{int(r['total_runs']):,}",
                    "Strike Rate", f"{r['strike_rate']}",
                    "Avg/Match",   f"{r['avg_runs_match']}"),
                    unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"No data: {e}")

    with col2:
        _section_header(f"🎳 Top Bowlers — {bowling_team}")
        try:
            for _, r in get_top_bowlers(bowling_team).iterrows():
                st.markdown(_card(r["player"],
                    "Wickets",  f"{int(r['wickets'])}",
                    "Economy",  f"{r['economy']}",
                    "Matches",  f"{int(r['matches_played'])}",
                    icon="🎳"),
                    unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"No data: {e}")


# ─────────────────────────────────────────────
#  SECTION 3 — BATTER vs BOWLER MICRO-MATCHUP
# ─────────────────────────────────────────────

@st.cache_data
def get_all_batters():
    df  = load_deliveries()
    col = "batsman" if "batsman" in df.columns else "batter"
    return sorted(df[col].dropna().unique().tolist())

@st.cache_data
def get_all_bowlers():
    df = load_deliveries()
    return sorted(df["bowler"].dropna().unique().tolist())


def compute_matchup(batter, bowler):
    df       = load_deliveries().copy()
    btr_col  = "batsman"      if "batsman"      in df.columns else "batter"
    runs_col = "batsman_runs" if "batsman_runs" in df.columns else "runs_off_bat"

    df = df[(df[btr_col] == batter) & (df["bowler"] == bowler)]
    if df.empty:
        return None

    df["is_wicket"] = (df["player_dismissed"] == batter).astype(int) if "player_dismissed" in df.columns else 0

    balls      = len(df)
    runs       = int(df[runs_col].sum())
    dismissals = int(df["is_wicket"].sum())
    dots       = int((df[runs_col] == 0).sum())
    fours      = int((df[runs_col] == 4).sum())
    sixes      = int((df[runs_col] == 6).sum())

    return dict(
        balls=balls, runs=runs, dismissals=dismissals,
        dots=dots, fours=fours, sixes=sixes,
        strike_rate   = round(runs / balls * 100, 1) if balls else 0,
        dot_pct       = round(dots / balls * 100, 1) if balls else 0,
        boundary_pct  = round((fours + sixes) / balls * 100, 1) if balls else 0,
    )


def render_matchup_section():
    _section_header("⚔️ Batter vs Bowler — Micro Matchup")

    all_batters = get_all_batters()
    all_bowlers = get_all_bowlers()

    c1, c2 = st.columns(2)
    batter = c1.selectbox("Select Batter", all_batters, key="mu_batter")
    bowler = c2.selectbox("Select Bowler", all_bowlers, key="mu_bowler")

    if st.button("🔍 Analyse Matchup", use_container_width=True):
        m = compute_matchup(batter, bowler)
        if m is None:
            st.warning(f"No head-to-head data found for **{batter}** vs **{bowler}**.")
            return

        # Scorecards
        cols = st.columns(4)
        for col, (lbl, val) in zip(cols, [
            ("Balls Faced", m["balls"]),
            ("Runs Scored", m["runs"]),
            ("Dismissals",  m["dismissals"]),
            ("Strike Rate", f"{m['strike_rate']}%"),
        ]):
            col.metric(lbl, val)

        cols2 = st.columns(3)
        cols2[0].metric("Dot Ball %",  f"{m['dot_pct']}%")
        cols2[1].metric("Boundary %",  f"{m['boundary_pct']}%")
        cols2[2].metric("4s / 6s",     f"{m['fours']} / {m['sixes']}")

        # Chart
        categories = ["Dot %", "Boundary %", "SR / 10", "Dismissals"]
        batter_v   = [m["dot_pct"], m["boundary_pct"], m["strike_rate"] / 10, m["dismissals"] * 5]
        bowler_v   = [100 - m["dot_pct"], 100 - m["boundary_pct"],
                      max(0, (150 - m["strike_rate"]) / 10), m["dismissals"] * 5]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=batter_v, name=batter,
                             marker_color=GOLD,
                             hovertemplate="<b>%{x}</b>: %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Bar(x=categories, y=bowler_v, name=bowler,
                             marker_color="rgba(224,85,85,0.75)",
                             hovertemplate="<b>%{x}</b>: %{y:.1f}<extra></extra>"))
        apply_gold_theme(fig, f"⚔️ {batter}  vs  {bowler}")
        fig.update_layout(barmode="group", height=300)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT — call from application.py
# ─────────────────────────────────────────────

def render_player_spotlight(batting_team: str, bowling_team: str):
    """
    Call this function from the Analysis page in application.py.
    Pass the currently selected batting_team and bowling_team strings.
    """
    st.markdown("""
    <div style="background:linear-gradient(90deg,rgba(212,175,55,0.15),rgba(212,175,55,0.02));
                border-left:3px solid #d4af37;padding:14px 20px;margin:24px 0 20px 0;
                border-radius:0 8px 8px 0;">
        <span style="font-family:'Cormorant Garamond',serif;font-size:22px;
                     color:#d4af37;letter-spacing:2px;text-transform:uppercase;">
            🔭 Player Spotlight &amp; Phase Intelligence
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Phase Heatmap", "🌟 Top Performers", "⚔️ Batter vs Bowler"])

    with tab1:
        st.markdown("<p style='color:#a08c50;font-size:13px;'>Average runs and wickets across "
                    "Powerplay, Middle, and Death overs based on historical IPL data.</p>",
                    unsafe_allow_html=True)
        render_phase_heatmap(batting_team, bowling_team)

    with tab2:
        st.markdown("<p style='color:#a08c50;font-size:13px;'>Top 3 historical performers "
                    "for each team ranked by aggregate stats across all IPL seasons in the dataset.</p>",
                    unsafe_allow_html=True)
        render_performer_cards(batting_team, bowling_team)

    with tab3:
        render_matchup_section()
