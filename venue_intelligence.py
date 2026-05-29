import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────

GOLD       = "#d4af37"
GOLD_LIGHT = "#f0d060"
RED        = "#e05555"
TEAL       = "#4ecdc4"
PAPER_BG   = "rgba(15,15,20,0.0)"
GRID_COL   = "rgba(212,175,55,0.12)"
FONT_COL   = "#e8d5a3"

# ─────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_matches() -> pd.DataFrame:
    return pd.read_csv("matches.csv")

@st.cache_data
def load_deliveries() -> pd.DataFrame:
    return pd.read_csv("deliveries.csv")

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def apply_gold_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=GOLD, size=15, family="Cormorant Garamond, serif")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(color=FONT_COL, family="DM Sans, sans-serif"),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=FONT_COL)),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=FONT_COL)),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GOLD, borderwidth=1, font=dict(color=FONT_COL)),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _section_header(text: str):
    st.markdown(f"""
    <div style="font-family:'Cormorant Garamond',serif;color:{GOLD};
                font-size:15px;letter-spacing:2px;text-transform:uppercase;
                margin:20px 0 12px 0;border-bottom:1px solid rgba(212,175,55,0.3);
                padding-bottom:6px;">{text}</div>
    """, unsafe_allow_html=True)


def _info_card(label: str, value: str, sub: str = "", color: str = GOLD_LIGHT):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(212,175,55,0.07),rgba(15,15,20,0.95));
                border:1px solid rgba(212,175,55,0.3);border-radius:10px;
                padding:16px;text-align:center;margin-bottom:8px;">
        <div style="font-size:11px;color:#a08c50;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px;">{label}</div>
        <div style="font-size:26px;font-weight:700;color:{color};
                    font-family:'DM Mono',monospace;">{value}</div>
        <div style="font-size:11px;color:#a08c50;margin-top:4px;">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 1 — VENUE BIAS CARD
# ─────────────────────────────────────────────

@st.cache_data
def compute_venue_bias(city: str) -> dict:
    """
    Returns win rates for team batting first vs second at a given city.
    Also returns avg first innings score and avg winning margin.
    """
    matches = load_matches()

    # normalise city column name
    city_col = "city" if "city" in matches.columns else matches.columns[2]

    venue_df = matches[matches[city_col].str.lower() == city.lower()].copy()
    if venue_df.empty:
        return {}

    total = len(venue_df)

    # winner column
    winner_col = "winner" if "winner" in venue_df.columns else None
    toss_col   = "toss_winner"     if "toss_winner"     in venue_df.columns else None
    toss_dec   = "toss_decision"   if "toss_decision"   in venue_df.columns else None
    team1_col  = "team1"           if "team1"           in venue_df.columns else None
    team2_col  = "team2"           if "team2"           in venue_df.columns else None

    # batting first / second wins
    bat_first_wins  = 0
    bat_second_wins = 0

    if all(c is not None for c in [winner_col, toss_col, toss_dec]):
        for _, row in venue_df.iterrows():
            if pd.isna(row[winner_col]):
                continue
            toss_chose_bat = str(row.get(toss_dec, "")).lower() == "bat"
            if toss_chose_bat:
                # toss winner batted first
                if row[winner_col] == row[toss_col]:
                    bat_first_wins += 1
                else:
                    bat_second_wins += 1
            else:
                # toss winner fielded — so OTHER team batted first
                if row[winner_col] != row[toss_col]:
                    bat_first_wins += 1
                else:
                    bat_second_wins += 1

    bat_first_pct  = round(bat_first_wins  / total * 100, 1) if total > 0 else 50.0
    bat_second_pct = round(bat_second_wins / total * 100, 1) if total > 0 else 50.0

    # average first innings score from deliveries
    deliveries = load_deliveries()
    inn_col    = "inning" if "inning" in deliveries.columns else "innings"
    runs_col   = "total_runs" if "total_runs" in deliveries.columns else "batsman_runs"

    match_ids = venue_df["id"].tolist() if "id" in venue_df.columns else []
    avg_first_inns = 0
    if match_ids:
        first_inn = deliveries[
            (deliveries["match_id"].isin(match_ids)) &
            (deliveries[inn_col] == 1)
        ]
        if not first_inn.empty:
            scores = first_inn.groupby("match_id")[runs_col].sum()
            avg_first_inns = int(scores.mean())

    # winning margins
    avg_win_runs = 0
    avg_win_wkts = 0
    if "win_by_runs" in venue_df.columns:
        runs_wins = venue_df[venue_df["win_by_runs"] > 0]["win_by_runs"]
        avg_win_runs = int(runs_wins.mean()) if len(runs_wins) > 0 else 0
    if "win_by_wickets" in venue_df.columns:
        wkt_wins = venue_df[venue_df["win_by_wickets"] > 0]["win_by_wickets"]
        avg_win_wkts = int(wkt_wins.mean()) if len(wkt_wins) > 0 else 0

    return {
        "total_matches"  : total,
        "bat_first_pct"  : bat_first_pct,
        "bat_second_pct" : bat_second_pct,
        "bat_first_wins" : bat_first_wins,
        "bat_second_wins": bat_second_wins,
        "avg_first_inns" : avg_first_inns,
        "avg_win_runs"   : avg_win_runs,
        "avg_win_wkts"   : avg_win_wkts,
    }


def render_venue_bias_card(city: str):
    _section_header(f"🏟️ Venue Bias — {city}")

    bias = compute_venue_bias(city)
    if not bias:
        st.warning(f"No match data found for **{city}** in the dataset.")
        return

    # determine dominant side
    if bias["bat_second_pct"] > bias["bat_first_pct"]:
        bias_label = f"This venue favours CHASERS — {bias['bat_second_pct']}% of matches won batting second"
        bias_color = TEAL
        bar_val    = bias["bat_second_pct"] / 100
    else:
        bias_label = f"This venue favours DEFENDERS — {bias['bat_first_pct']}% of matches won batting first"
        bias_color = GOLD
        bar_val    = bias["bat_first_pct"] / 100

    # Bias bar
    fig = go.Figure(go.Bar(
        x=[bias["bat_first_pct"], bias["bat_second_pct"]],
        y=["Bat First", "Bat Second"],
        orientation="h",
        marker=dict(
            color=[GOLD, TEAL],
            line=dict(color="rgba(212,175,55,0.4)", width=0.8),
        ),
        text=[f"{bias['bat_first_pct']}%", f"{bias['bat_second_pct']}%"],
        textposition="inside",
        textfont=dict(color="#111", size=12, family="DM Mono, monospace"),
        hovertemplate="<b>%{y}</b>: %{x}%<extra></extra>",
    ))
    apply_gold_theme(fig, "Win % — Batting First vs Second")
    fig.update_layout(height=160, xaxis=dict(range=[0, 100], title="Win %"),
                      showlegend=False, margin=dict(l=80, r=20, t=45, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Insight callout
    st.markdown(f"""
    <div style="background:rgba(212,175,55,0.06);border-left:3px solid {bias_color};
                border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:12px;">
        <span style="color:{bias_color};font-size:13px;font-family:'DM Sans',sans-serif;">
            💡 {bias_label}
        </span>
    </div>""", unsafe_allow_html=True)

    # Stat row
    c1, c2, c3 = st.columns(3)
    _info_card.__func__ = _info_card  # avoid issue with reuse
    with c1:
        _info_card("Total Matches", str(bias["total_matches"]), "at this venue")
    with c2:
        _info_card("Avg 1st Inn Score", str(bias["avg_first_inns"]), "historical average")
    with c3:
        avg_margin = f"{bias['avg_win_runs']} runs / {bias['avg_win_wkts']} wkts"
        _info_card("Avg Win Margin", avg_margin, "runs or wickets")


# ─────────────────────────────────────────────
#  SECTION 2 — TOSS IMPACT SCORE
# ─────────────────────────────────────────────

@st.cache_data
def compute_toss_impact(city: str) -> dict:
    matches  = load_matches()
    city_col = "city" if "city" in matches.columns else matches.columns[2]
    venue_df = matches[matches[city_col].str.lower() == city.lower()].copy()

    if venue_df.empty or "toss_winner" not in venue_df.columns or "winner" not in venue_df.columns:
        return {}

    valid = venue_df.dropna(subset=["toss_winner", "winner"])
    total = len(valid)
    if total == 0:
        return {}

    toss_also_won = (valid["toss_winner"] == valid["winner"]).sum()
    toss_win_pct  = round(toss_also_won / total * 100, 1)
    toss_score    = int(toss_win_pct)   # 0-100 index

    if toss_score >= 60:
        label = "HIGH"
        color = RED
    elif toss_score >= 45:
        label = "MODERATE"
        color = GOLD
    else:
        label = "LOW"
        color = TEAL

    # preferred toss decision at this venue
    preferred_decision = ""
    if "toss_decision" in valid.columns:
        decision_counts = valid["toss_decision"].value_counts()
        preferred_decision = decision_counts.idxmax().upper() if not decision_counts.empty else ""

    return {
        "toss_win_pct"       : toss_win_pct,
        "toss_score"         : toss_score,
        "label"              : label,
        "color"              : color,
        "toss_also_won"      : int(toss_also_won),
        "total"              : total,
        "preferred_decision" : preferred_decision,
    }


def render_toss_impact(city: str):
    _section_header("🎲 Toss Impact Score")

    toss = compute_toss_impact(city)
    if not toss:
        st.info("Not enough toss data for this venue.")
        return

    c1, c2 = st.columns([1, 1.8])

    with c1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(212,175,55,0.07),rgba(15,15,20,0.95));
                    border:1px solid {toss['color']};border-radius:12px;
                    padding:20px;text-align:center;">
            <div style="font-size:11px;color:#a08c50;text-transform:uppercase;
                        letter-spacing:1px;">Toss Advantage Index</div>
            <div style="font-size:48px;font-weight:700;color:{toss['color']};
                        font-family:'DM Mono',monospace;margin:6px 0;">{toss['toss_score']}</div>
            <div style="font-size:14px;color:{toss['color']};font-weight:600;
                        letter-spacing:2px;">{toss['label']}</div>
            <div style="font-size:11px;color:#a08c50;margin-top:8px;">
                Toss winner also won match {toss['toss_win_pct']}% of times
                ({toss['toss_also_won']} / {toss['total']} games)
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        # Gauge-style bar
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=toss["toss_score"],
            title=dict(text="Toss Influence", font=dict(color=FONT_COL, size=13)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color=FONT_COL)),
                bar=dict(color=toss["color"]),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=1,
                bordercolor=GOLD,
                steps=[
                    dict(range=[0,  45], color="rgba(78,205,196,0.15)"),
                    dict(range=[45, 60], color="rgba(212,175,55,0.15)"),
                    dict(range=[60,100], color="rgba(224,85,85,0.15)"),
                ],
                threshold=dict(line=dict(color=GOLD_LIGHT, width=2), thickness=0.75, value=50),
            ),
            number=dict(font=dict(color=toss["color"], size=28, family="DM Mono, monospace"),
                        suffix="/100"),
        ))
        fig.update_layout(
            paper_bgcolor=PAPER_BG, height=180,
            font=dict(color=FONT_COL),
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    if toss["preferred_decision"]:
        st.markdown(f"""
        <div style="background:rgba(212,175,55,0.06);border-left:3px solid {GOLD};
                    border-radius:0 8px 8px 0;padding:10px 14px;margin-top:8px;">
            <span style="color:{GOLD};font-size:13px;">
                🏏 Teams at this venue most commonly choose to <b>{toss['preferred_decision']}</b> after winning the toss
            </span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 3 — TARGET DIFFICULTY CONTEXTUALIZER
# ─────────────────────────────────────────────

@st.cache_data
def get_avg_first_innings_score(city: str) -> int:
    bias = compute_venue_bias(city)
    return bias.get("avg_first_inns", 160)


def render_target_difficulty(city: str, target: int):
    _section_header("🎯 Target Difficulty Contextualizer")

    avg_score = get_avg_first_innings_score(city)
    if avg_score == 0:
        avg_score = 160   # safe default

    diff_pct = round((target - avg_score) / avg_score * 100, 1)

    if target <= avg_score * 0.85:
        label = "BELOW AVERAGE"
        color = TEAL
        desc  = "Well within historical norms — chasing team has a significant advantage."
    elif target <= avg_score * 1.0:
        label = "AVERAGE"
        color = GOLD
        desc  = "Typical score for this venue — expect a closely contested chase."
    elif target <= avg_score * 1.15:
        label = "ABOVE AVERAGE"
        color = GOLD_LIGHT
        desc  = "Marginally higher than the historical average — chasing is difficult but achievable."
    else:
        label = "EXTREME"
        color = RED
        desc  = "Significantly above historical norms — the defending team holds a clear edge."

    sign = "+" if diff_pct >= 0 else ""

    c1, c2, c3 = st.columns(3)
    with c1:
        _info_card("Venue Avg Target", str(avg_score), "historical 1st inn avg")
    with c2:
        _info_card("Current Target", str(target), "runs to chase")
    with c3:
        _info_card("Deviation", f"{sign}{diff_pct}%", "vs historical avg", color=color)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(212,175,55,0.07),rgba(15,15,20,0.95));
                border:1px solid {color};border-radius:10px;
                padding:14px 18px;margin-top:4px;">
        <span style="font-size:14px;font-weight:700;color:{color};
                     font-family:'DM Mono',monospace;letter-spacing:1px;">{label}</span>
        <span style="font-size:12px;color:#e8d5a3;margin-left:12px;">{desc}</span>
    </div>""", unsafe_allow_html=True)

    # Visual bar comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Venue Average", "Current Target"],
        y=[avg_score, target],
        marker=dict(
            color=[GOLD, color],
            line=dict(color=GOLD_LIGHT, width=0.8),
        ),
        text=[str(avg_score), str(target)],
        textposition="outside",
        textfont=dict(color=GOLD_LIGHT, size=13),
        hovertemplate="<b>%{x}</b>: %{y} runs<extra></extra>",
    ))
    apply_gold_theme(fig, "Target vs Historical Average")
    fig.update_layout(height=240, showlegend=False,
                      yaxis=dict(range=[0, max(avg_score, target) * 1.2]))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  SECTION 4 — HEAD-TO-HEAD VENUE RECORD
# ─────────────────────────────────────────────

@st.cache_data
def compute_h2h_venue(batting_team: str, bowling_team: str, city: str) -> dict:
    matches  = load_matches()
    city_col = "city" if "city" in matches.columns else matches.columns[2]

    venue_df = matches[matches[city_col].str.lower() == city.lower()]
    if "team1" not in venue_df.columns:
        return {}

    h2h = venue_df[
        ((venue_df["team1"] == batting_team) & (venue_df["team2"] == bowling_team)) |
        ((venue_df["team1"] == bowling_team) & (venue_df["team2"] == batting_team))
    ].copy()

    total = len(h2h)
    if total == 0:
        return {"total": 0}

    winner_col = "winner" if "winner" in h2h.columns else None
    bat_wins   = 0
    bowl_wins  = 0
    last_result = "N/A"

    if winner_col:
        bat_wins   = int((h2h[winner_col] == batting_team).sum())
        bowl_wins  = int((h2h[winner_col] == bowling_team).sum())
        last_row   = h2h.iloc[-1]
        last_result = str(last_row.get(winner_col, "N/A"))

    return {
        "total"      : total,
        "bat_wins"   : bat_wins,
        "bowl_wins"  : bowl_wins,
        "last_result": last_result,
        "no_result"  : total - bat_wins - bowl_wins,
    }


def render_h2h_venue(batting_team: str, bowling_team: str, city: str):
    _section_header(f"⚔️ Head-to-Head at {city} — {batting_team} vs {bowling_team}")

    h2h = compute_h2h_venue(batting_team, bowling_team, city)

    if not h2h or h2h.get("total", 0) == 0:
        st.info(f"No head-to-head matches found between **{batting_team}** and **{bowling_team}** at **{city}**.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _info_card("Matches Played", str(h2h["total"]), "at this venue")
    with c2:
        _info_card(f"{batting_team[:12]} Wins", str(h2h["bat_wins"]), "batting team", color=GOLD)
    with c3:
        _info_card(f"{bowling_team[:12]} Wins", str(h2h["bowl_wins"]), "bowling team", color=TEAL)
    with c4:
        winner_color = GOLD if h2h["last_result"] == batting_team else TEAL
        _info_card("Last Result", h2h["last_result"][:14], "most recent match", color=winner_color)

    # donut chart
    if h2h["total"] > 0:
        fig = go.Figure(go.Pie(
            labels=[batting_team, bowling_team, "No Result"],
            values=[h2h["bat_wins"], h2h["bowl_wins"], h2h["no_result"]],
            hole=0.62,
            marker=dict(colors=[GOLD, TEAL, "rgba(255,255,255,0.1)"],
                        line=dict(color="rgba(0,0,0,0.5)", width=1.5)),
            textfont=dict(color=FONT_COL, size=11),
            hovertemplate="<b>%{label}</b>: %{value} wins (%{percent})<extra></extra>",
        ))
        apply_gold_theme(fig, "Head-to-Head Win Split")
        fig.update_layout(height=260, showlegend=True,
                          annotations=[dict(text=f"{h2h['total']}<br>matches",
                                           x=0.5, y=0.5, showarrow=False,
                                           font=dict(color=GOLD, size=14,
                                                     family="DM Mono, monospace"))])
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def render_venue_intelligence(batting_team: str, bowling_team: str, city: str, target: int):
    """
    Call this from application.py, passing current match inputs.
    Renders the full Venue Context Intelligence panel with 4 sub-sections.
    """
    st.markdown("""
    <div style="background:linear-gradient(90deg,rgba(212,175,55,0.15),rgba(212,175,55,0.02));
                border-left:3px solid #d4af37;padding:14px 20px;margin:24px 0 20px 0;
                border-radius:0 8px 8px 0;">
        <span style="font-family:'Cormorant Garamond',serif;font-size:22px;
                     color:#d4af37;letter-spacing:2px;text-transform:uppercase;">
            🧠 Match Context Intelligence
        </span><br>
        <span style="font-size:13px;color:#a08c50;">
            Venue bias, toss influence, target difficulty & head-to-head records
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏟️ Venue Bias",
        "🎲 Toss Impact",
        "🎯 Target Difficulty",
        "⚔️ Head-to-Head",
    ])

    with tab1:
        st.markdown(
            "<p style='color:#a08c50;font-size:13px;'>Historical win rates for teams batting "
            "first vs second, average first innings score, and typical winning margins at this venue.</p>",
            unsafe_allow_html=True)
        render_venue_bias_card(city)

    with tab2:
        st.markdown(
            "<p style='color:#a08c50;font-size:13px;'>Quantifies how much winning the toss "
            "has historically influenced match outcomes at this venue.</p>",
            unsafe_allow_html=True)
        render_toss_impact(city)

    with tab3:
        st.markdown(
            "<p style='color:#a08c50;font-size:13px;'>Compares the current target against "
            "the historical average first innings score for this venue.</p>",
            unsafe_allow_html=True)
        render_target_difficulty(city, target)

    with tab4:
        st.markdown(
            "<p style='color:#a08c50;font-size:13px;'>Direct match history between the "
            "two selected teams specifically at this venue.</p>",
            unsafe_allow_html=True)
        render_h2h_venue(batting_team, bowling_team, city)
