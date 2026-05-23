import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime
# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="CricScope | IPL Analytics", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "prob_history" not in st.session_state:
    st.session_state.prob_history = []

# -----------------------------------
# STADIUM NIGHT THEME CSS
# -----------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond...');

/* ... all your existing CSS ... */

/* ========== ADD THIS AT THE END OF YOUR CSS ========== */
/* Column Card Styles for Fan Pulse Arena */
.column-card {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.column-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(251,191,36,0.2);
}

/* Optional: Add responsive behavior */
@media (max-width: 768px) {
    .column-card {
        margin-bottom: 20px;
    }
}
</style>

<!-- Floating Cricket Balls -->
<div class="cricket-ball-1"></div>
...
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ---- STADIUM NIGHT PALETTE ---- */
:root {
    --stadium-bg: #020617;
    --stadium-navy: #0f172a;
    --stadium-blue: #1e293b;
    --floodlight-gold: #fbbf24;
    --floodlight-amber: #f59e0b;
    --pitch-green: #10b981;
    --pitch-dark: #059669;
    --cricket-red: #dc2626;
    --cricket-leather: #991b1b;
    --light-beam: rgba(251, 191, 36, 0.15);
    --glass-bg: rgba(15, 23, 42, 0.6);
    --glass-border: rgba(251, 191, 36, 0.15);
}

/* ---- BASE & RESET ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
    position: relative;
    overflow-x: hidden;
    min-height: 100vh;
}

/* ---- FLOODLIGHT BEAMS ---- */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -20%;
    left: -10%;
    width: 50%;
    height: 80%;
    background: radial-gradient(ellipse at center, rgba(251, 191, 36, 0.12) 0%, transparent 70%);
    transform: rotate(-20deg);
    pointer-events: none;
    z-index: 0;
    filter: blur(40px);
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: -20%;
    right: -10%;
    width: 50%;
    height: 80%;
    background: radial-gradient(ellipse at center, rgba(251, 191, 36, 0.12) 0%, transparent 70%);
    transform: rotate(20deg);
    pointer-events: none;
    z-index: 0;
    filter: blur(40px);
}

/* ---- FLOATING CRICKET BALLS ---- */
.cricket-ball-1, .cricket-ball-2, .cricket-ball-3 {
    position: fixed;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #dc2626, #7f1d1d);
    box-shadow: 0 0 40px rgba(220, 38, 38, 0.2), inset -5px -5px 15px rgba(0,0,0,0.4);
    z-index: 0;
    opacity: 0.08;
    pointer-events: none;
}

.cricket-ball-1 {
    width: 80px;
    height: 80px;
    top: 20%;
    left: 5%;
    animation: float-ball-1 8s ease-in-out infinite;
}

.cricket-ball-2 {
    width: 60px;
    height: 60px;
    top: 60%;
    right: 8%;
    animation: float-ball-2 10s ease-in-out infinite;
    background: radial-gradient(circle at 30% 30%, #fbbf24, #d97706);
    box-shadow: 0 0 40px rgba(251, 191, 36, 0.2);
}

.cricket-ball-3 {
    width: 40px;
    height: 40px;
    bottom: 20%;
    left: 15%;
    animation: float-ball-3 7s ease-in-out infinite;
}

@keyframes float-ball-1 {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-30px) rotate(180deg); }
}

@keyframes float-ball-2 {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-40px) rotate(-180deg); }
}

@keyframes float-ball-3 {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(90deg); }
}

/* ---- PITCH GRASS OVERLAY ---- */
.pitch-overlay {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 30%;
    background: linear-gradient(to top, rgba(16, 185, 129, 0.03), transparent);
    pointer-events: none;
    z-index: 0;
}

/* ---- HIDE STREAMLIT BRANDING ---- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid var(--glass-border);
    width: 280px !important;
    position: relative;
    z-index: 10;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
    background: transparent;
}

/* Sidebar floodlight effect */
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 300px;
    background: radial-gradient(ellipse at 50% 0%, rgba(251, 191, 36, 0.1) 0%, transparent 70%);
    pointer-events: none;
}

.sidebar-brand {
    padding: clamp(24px, 5vw, 40px) clamp(20px, 4vw, 28px) clamp(20px, 4vw, 28px);
    border-bottom: 1px solid rgba(251, 191, 36, 0.15);
    margin-bottom: 20px;
    position: relative;
}

/* FIX 1: Prevent logo text from wrapping */
.sidebar-logo-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(20px, 3.5vw, 28px);
    font-weight: 700;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    margin-bottom: 6px;
    text-shadow: 0 0 30px rgba(251, 191, 36, 0.3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.sidebar-tagline {
    font-size: clamp(9px, 2vw, 11px);
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(251, 191, 36, 0.5);
    font-weight: 500;
}

.sidebar-section-label {
    font-size: clamp(9px, 1.5vw, 10px);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.5);
    padding: clamp(12px, 2vw, 16px) clamp(20px, 4vw, 28px) 8px;
    font-weight: 600;
}

/* ---- NAV BUTTONS ---- */
.stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 0;
    color: rgba(226, 232, 240, 0.7);
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(12px, 2vw, 14px);
    font-weight: 500;
    letter-spacing: 0.5px;
    padding: clamp(12px, 2vw, 14px) clamp(20px, 3vw, 28px);
    height: auto;
    min-height: 48px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    border-left: 3px solid transparent;
}

.stButton > button:hover {
    background: rgba(251, 191, 36, 0.08);
    color: #fbbf24;
    border-left: 3px solid #fbbf24;
    padding-left: 32px;
}

.stButton > button:active,
.stButton > button:focus {
    background: rgba(251, 191, 36, 0.12);
    color: #fbbf24;
    border-left: 3px solid #fbbf24;
    outline: none;
}

/* ---- MAIN CONTENT ---- */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    position: relative;
    z-index: 1;
}

/* ---- HERO SECTION ---- */
.hero-wrapper {
    padding: clamp(40px, 8vw, 80px) clamp(20px, 8vw, 60px) clamp(32px, 6vw, 48px);
    border-bottom: 1px solid rgba(251, 191, 36, 0.1);
    position: relative;
    overflow: hidden;
    background: radial-gradient(ellipse at 50% 0%, rgba(251, 191, 36, 0.05) 0%, transparent 50%);
}

.hero-eyebrow {
    font-size: clamp(10px, 2vw, 12px);
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(251, 191, 36, 0.7);
    margin-bottom: clamp(16px, 3vw, 24px);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: clamp(10px, 2vw, 12px);
    color: #10b981;
    letter-spacing: 1px;
    margin-bottom: clamp(20px, 4vw, 28px);
    backdrop-filter: blur(10px);
}

.hero-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px #10b981; }
    50% { opacity: 0.6; transform: scale(0.8); box-shadow: 0 0 20px #10b981; }
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(40px, 10vw, 96px);
    font-weight: 700;
    line-height: 0.9;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #ffffff 0%, #fbbf24 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: clamp(16px, 3vw, 24px);
    text-shadow: 0 0 60px rgba(251, 191, 36, 0.3);
}

.hero-subtitle {
    font-size: clamp(14px, 3vw, 18px);
    color: rgba(226, 232, 240, 0.6);
    font-weight: 400;
    letter-spacing: 0.5px;
    max-width: 520px;
    line-height: 1.7;
}

/* ---- STAT PILLS ---- */
.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: clamp(12px, 3vw, 20px);
    padding: clamp(24px, 4vw, 32px) clamp(20px, 8vw, 60px);
    border-bottom: 1px solid rgba(251, 191, 36, 0.08);
    background: linear-gradient(90deg, transparent, rgba(251, 191, 36, 0.03), transparent);
}

.stat-pill {
    flex: 1;
    min-width: clamp(120px, 20vw, 160px);
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 16px;
    padding: clamp(16px, 3vw, 24px) clamp(16px, 3vw, 24px);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.stat-pill::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--floodlight-gold), transparent);
    opacity: 0.5;
}

.stat-pill:hover {
    transform: translateY(-4px);
    border-color: rgba(251, 191, 36, 0.4);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(251, 191, 36, 0.1);
}

.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: clamp(24px, 4vw, 32px);
    font-weight: 600;
    color: #fbbf24;
    line-height: 1;
    margin-bottom: 8px;
    text-shadow: 0 0 20px rgba(251, 191, 36, 0.3);
}

.stat-label {
    font-size: clamp(9px, 1.5vw, 11px);
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.6);
    font-weight: 500;
}

/* ---- SECTION HEADERS ---- */
.section-header {
    padding: clamp(32px, 6vw, 48px) clamp(20px, 8vw, 60px) 0;
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(28px, 6vw, 40px);
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: 1px;
    margin-bottom: 8px;
    position: relative;
    display: inline-block;
}

.section-title::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #fbbf24, transparent);
    border-radius: 3px;
}

.section-desc {
    font-size: clamp(12px, 2vw, 14px);
    color: rgba(148, 163, 184, 0.7);
    letter-spacing: 0.5px;
    margin-top: 16px;
}

/* ---- INPUT CARDS ---- */
.input-card {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 24px;
    padding: clamp(24px, 4vw, 32px);
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.input-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(251, 191, 36, 0.3), transparent);
}

.input-card:hover {
    border-color: rgba(251, 191, 36, 0.3);
    box-shadow: 0 0 40px rgba(251, 191, 36, 0.05);
}

.input-label {
    font-size: clamp(10px, 1.5vw, 12px);
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(251, 191, 36, 0.8);
    margin-bottom: clamp(16px, 2vw, 20px);
    font-weight: 600;
}

/* ---- STREAMLIT INPUT OVERRIDES ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(251, 191, 36, 0.2) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: clamp(10px, 1.5vw, 11px) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(251, 191, 36, 0.7) !important;
    font-weight: 600 !important;
}

/* Slider track */
.stSlider [data-testid="stSlider"] > div {
    background: rgba(251, 191, 36, 0.2) !important;
    height: 6px !important;
    border-radius: 3px !important;
}

.stSlider [data-testid="stSlider"] > div > div {
    background: linear-gradient(90deg, #f59e0b, #fbbf24) !important;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.5) !important;
}

/* ---- TEAM VS CARD ---- */
.team-vs-wrapper {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 32px;
    padding: clamp(28px, 5vw, 48px) clamp(24px, 4vw, 40px);
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.team-vs-wrapper::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(251, 191, 36, 0.08) 0%, transparent 60%);
    pointer-events: none;
}

.vs-divider {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(48px, 10vw, 72px);
    font-weight: 300;
    color: rgba(251, 191, 36, 0.3);
    line-height: 1;
    letter-spacing: -4px;
    text-shadow: 0 0 30px rgba(251, 191, 36, 0.2);
}

.team-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(20px, 4vw, 28px);
    font-weight: 700;
    letter-spacing: 4px;
    margin-top: 20px;
    color: #f8fafc;
}

/* ---- ANALYZE BUTTON ---- */
.stButton.analyze-btn > button {
    background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 50%, #f59e0b 100%);
    color: #0f172a;
    border: none;
    border-radius: 16px;
    height: clamp(56px, 10vw, 64px);
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(13px, 2vw, 15px);
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    transition: all 0.4s ease;
    box-shadow: 0 10px 40px rgba(245, 158, 11, 0.3), 0 0 0 0 rgba(251, 191, 36, 0);
    width: 100%;
    position: relative;
    overflow: hidden;
}

.stButton.analyze-btn > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.stButton.analyze-btn > button:hover {
    box-shadow: 0 15px 50px rgba(245, 158, 11, 0.5), 0 0 60px rgba(251, 191, 36, 0.2);
    transform: translateY(-3px);
    filter: brightness(1.1);
}

.stButton.analyze-btn > button:hover::before {
    left: 100%;
}

/* ---- PREDICTION CARDS ---- */
.prediction-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-radius: 28px;
    padding: clamp(28px, 5vw, 40px);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #fbbf24, transparent);
    box-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
}

.prediction-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(251, 191, 36, 0.1) 0%, transparent 60%);
    pointer-events: none;
}

.prediction-label {
    font-size: clamp(10px, 1.5vw, 11px);
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(251, 191, 36, 0.6);
    margin-bottom: clamp(20px, 4vw, 28px);
    font-weight: 600;
}

.win-team-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(20px, 5vw, 32px);
    font-weight: 600;
    color: #f8fafc;
    line-height: 1.2;
    margin-bottom: 12px;
}

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: clamp(56px, 12vw, 80px);
    font-weight: 600;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 8px;
    text-shadow: 0 0 40px rgba(251, 191, 36, 0.3);
}

.win-prob-label {
    font-size: clamp(10px, 1.5vw, 11px);
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.5);
    margin-bottom: clamp(24px, 4vw, 32px);
    font-weight: 500;
}

/* ---- PROGRESS BAR ---- */
.prob-bar-track {
    height: 8px;
    background: rgba(30, 41, 59, 0.8);
    border-radius: 100px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}

.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #fcd34d);
    transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 0 20px rgba(251, 191, 36, 0.6);
    position: relative;
    overflow: hidden;
}

.prob-bar-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.prob-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 12px;
    font-size: clamp(11px, 2vw, 12px);
    color: rgba(148, 163, 184, 0.6);
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
}

/* ---- METRICS ROW ---- */
.metrics-row {
    display: flex;
    gap: clamp(10px, 2vw, 14px);
    margin-top: clamp(20px, 3vw, 24px);
}

.metric-chip {
    flex: 1;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 16px;
    padding: clamp(12px, 2vw, 16px);
    text-align: center;
    min-height: 60px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: all 0.3s ease;
}

.metric-chip:hover {
    border-color: rgba(251, 191, 36, 0.3);
    transform: translateY(-2px);
}

.metric-chip-value {
    font-family: 'DM Mono', monospace;
    font-size: clamp(18px, 3vw, 20px);
    color: #fbbf24;
    font-weight: 600;
    margin-bottom: 6px;
}

.metric-chip-label {
    font-size: clamp(8px, 1.5vw, 9px);
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.5);
    font-weight: 600;
}

/* ---- STRAY COMPONENTS ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #f59e0b, #fbbf24) !important;
    border-radius: 100px !important;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.5) !important;
}

.stProgress > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 100px !important;
    height: 8px !important;
}

div[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 16px;
    padding: clamp(16px, 3vw, 20px);
    backdrop-filter: blur(10px);
}

div[data-testid="metric-container"] label {
    color: rgba(251, 191, 36, 0.7) !important;
    font-size: clamp(10px, 1.5vw, 11px) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #fbbf24 !important;
    font-size: clamp(24px, 5vw, 32px) !important;
    font-weight: 600 !important;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: rgba(251, 191, 36, 0.3); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(251, 191, 36, 0.5); }

/* ---- MAIN PADDING ---- */
.main-pad {
    padding: 0 clamp(20px, 8vw, 60px) clamp(40px, 8vw, 80px);
}

/* ---- SEPARATOR ---- */
hr {
    border: none;
    border-top: 1px solid rgba(251, 191, 36, 0.1);
    margin: 0;
}

/* ---- MOBILE RESPONSIVE ---- */
@media (max-width: 768px) {
    .hero-wrapper {
        padding: clamp(32px, 6vw, 48px) clamp(16px, 6vw, 32px);
    }
    
    .hero-title {
        font-size: clamp(36px, 12vw, 64px);
    }
    
    .stats-row {
        padding: clamp(16px, 3vw, 24px) clamp(16px, 6vw, 32px);
    }
    
    .stat-pill {
        min-width: calc(50% - 10px);
        flex: 1 1 calc(50% - 10px);
    }
    
    section[data-testid="stSidebar"] {
        width: 260px !important;
    }
}

@media (max-width: 480px) {
    .stat-pill {
        min-width: 100%;
        flex: 1 1 100%;
    }
    
    .metrics-row {
        flex-direction: column;
    }
    
    .metric-chip {
        min-width: 100%;
    }
}

/* ---- ANIMATED BACKGROUND ELEMENTS ---- */
.stadium-light-left, .stadium-light-right {
    position: fixed;
    width: 300px;
    height: 600px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

.stadium-light-left {
    top: -100px;
    left: -100px;
    background: radial-gradient(ellipse at center, rgba(251, 191, 36, 0.15) 0%, transparent 70%);
    transform: rotate(-30deg);
    filter: blur(60px);
}

.stadium-light-right {
    top: -100px;
    right: -100px;
    background: radial-gradient(ellipse at center, rgba(251, 191, 36, 0.15) 0%, transparent 70%);
    transform: rotate(30deg);
    filter: blur(60px);
}

/* FIX 2: Team card abbreviation — prevent overflow truncation */
.team-card-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(14px, 2.5vw, 20px);
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: clamp(10px, 2vw, 16px);
    white-space: nowrap;
    overflow: visible;
}

/* FIX 3: Win rate card layout */
.wr-card {
    background: rgba(15,23,42,0.5);
    border: 1px solid rgba(251,191,36,0.1);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.wr-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}

.wr-logo-wrap {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    overflow: hidden;
    background: #0f172a;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.wr-logo-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    mix-blend-mode: screen;
}

.wr-info {
    flex: 1;
    min-width: 0;
}

.wr-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 1px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.wr-record {
    font-size: 10px;
    color: rgba(148,163,184,0.4);
    letter-spacing: 0.5px;
    white-space: nowrap;
}

.wr-pct {
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #fbbf24;
    flex-shrink: 0;
}

.wr-bar-track {
    height: 6px;
    background: rgba(30,41,59,0.8);
    border-radius: 100px;
    overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);
}
</style>

<!-- Floating Cricket Balls -->
<div class="cricket-ball-1"></div>
<div class="cricket-ball-2"></div>
<div class="cricket-ball-3"></div>

<!-- Stadium Lights -->
<div class="stadium-light-left"></div>
<div class="stadium-light-right"></div>

<!-- Pitch Overlay -->
<div class="pitch-overlay"></div>
""", unsafe_allow_html=True)

# -----------------------------------
# TEAM DATA
# FIX: Updated broken logo URLs for LSG, GT and others
# -----------------------------------
team_data = {
    "Chennai Super Kings": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/2/2b/Chennai_Super_Kings_Logo.svg",
        "abbr": "CSK", "color": "#facc15"
    },
    "Delhi Capitals": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/c/c2/Delhi_Capitals_Logo.png",
        "abbr": "DC", "color": "#3b82f6"
    },
    "Punjab Kings": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg",
        "abbr": "PBKS", "color": "#ef4444"
    },
    "Kolkata Knight Riders": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg",
        "abbr": "KKR", "color": "#7c3aed"
    },
    "Mumbai Indians": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/c/cd/Mumbai_Indians_Logo.svg",
        "abbr": "MI", "color": "#3b82f6"
    },
    "Rajasthan Royals": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg",
        "abbr": "RR", "color": "#ec4899"
    },
    "Royal Challengers Bangalore": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/2/2a/Royal_Challengers_Bangalore_2020.svg",
        "abbr": "RCB", "color": "#dc2626"
    },
    "Sunrisers Hyderabad": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad.svg",
        "abbr": "SRH", "color": "#f97316"
    },
    "Lucknow Super Giants": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/a/a6/Lucknow_Super_Giants_IPL_Logo.svg",
        "abbr": "LSG",
        "color": "#06b6d4"
    },
    "Gujarat Titans": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/0/09/Gujarat_Titans_Logo.svg",
        "abbr": "GT",
        "color": "#06b6d4"
    },
    "Deccan Chargers": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/7/70/DeccanChargersLogo.png",
        "abbr": "DC2",
        "color": "#2563eb"
    },
    "Pune Warriors India": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/4/46/Pune_Warriors_India_Logo.svg",
        "abbr": "PWI",
        "color": "#0f766e"
    },
    "Rising Pune Supergiant": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/f/f5/RisingPuneSupergiants.png",
        "abbr": "RPS",
        "color": "#7c3aed"
    },
    "Gujarat Lions": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/8/80/Gujarat_Lions.png",
        "abbr": "GL",
        "color": "#f59e0b"
    },
    "Kochi Tuskers Kerala": {
        "logo": "https://upload.wikimedia.org/wikipedia/en/7/7e/Kochi_Tuskers_Kerala_Logo.png",
        "abbr": "KTK",
        "color": "#10b981"
    }
}

# -----------------------------------
# WIN RATE STATS
# -----------------------------------
# ============================================
# NEW FUNCTIONS FOR FAN PULSE ARENA (ADD THESE)
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
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        import joblib
        
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
# -----------------------------------
# MODEL
# -----------------------------------
# ============================================
# ORIGINAL MODEL FOR MATCH ANALYSIS (KEEP THIS)
# ============================================
@st.cache_resource
def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    df['over'] = df['over'].replace(0, 0.1)

    df['crr'] = df['current_score'] / (df['over'] + df['ball'] / 6)
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
    ])

    pipe.fit(X, y)
    return pipe

pipe = train_model()

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    # FIX 1: Logo text on one line — reduced letter-spacing, nowrap enforced via CSS
    st.markdown("""
        <div class="sidebar-brand">
            <span class="sidebar-logo-text">CRICSCOPE</span>
            <span class="sidebar-tagline">Stadium Intelligence</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    if st.button("◈  Dashboard", key="nav_dash"):
        st.session_state.page = "Dashboard"

    if st.button("◉  Match Analysis", key="nav_analysis"):
        st.session_state.page = "Analysis"

    if st.button("🔥  Fan Pulse Arena", key="nav_fanpulse"):
        st.session_state.page = "FanPulse"


    st.markdown('<div style="height:1px; background:rgba(251,191,36,0.1); margin:20px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Built By</div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="padding:0 20px 10px;">
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(251,191,36,0.15);
                        border-radius:20px;padding:24px 20px 16px;position:relative;overflow:hidden;
                        backdrop-filter:blur(10px);">
                <div style="position:absolute;top:0;left:0;right:0;height:60px;
                            background:radial-gradient(ellipse at 50% 0%,rgba(251,191,36,0.1) 0%,transparent 70%);
                            pointer-events:none;"></div>
                <div style="width:48px;height:48px;border-radius:50%;
                            background:linear-gradient(135deg,#f59e0b,#fbbf24);
                            display:flex;align-items:center;justify-content:center;
                            font-size:18px;font-weight:700;color:#0f172a;
                            margin-bottom:14px;box-shadow:0 0 20px rgba(251,191,36,0.4);">AS</div>
                <div style="font-size:18px;font-weight:600;color:#f8fafc;
                            letter-spacing:0.5px;margin-bottom:4px;">Arnav Singh</div>
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:rgba(251,191,36,0.6);margin-bottom:20px;font-weight:600;">ML · Data · Analytics</div>
                <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(251,191,36,0.2),transparent);margin-bottom:14px;"></div>
                <div style="font-size:11px;color:rgba(226,232,240,0.5);line-height:1.6;">
                    Predictive analytics for modern cricket.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="padding:0 20px;">
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(251,191,36,0.15);
                        border-top:none;border-radius:0 0 20px 20px;padding:4px 12px 16px;">
                <p style="margin:0 0 4px 0;padding:10px 8px;">
                    <span style="color:rgba(251,191,36,0.8);margin-right:10px;font-size:13px;">✉</span>
                    <a href="mailto:itsarnav.singh80@gmail.com"
                       style="color:rgba(226,232,240,0.7);font-size:12px;text-decoration:none;letter-spacing:0.3px;">
                        itsarnav.singh80@gmail.com
                    </a>
                </p>
                <p style="margin:0 0 4px 0;padding:10px 8px;">
                    <span style="color:rgba(251,191,36,0.8);margin-right:10px;font-size:13px;">in</span>
                    <a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank"
                       style="color:rgba(226,232,240,0.7);font-size:12px;text-decoration:none;letter-spacing:0.3px;">
                        linkedin.com/in/arnav-singh
                    </a>
                </p>
                <p style="margin:0;padding:10px 8px;">
                    <span style="color:rgba(251,191,36,0.8);margin-right:10px;font-size:13px;">◆</span>
                    <a href="https://github.com/Arnav-Singh-5080" target="_blank"
                       style="color:rgba(226,232,240,0.7);font-size:12px;text-decoration:none;letter-spacing:0.3px;">
                        Arnav-Singh-5080
                    </a>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center;margin-top:20px;padding-bottom:28px;font-size:10px;
                    letter-spacing:2px;text-transform:uppercase;color:rgba(148,163,184,0.3);font-weight:500;">
            CricScope v2.0 · IPL Edition
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# DASHBOARD PAGE
# -----------------------------------
if st.session_state.page == "Dashboard":

    st.markdown("""
        <div class="hero-wrapper">
            <div class="hero-eyebrow">
                <span style="color:#10b981;">●</span> IPL Match Intelligence · Season 2025
            </div>
            <div class="hero-badge">
                <div class="hero-dot"></div>
                Live Predictions Active
            </div>
            <div class="hero-title">CricScope</div>
            <div class="hero-subtitle">
                Precision match analytics engineered for modern cricket.
                Real-time win probability powered by machine learning 
                under the stadium lights.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stats-row">
            <div class="stat-pill">
                <div class="stat-value">15</div>
                <div class="stat-label">IPL Teams</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">ML</div>
                <div class="stat-label">XGBoost Engine</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">120</div>
                <div class="stat-label">Balls Tracked</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">6+</div>
                <div class="stat-label">Key Signals</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section-header">
            <div class="section-title">Franchise Overview</div>
            <div class="section-desc">All-time performance metrics and team statistics</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    
    # FIX 2: Team cards — full abbreviation visible, no truncation
    team_cols = st.columns(5)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        color = tdata['color']
        logo  = tdata['logo']
        abbr  = tdata['abbr']
        with team_cols[i % 5]:
            st.markdown(f"""
                <div style="
                    background:rgba(15,23,42,0.5);
                    border:1px solid rgba(251,191,36,0.1);
                    border-radius:20px;
                    padding:20px 12px;
                    text-align:center;
                    transition:all 0.3s ease;
                    margin-bottom:16px;
                    backdrop-filter:blur(10px);
                    overflow:visible;
                ">
                    <div style="width:64px;height:64px;border-radius:50%;margin:0 auto;
                                overflow:hidden;background:#0f172a;
                                box-shadow:0 0 30px {color}40,0 0 0 1px rgba(251,191,36,0.1);
                                display:flex;align-items:center;justify-content:center;">
                        <img src="{logo}"
                             style="width:100%;height:100%;object-fit:cover;mix-blend-mode:screen;border-radius:50%;"
                             onerror="this.style.display='none';" />
                    </div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:18px;font-weight:700;
                                color:{color};letter-spacing:2px;margin-top:14px;
                                white-space:nowrap;overflow:visible;">
                        {abbr}
                    </div>
                    <div style="font-size:10px;color:rgba(148,163,184,0.45);margin-top:5px;
                                letter-spacing:0.3px;font-weight:500;word-break:break-word;line-height:1.4;">
                        {team_name}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ---- WIN RATE STATS SECTION ----
    st.markdown("""
        <div style="margin-top:48px;">
            <div style="font-family:'Cormorant Garamond',serif;font-size:clamp(24px,5vw,32px);font-weight:600;
                        color:#f8fafc;letter-spacing:1px;margin-bottom:32px;position:relative;display:inline-block;">
                All-Time Win Rates
                <div style="position:absolute;bottom:-8px;left:0;width:50px;height:3px;background:linear-gradient(90deg,#fbbf24,transparent);border-radius:3px;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # FIX 3: Win rate cards — clean layout with correct stats display
    wr_cols = st.columns(5)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        s = win_stats.get(team_name, {"wins": 0, "total": 0, "rate": 0})
        bar_pct = min(s["rate"], 100)
        with wr_cols[i % 5]:
            st.markdown(f"""
                <div class="wr-card">
                    <div class="wr-card-header">
                        <div class="wr-logo-wrap" style="box-shadow:0 0 15px {tdata['color']}40;">
                            <img src="{tdata['logo']}"
                                 onerror="this.style.display='none';"
                                 style="width:100%;height:100%;object-fit:cover;mix-blend-mode:screen;" />
                        </div>
                        <div class="wr-info">
                            <div class="wr-abbr" style="color:{tdata['color']};">{tdata['abbr']}</div>
                            <div class="wr-record">{s['wins']}W / {s['total']}M</div>
                        </div>
                        <div class="wr-pct">{bar_pct}%</div>
                    </div>
                    <div class="wr-bar-track">
                        <div style="height:100%;width:{bar_pct}%;border-radius:100px;
                                    background:linear-gradient(90deg,{tdata['color']},{tdata['color']}99);
                                    box-shadow:0 0 10px {tdata['color']}80;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-top:48px;text-align:center;">
            <div style="display:inline-block;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                        border-radius:16px;padding:24px 40px;">
                <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                            color:#10b981;margin-bottom:10px;font-weight:600;">Get Started</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:24px;color:#f8fafc;font-weight:500;">
                    Navigate to Match Analysis →
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Analysis":

    st.markdown("""
        <div class="hero-wrapper" style="padding-bottom:clamp(24px,5vw,40px);">
            <div class="hero-eyebrow">Win Probability Engine</div>
            <div class="hero-title" style="font-size:clamp(36px,8vw,72px);margin-bottom:clamp(12px,2vw,16px);">
                Match Analysis
            </div>
            <div class="hero-subtitle">
                Configure the match state below to compute real-time win probabilities 
                powered by our XGBoost model.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div style="height:clamp(24px,4vw,40px);"></div>', unsafe_allow_html=True)

    teams = list(team_data.keys())

    # ---- INPUT SECTION ----
    st.markdown("""
        <div style="font-size:clamp(11px,1.5vw,12px);letter-spacing:3px;text-transform:uppercase;
                    color:rgba(251,191,36,0.8);margin-bottom:clamp(16px,3vw,24px);font-weight:600;">
            Match Configuration
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Select Teams</div>', unsafe_allow_html=True)
        batting_team = st.selectbox("Batting Team", teams, key="bat")
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], key="bowl")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Match State</div>', unsafe_allow_html=True)
        target = st.number_input("Target Score", min_value=50, max_value=300, value=180, step=1)
        score = st.number_input("Current Score", min_value=0, max_value=target - 1, value=50, step=1)
        col_ov, col_wk = st.columns(2)
        with col_ov:
            overs = st.slider("Overs Completed", min_value=1, max_value=19, value=10)
        with col_wk:
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:clamp(24px,4vw,32px);"></div>', unsafe_allow_html=True)

    # ---- TEAM VS DISPLAY ----
    t1 = team_data[batting_team]
    t2 = team_data[bowling_team]

    st.markdown("""
        <div style="font-size:clamp(11px,1.5vw,12px);letter-spacing:3px;text-transform:uppercase;
                    color:rgba(251,191,36,0.8);margin-bottom:clamp(16px,3vw,20px);font-weight:600;">
            Fixture
        </div>
    """, unsafe_allow_html=True)

    vs_col1, vs_col2, vs_col3 = st.columns([2, 1, 2])

    with vs_col1:
        st.markdown(f"""
            <div style="background:rgba(15,23,42,0.5);border:1px solid rgba(251,191,36,0.15);
                        border-radius:28px;padding:clamp(24px,5vw,36px);text-align:center;
                        box-shadow:0 0 60px {t1['color']}15,0 25px 50px rgba(0,0,0,0.3);
                        backdrop-filter:blur(20px);position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{t1['color']},transparent);opacity:0.5;"></div>
                <div style="width:100px;height:100px;border-radius:50%;margin:0 auto;
                            overflow:hidden;background:#0f172a;
                            box-shadow:0 0 40px {t1['color']}50;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t1['logo']}"
                         style="width:100%;height:100%;object-fit:cover;mix-blend-mode:screen;"
                         onerror="this.style.display='none';" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:clamp(22px,4vw,30px);font-weight:700;
                            color:{t1['color']};letter-spacing:3px;margin-top:18px;
                            text-shadow:0 0 30px {t1['color']}40;">
                    {t1['abbr']}
                </div>
                <div style="font-size:11px;color:rgba(148,163,184,0.5);margin-top:8px;letter-spacing:2px;font-weight:600;">
                    BATTING
                </div>
            </div>
        """, unsafe_allow_html=True)

    with vs_col2:
        st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;height:100%;
                        font-family:'Cormorant Garamond',serif;font-size:clamp(40px,8vw,64px);font-weight:300;
                        color:rgba(251,191,36,0.3);letter-spacing:-4px;padding:32px 0;">
                vs
            </div>
        """, unsafe_allow_html=True)

    with vs_col3:
        st.markdown(f"""
            <div style="background:rgba(15,23,42,0.5);border:1px solid rgba(251,191,36,0.15);
                        border-radius:28px;padding:clamp(24px,5vw,36px);text-align:center;
                        box-shadow:0 0 60px {t2['color']}15,0 25px 50px rgba(0,0,0,0.3);
                        backdrop-filter:blur(20px);position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{t2['color']},transparent);opacity:0.5;"></div>
                <div style="width:100px;height:100px;border-radius:50%;margin:0 auto;
                            overflow:hidden;background:#0f172a;
                            box-shadow:0 0 40px {t2['color']}50;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t2['logo']}"
                         style="width:100%;height:100%;object-fit:cover;mix-blend-mode:screen;"
                         onerror="this.style.display='none';" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:clamp(22px,4vw,30px);font-weight:700;
                            color:{t2['color']};letter-spacing:3px;margin-top:18px;
                            text-shadow:0 0 30px {t2['color']}40;">
                    {t2['abbr']}
                </div>
                <div style="font-size:11px;color:rgba(148,163,184,0.5);margin-top:8px;letter-spacing:2px;font-weight:600;">
                    BOWLING
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:clamp(24px,4vw,40px);"></div>', unsafe_allow_html=True)

    # ---- ANALYZE BUTTON ----
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    analyze = st.button("Run Analysis", key="analyze_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- PREDICTION OUTPUT ----
    if analyze:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # ---- MATCH-STATE VALIDATION (Issue #31) ----
        # Guard 1: batting team has already reached or crossed the target
        if runs_left <= 0:
            st.success(
                f"🏆 **Match Over** — **{batting_team}** have already reached the target! "
                f"No prediction needed."
            )
            st.stop()

        # Guard 2: no balls remaining — innings is over, batting team fell short
        if balls_left <= 0:
            st.error(
                f"⏱️ **Match Over** — No balls remaining. **{bowling_team}** win! "
                f"No prediction needed."
            )
            st.stop()

        # Guard 3: RRR physically impossible (> 36 runs/over = 6 runs every ball)
        if rrr > 36.0:
            st.error(
                f"⚠️ **Invalid match state** — Required Run Rate is **{round(rrr, 2)} runs/over**, "
                f"which is physically impossible in cricket. Please check your inputs."
            )
            st.stop()

        # Guard 4: RRR extremely high (> 24 runs/over) — warn but allow prediction
        if rrr > 24.0:
            st.warning(
                f"⚠️ **Extreme match state** — Required Run Rate is **{round(rrr, 2)} runs/over**. "
                f"This is a very unlikely scenario; the prediction below may be less reliable."
            )

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': ['Mumbai'],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        with st.spinner(""):
            time.sleep(0.4)
            proba = pipe.predict_proba(input_df)[0]

        win = proba[1]
        lose = proba[0]
        st.session_state.prob_history.append(round(win * 100, 2))

        st.markdown('<div style="height:clamp(24px,4vw,40px);"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:clamp(11px,1.5vw,12px);letter-spacing:3px;text-transform:uppercase;
                        color:rgba(251,191,36,0.8);margin-bottom:clamp(16px,3vw,20px);font-weight:600;">
                Prediction Output
            </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2, gap="large")

        with res_col1:
            bat_pct = round(win * 100)
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Batting Team · {t1['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:clamp(20px,4vw,28px);
                                font-weight:600;color:#f8fafc;margin-bottom:clamp(16px,3vw,20px);">
                        {batting_team}
                    </div>
                    <div class="win-probability">{bat_pct}%</div>
                    <div class="win-prob-label">Win Probability</div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width:{bat_pct}%;"></div>
                    </div>
                    <div class="prob-bar-labels">
                        <span>0%</span><span>{bat_pct}%</span><span>100%</span>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{score}</div>
                            <div class="metric-chip-label">Current</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{runs_left}</div>
                            <div class="metric-chip-label">Needed</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{balls_left}</div>
                            <div class="metric-chip-label">Balls</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            bowl_pct = round(lose * 100)
            st.markdown(f"""
                <div style="background:rgba(15,23,42,0.4);border:1px solid rgba(148,163,184,0.15);
                            border-radius:28px;padding:clamp(28px,5vw,40px);position:relative;overflow:hidden;
                            backdrop-filter:blur(20px);">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(148,163,184,0.3),transparent);"></div>
                    <div class="prediction-label">Bowling Team · {t2['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:clamp(20px,4vw,28px);
                                font-weight:600;color:#f8fafc;margin-bottom:clamp(16px,3vw,20px);">
                        {bowling_team}
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:clamp(48px,10vw,80px);font-weight:600;
                                color:rgba(148,163,184,0.6);line-height:1;margin-bottom:8px;">
                        {bowl_pct}%
                    </div>
                    <div class="win-prob-label">Win Probability</div>
                    <div class="prob-bar-track">
                        <div style="height:100%;border-radius:100px;
                                    background:rgba(148,163,184,0.3);
                                    width:{bowl_pct}%;transition:width 1s ease;"></div>
                    </div>
                    <div class="prob-bar-labels">
                        <span>0%</span><span>{bowl_pct}%</span><span>100%</span>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{round(crr, 2)}</div>
                            <div class="metric-chip-label">CRR</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{round(rrr, 2)}</div>
                            <div class="metric-chip-label">RRR</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{10 - wickets}</div>
                            <div class="metric-chip-label">Wickets</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ---- SUMMARY ROW ----
        st.markdown('<div style="height:clamp(16px,3vw,24px);"></div>', unsafe_allow_html=True)
        verdict = batting_team if win > 0.5 else bowling_team
        conf = max(win, lose)
        conf_color = "#10b981" if conf > 0.75 else "#fbbf24" if conf > 0.55 else "#f87171"
        conf_label = "High Confidence" if conf > 0.75 else "Moderate" if conf > 0.55 else "Close Match"

        st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(251,191,36,0.2);
                        border-radius:20px;padding:clamp(20px,4vw,28px) clamp(24px,4vw,32px);
                        backdrop-filter:blur(20px);position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,{conf_color},transparent);opacity:0.6;"></div>
                <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:20px;">
                    <div>
                        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                                    color:rgba(148,163,184,0.5);margin-bottom:8px;font-weight:600;">Model Verdict</div>
                        <div style="font-family:'Cormorant Garamond',serif;font-size:clamp(24px,5vw,32px);
                                    font-weight:600;color:#f8fafc;">
                            {verdict} <span style="color:{conf_color};">favoured</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                                    color:rgba(148,163,184,0.5);margin-bottom:8px;font-weight:600;">Confidence Level</div>
                        <div style="font-family:'DM Mono',monospace;font-size:clamp(20px,4vw,28px);color:{conf_color};font-weight:600;">
                            {conf_label} · {round(conf*100)}%
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if len(st.session_state.prob_history) > 1:
            st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

            fig = px.line(
                x=list(range(1, len(st.session_state.prob_history)+1)),
                y=st.session_state.prob_history,
                labels={'x': 'Analysis Point', 'y': 'Win Probability (%)'},
                title="Win Probability Progression"
            )

            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="DM Sans"),
                title_font_size=20,
                title_font_color="#fbbf24",
                xaxis_gridcolor="rgba(251,191,36,0.1)",
                yaxis_gridcolor="rgba(251,191,36,0.1)",
                xaxis_linecolor="rgba(251,191,36,0.2)",
                yaxis_linecolor="rgba(251,191,36,0.2)"
            )
            
            fig.update_traces(
                line_color="#fbbf24",
                line_width=3,
                marker_color="#f59e0b",
                marker_size=8,
                fill='tozeroy',
                fillcolor="rgba(251,191,36,0.1)"
            )

            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------
# PROFESSIONAL FAN PULSE ARENA PAGE
# -----------------------------------
elif st.session_state.page == "FanPulse":
    
    # ============================================
    # ADD CUSTOM CSS FOR GLASSMORPHISM & ANIMATIONS
    # ============================================
    st.markdown("""
    <style>
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border-radius: 28px;
        border: 1px solid rgba(251, 191, 36, 0.25);
        box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        padding: 24px;
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(251, 191, 36, 0.5);
        box-shadow: 0 25px 40px -12px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(251, 191, 36, 0.2);
    }
    
    /* Animated Mood Buttons */
    .mood-btn {
        transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        background: rgba(15, 23, 42, 0.6);
        border: none;
        border-radius: 60px;
        padding: 12px 0;
        font-size: 28px;
        text-align: center;
        width: 100%;
        transition: transform 0.15s, background 0.2s;
    }
    .mood-btn:hover {
        transform: scale(1.02);
        background: rgba(251, 191, 36, 0.15);
    }
    .mood-btn:active {
        transform: scale(0.96);
    }
    
    /* Glow Button for Predict */
    .glow-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        border: none;
        border-radius: 40px;
        padding: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #0f172a;
        transition: all 0.2s;
        box-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
        width: 100%;
        cursor: pointer;
    }
    .glow-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(251, 191, 36, 0.8);
    }
    .glow-btn:active {
        transform: translateY(1px);
    }
    
    /* Circular Gauge */
    .circular-gauge {
        width: 140px;
        height: 140px;
        margin: 0 auto;
        position: relative;
    }
    .circular-gauge svg {
        transform: rotate(-90deg);
    }
    .gauge-bg {
        stroke: #334155;
        stroke-width: 12;
        fill: none;
    }
    .gauge-fill {
        stroke: #fbbf24;
        stroke-width: 12;
        fill: none;
        stroke-linecap: round;
        transition: stroke-dasharray 0.8s ease;
    }
    .gauge-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #fbbf24;
    }
    
    /* Blinking live indicator */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.2; }
        100% { opacity: 1; }
    }
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #ef4444;
        border-radius: 50%;
        margin-left: 10px;
        animation: blink 1.2s infinite;
        vertical-align: middle;
    }
    
    /* Toast notification */
    .toast-success {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 10px 20px;
        border-radius: 40px;
        font-size: 14px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: fadeInUp 0.3s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Dark/Light mode toggle */
    .theme-toggle {
        background: rgba(30,41,59,0.6);
        border-radius: 40px;
        padding: 5px;
        display: inline-flex;
        cursor: pointer;
        transition: 0.2s;
    }
    .theme-option {
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 600;
        transition: 0.2s;
    }
    .theme-option.active {
        background: #fbbf24;
        color: #0f172a;
    }
    
    /* Responsive tweaks */
    @media (max-width: 768px) {
        .glass-card { padding: 16px; }
        .circular-gauge { width: 100px; height: 100px; }
        .gauge-text { font-size: 24px; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================
    # DARK/LIGHT MODE TOGGLE (Using session state)
    # ============================================
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"
    
    col_theme, col_empty = st.columns([1, 5])
    with col_theme:
        if st.button("🌙" if st.session_state.theme == "dark" else "☀️", key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    
    # Apply theme class to main container via markdown (optional)
    theme_bg = "#0f172a" if st.session_state.theme == "dark" else "#f1f5f9"
    theme_text = "#e2e8f0" if st.session_state.theme == "dark" else "#0f172a"
    st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background: {theme_bg};
        }}
        .glass-card {{
            background: { 'rgba(30,41,59,0.4)' if st.session_state.theme == "dark" else 'rgba(255,255,255,0.7)' };
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================
    # INITIALIZE SESSION STATES
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
    if 'toast_msg' not in st.session_state:
        st.session_state.toast_msg = None
    
    # ============================================
    # HELPER TO SHOW TOAST
    # ============================================
    def show_toast(message):
        st.session_state.toast_msg = message
    
    # ============================================
    # HEADER WITH LIVE INDICATOR
    # ============================================
    st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 30px; text-align: center;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 32px;">🔥</span>
                        <span style="font-size: 28px; font-weight: 700; color: {theme_text};">Fan Pulse Arena</span>
                        <span class="live-dot"></span>
                        <span style="font-size: 12px; color: #10b981;">LIVE</span>
                    </div>
                    <div style="font-size: 14px; color: #94a3b8; margin-top: 8px;">
                        Real‑time crowd sentiment · AI predictions
                    </div>
                </div>
                <div style="background: rgba(0,0,0,0.3); border-radius: 60px; padding: 5px 12px;">
                    <span style="font-size: 12px;">👥 {sum(st.session_state.mood_votes.values())} votes today</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Display toast if exists
    if st.session_state.toast_msg:
        st.markdown(f"""
            <div class="toast-success">{st.session_state.toast_msg}</div>
            <script>setTimeout(() => {{ document.querySelector('.toast-success')?.remove(); }}, 2000);</script>
        """, unsafe_allow_html=True)
        st.session_state.toast_msg = None
    
    # ============================================
    # 3-COLUMN LAYOUT WITH GLASS CARDS
    # ============================================
    col1, col2, col3 = st.columns(3, gap="large")
    
    # ---------- COLUMN 1: MOOD METER WITH CIRCULAR GAUGE ----------
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                <span style="font-size: 20px;">📊</span>
                <span style="font-size: 13px; letter-spacing: 2px; color: #fbbf24;">LIVE MOOD METER</span>
                <span class="live-dot"></span>
            </div>
        """, unsafe_allow_html=True)
        
        # Animated mood buttons
        btn_cols = st.columns(3)
        moods = ['😡', '😐', '🔥']
        mood_names = ['ANGRY', 'NEUTRAL', 'EXCITED']
        for i, (emoji, name) in enumerate(zip(moods, mood_names)):
            with btn_cols[i]:
                if st.button(f"{emoji}<br><span style='font-size:12px'>{name}</span>", key=f"mood_{name}", help=f"Click to vote {name}"):
                    st.session_state.mood_votes[emoji] += 1
                    show_toast(f"🎉 You voted {emoji}!")
                    st.rerun()
        
        total_votes = sum(st.session_state.mood_votes.values())
        
        # Circular gauge for dominant mood percentage
        if total_votes > 0:
            dominant = max(st.session_state.mood_votes, key=st.session_state.mood_votes.get)
            dom_pct = (st.session_state.mood_votes[dominant] / total_votes) * 100
            dom_emoji = dominant
            dom_text = {'😡':'ANGRY','😐':'NEUTRAL','🔥':'EXCITED'}[dominant]
            dom_color = {'😡':'#ef4444','😐':'#94a3b8','🔥':'#fbbf24'}[dominant]
            
            # SVG circular gauge
            radius = 50
            circumference = 2 * 3.14159 * radius
            stroke_dash = (dom_pct / 100) * circumference
            st.markdown(f"""
                <div class="circular-gauge">
                    <svg width="140" height="140" viewBox="0 0 120 120">
                        <circle class="gauge-bg" cx="60" cy="60" r="50"></circle>
                        <circle class="gauge-fill" cx="60" cy="60" r="50" stroke="{dom_color}"
                                stroke-dasharray="{stroke_dash} {circumference}"
                                stroke-dashoffset="0"></circle>
                    </svg>
                    <div class="gauge-text">{dom_emoji}<br><span style="font-size: 12px;">{dom_text}</span></div>
                </div>
            """, unsafe_allow_html=True)
            
            # Progress bars for all moods
            for emoji, name, color in zip(['😡','😐','🔥'], ['ANGRY','NEUTRAL','EXCITED'], ['#ef4444','#94a3b8','#fbbf24']):
                cnt = st.session_state.mood_votes[emoji]
                pct = (cnt/total_votes)*100 if total_votes else 0
                st.markdown(f"""
                    <div style="margin: 12px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px;">
                            <span>{emoji} {name}</span>
                            <span style="color:{color};">{pct:.0f}%</span>
                        </div>
                        <div style="background:#334155; border-radius:20px; height:6px; overflow:hidden;">
                            <div style="width:{pct}%; background:{color}; height:100%; border-radius:20px; transition:width 0.5s;"></div>
                        </div>
                        <div style="font-size:10px; color:#64748b;">{cnt} votes</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click any mood to start voting!")
        
        st.markdown(f'<div style="text-align:center; margin-top:15px;"><span style="font-size:28px;">{total_votes}</span><br><span style="font-size:11px;">TOTAL VOTES</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------- COLUMN 2: ROLE POLLS (Enhanced) ----------
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <span style="font-size: 20px;">🏆</span>
                <span style="font-size: 13px; letter-spacing: 2px; color: #fbbf24;">ROLE POLLS</span>
            </div>
        """, unsafe_allow_html=True)
        
        poll_type = st.selectbox(
            "Select Category",
            ["Best Powerplay Opener", "Most Clutch Finisher", "Best Death Bowler", "MVP of the Tournament"],
            key="poll_type_select",
            label_visibility="collapsed"
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
            st.session_state.poll_votes = {p:0 for p in players}
        
        st.markdown('<div style="margin:15px 0 10px 0; font-size:12px; color:#cbd5e1;">🗳️ CAST YOUR VOTE</div>', unsafe_allow_html=True)
        
        # 2x2 grid for vote buttons with animations
        for i in range(0, len(players), 2):
            cols = st.columns(2)
            for j, player in enumerate(players[i:i+2]):
                with cols[j]:
                    if st.button(f"⭐ {player}", key=f"poll_{player}", use_container_width=True):
                        st.session_state.poll_votes[player] = st.session_state.poll_votes.get(player,0)+1
                        show_toast(f"✅ Voted for {player}!")
                        st.rerun()
        
        st.markdown('<div style="margin:20px 0 10px 0; font-size:12px; color:#cbd5e1;">📊 LIVE RESULTS</div>', unsafe_allow_html=True)
        if st.session_state.poll_votes and sum(st.session_state.poll_votes.values())>0:
            df_votes = pd.DataFrame(list(st.session_state.poll_votes.items()), columns=['Player','Votes']).sort_values('Votes', ascending=True)
            st.bar_chart(df_votes.set_index('Player'), height=250)
            winner = max(st.session_state.poll_votes, key=st.session_state.poll_votes.get)
            winner_votes = st.session_state.poll_votes[winner]
            st.markdown(f"""
                <div style="background:#fbbf2410; border-radius:16px; padding:12px; margin-top:12px; text-align:center;">
                    <div style="font-size:11px; color:#94a3b8;">CURRENT LEADER</div>
                    <div style="font-size:18px; font-weight:700; color:#fbbf24;">🏆 {winner}</div>
                    <div style="font-size:12px;">{winner_votes} votes</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No votes yet.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------- COLUMN 3: AI SENTIMENT ENGINE (Upgraded) ----------
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <span style="font-size: 20px;">🤖</span>
                <span style="font-size: 13px; letter-spacing: 2px; color: #fbbf24;">AI SENTIMENT ENGINE</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Inputs in two columns
        a1, a2 = st.columns(2)
        with a1:
            over = st.number_input("Over", 1,20,10, key="ml_over")
            wickets = st.selectbox("Wickets", [0,1,2,3], key="ml_wickets")
            run_rate = st.number_input("CRR", 4.0,12.0,7.5,0.1, key="ml_crr")
        with a2:
            runs_in_over = st.number_input("Runs", 0,36,8, key="ml_runs")
            req_rate = st.number_input("RRR", 5.0,15.0,9.0,0.1, key="ml_rrr")
            time_of_day = st.selectbox("Time", ["morning","afternoon","evening","night"], key="ml_time")
        
        # Load or train model
        import os
        model_files_exist = all(os.path.exists(f) for f in ['mood_predictor.pkl', 'le_time.pkl', 'le_importance.pkl', 'le_mood.pkl'])
        if not model_files_exist or not st.session_state.model_trained:
            with st.spinner("Training XGBoost model..."):
                df = generate_training_data()
                model, le_time, le_importance, le_mood, accuracy = train_mood_predictor(df)
                st.session_state.model_trained = True
                st.session_state.ml_accuracy = accuracy
                st.session_state.encoders = (model, le_time, le_importance, le_mood)
        else:
            import joblib
            model = joblib.load('mood_predictor.pkl')
            le_time = joblib.load('le_time.pkl')
            le_importance = joblib.load('le_importance.pkl')
            le_mood = joblib.load('le_mood.pkl')
            st.session_state.encoders = (model, le_time, le_importance, le_mood)
        
        # Glowing predict button
        if st.button("🔮 PREDICT SENTIMENT", key="predict_btn", use_container_width=True):
            mood, confidence = predict_current_mood(
                over, runs_in_over, wickets, run_rate, req_rate,
                time_of_day, "league", st.session_state.encoders
            )
            mood_emoji = {'angry':'😡','neutral':'😐','excited':'🔥'}.get(mood,'😐')
            mood_color = {'angry':'#ef4444','neutral':'#94a3b8','excited':'#fbbf24'}.get(mood,'#94a3b8')
            
            # Show animated result card
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, {mood_color}20, transparent);
                            border: 1px solid {mood_color}40;
                            border-radius: 20px;
                            padding: 18px;
                            margin-top: 16px;
                            text-align: center;
                            animation: fadeInUp 0.4s ease;">
                    <div style="font-size: 52px;">{mood_emoji}</div>
                    <div style="font-size: 24px; font-weight: 700; color: {mood_color};">{mood.upper()}</div>
                    <div style="font-size: 13px;">Confidence: {confidence:.1f}%</div>
                    <div style="background: #334155; border-radius: 20px; height: 4px; margin-top: 12px;">
                        <div style="width:{confidence}%; background:{mood_color}; height:100%; border-radius:20px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.session_state.ml_predictions.append({
                'timestamp': datetime.now(),
                'mood': mood,
                'confidence': confidence
            })
        
        if hasattr(st.session_state, 'ml_accuracy'):
            st.markdown(f"""
                <div style="margin-top: 15px; text-align: center; padding: 8px; background: #0f172a; border-radius: 20px;">
                    <span style="font-size: 11px;">Model Accuracy</span>
                    <span style="font-size: 14px; font-weight: 700; color: #10b981;"> {st.session_state.ml_accuracy:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # BOTTOM SECTION: ACTIVITY FEED & HISTORY
    # ============================================
    st.markdown("---")
    col_act, col_hist = st.columns(2, gap="large")
    
    with col_act:
        with st.container():
            st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
            st.markdown("🔄 **LIVE ACTIVITY FEED**")
            if st.button("🔄 Refresh", use_container_width=True):
                import random
                acts = [
                    f"⚡ {random.choice(['🔥','😡','😐'])} New mood vote!",
                    f"🗳️ Vote in {poll_type}",
                    f"📊 {total_votes} total fans engaged"
                ]
                for a in acts:
                    st.markdown(f'<div style="border-left: 2px solid #fbbf24; padding: 8px 12px; margin: 5px 0; background: rgba(0,0,0,0.2); border-radius: 12px;">{a}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_hist:
        with st.container():
            st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
            st.markdown("📜 **RECENT PREDICTIONS**")
            if st.session_state.ml_predictions:
                for pred in st.session_state.ml_predictions[-5:][::-1]:
                    emoji = {'angry':'😡','neutral':'😐','excited':'🔥'}.get(pred['mood'],'😐')
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 8px; margin: 5px 0; background: rgba(0,0,0,0.2); border-radius: 12px;">
                            <div><span style="font-size: 20px;">{emoji}</span> {pred['mood'].upper()}</div>
                            <div>{pred['confidence']:.0f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Click 'Predict Sentiment' above.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style="margin-top: 40px; padding: 20px; text-align: center; border-top: 1px solid rgba(251,191,36,0.1);">
            <span style="font-size: 11px; color: #64748b;">⚡ Real-time sync · 🤖 XGBoost engine · 🔥 Live reactions</span>
            <div style="font-size: 10px; margin-top: 8px;">© 2025 CricScope · GSSoC Contribution</div>
        </div>
    """, unsafe_allow_html=True)