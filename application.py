import streamlit as st
st.markdown("""
<style>

/* Sidebar full height */
section[data-testid="stSidebar"] {
    height: 100vh;
    background-color: #0f172a;
}

/* Main dashboard background */
.main {
    background-color: #020617;
    color: white;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)
import pandas as pd
import numpy as np
import time
import os
import joblib
import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="CricScope", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "logistic"

# -----------------------------------
# LUXURY CSS
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ---- RESET & BASE ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2dfd8;
}

[data-testid="stAppViewContainer"] {
    background: #080808;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%);
    min-height: 100vh;
}

/* Hide only Streamlit branding — leave header & sidebar toggle untouched */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #0c0c0c;
    border-right: 1px solid rgba(212,175,55,0.12);
    width: 300px !important;
    min-width: 300px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
}

.sidebar-brand {
    padding: 40px 32px 28px;
    border-bottom: 1px solid rgba(212,175,55,0.1);
    margin-bottom: 20px;
}

.sidebar-logo-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 600;
    letter-spacing: 3.5px;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    margin-bottom: 6px;
}

.sidebar-tagline {
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.45);
    font-weight: 400;
}

.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.2), transparent);
    margin: 8px 0;
}

.sidebar-section-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(180,160,100,0.35);
    padding: 14px 32px 8px;
    font-weight: 500;
}

/* ---- NAV BUTTONS ---- */
.stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 0;
    color: rgba(220,210,180,0.65);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 400;
    letter-spacing: 0.5px;
    padding: 13px 32px;
    height: auto;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    background: rgba(212,175,55,0.06);
    color: #d4af37;
    border: none;
    box-shadow: none;
}

.stButton > button:active,
.stButton > button:focus {
    background: rgba(212,175,55,0.1);
    color: #f0d060;
    border: none;
    box-shadow: none;
    outline: none;
}

/* ---- MAIN CONTENT AREA ---- */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ---- HERO SECTION ---- */
.hero-wrapper {
    padding: 64px 72px 40px;
    border-bottom: 1px solid rgba(212,175,55,0.08);
    position: relative;
    overflow: hidden;
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px; right: -60px;
    height: 200px;
    background: radial-gradient(ellipse, rgba(212,175,55,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    margin-bottom: 18px;
    font-weight: 400;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(52px, 7vw, 88px);
    font-weight: 600;
    line-height: 0.95;
    letter-spacing: -1px;
    background: linear-gradient(160deg, #ffffff 0%, #f8f0d0 30%, #d4af37 70%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 18px;
}

.hero-subtitle {
    font-size: 15px;
    color: rgba(220,210,185,0.55);
    font-weight: 300;
    letter-spacing: 0.3px;
    max-width: 460px;
    line-height: 1.6;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 100px;
    padding: 5px 14px 5px 10px;
    font-size: 11px;
    color: rgba(212,175,55,0.8);
    letter-spacing: 0.5px;
    margin-bottom: 24px;
    width: fit-content;
}

.hero-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #d4af37;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ---- STAT PILLS ---- */
.stats-row {
    display: flex;
    gap: 16px;
    padding: 24px 72px;
    border-bottom: 1px solid rgba(212,175,55,0.06);
}

.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 22px;
    transition: all 0.25s ease;
}

.stat-pill:hover {
    background: rgba(212,175,55,0.04);
    border-color: rgba(212,175,55,0.15);
    transform: translateY(-1px);
}

.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    color: #e8d89a;
    line-height: 1;
    margin-bottom: 6px;
}

.stat-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.4);
}

/* ---- ANALYSIS SECTION ---- */
.section-header {
    padding: 40px 72px 0;
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 500;
    color: #f0e8cc;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.section-desc {
    font-size: 13px;
    color: rgba(200,185,140,0.4);
    letter-spacing: 0.3px;
}

/* ---- INPUT CARD ---- */
.input-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.3s ease;
}

.input-card:hover {
    border-color: rgba(212,175,55,0.15);
}

.input-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    margin-bottom: 12px;
    font-weight: 500;
}

/* ---- STREAMLIT INPUT OVERRIDES ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
}

.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,0.5) !important;
    font-weight: 500 !important;
}

/* Slider track */
.stSlider [data-testid="stSlider"] > div {
    background: rgba(212,175,55,0.15) !important;
}

.stSlider [data-testid="stSlider"] > div > div {
    background: linear-gradient(90deg, #d4af37, #f0d060) !important;
}

/* ---- TEAM VS CARD ---- */
.team-vs-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 36px 28px;
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.team-vs-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(212,175,55,0.04) 0%, transparent 60%);
    pointer-events: none;
}

.team-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 3px;
    margin-top: 14px;
}

.vs-divider {
    font-family: 'Cormorant Garamond', serif;
    font-size: 48px;
    font-weight: 300;
    color: rgba(212,175,55,0.25);
    line-height: 1;
    letter-spacing: -2px;
}

.team-logo-glow {
    border-radius: 50%;
    transition: box-shadow 0.3s ease;
    width: 90px;
    height: 90px;
    object-fit: contain;
}

/* ---- ANALYZE BUTTON ---- */
.stButton.analyze-btn > button {
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 40%, #e8c84a 100%);
    color: #0a0800;
    border: none;
    border-radius: 14px;
    height: 52px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(212,175,55,0.2), 0 0 0 0 rgba(212,175,55,0);
    width: 100%;
}

.stButton.analyze-btn > button:hover {
    box-shadow: 0 12px 48px rgba(212,175,55,0.35), 0 0 60px rgba(212,175,55,0.1);
    transform: translateY(-2px);
    filter: brightness(1.05);
    color: #0a0800;
    border: none;
}

/* ---- PREDICTION CARD ---- */
.prediction-card {
    background: rgba(212,175,55,0.04);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 24px;
    padding: 36px 32px;
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -1px; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4af37, transparent);
}

.prediction-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(212,175,55,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.prediction-label {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.4);
    margin-bottom: 24px;
    font-weight: 500;
}

.win-team-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 38px;
    font-weight: 600;
    color: #f0e0a0;
    line-height: 1;
    margin-bottom: 8px;
}

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: 72px;
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
}

.win-prob-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.35);
    margin-bottom: 28px;
}

/* ---- PROGRESS BAR CUSTOM ---- */
.prob-bar-wrapper {
    position: relative;
    margin: 20px 0 14px;
}

.prob-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #b8962e, #d4af37, #f0d060);
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 0 12px rgba(212,175,55,0.4);
}

.prob-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: 11px;
    color: rgba(200,185,140,0.4);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}

/* ---- METRICS ROW ---- */
.metrics-row {
    display: flex;
    gap: 10px;
    margin-top: 18px;
}

.metric-chip {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
}

.metric-chip-value {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    color: #d4c080;
    font-weight: 500;
    margin-bottom: 4px;
}

.metric-chip-label {
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(180,165,115,0.35);
}

/* ---- STRAY STREAMLIT COMPONENTS ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #b8962e, #d4af37) !important;
    border-radius: 100px !important;
}

.stProgress > div {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 100px !important;
    height: 6px !important;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 20px;
}

div[data-testid="metric-container"] label {
    color: rgba(200,185,140,0.45) !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #e8d89a !important;
    font-size: 28px !important;
}

/* ---- SEPARATOR ---- */
hr {
    border: none;
    border-top: 1px solid rgba(212,175,55,0.08);
    margin: 0;
}

/* ---- CONTENT PADDING ---- */
.main-pad {
    padding: 0 72px 60px;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 4px; }

/* ============================================================
   SIDEBAR PROFILE SECTION - Premium Glassmorphism
   ============================================================ */

/* Kill Streamlit global link styles inside sidebar */
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] a:visited,
section[data-testid="stSidebar"] a:hover,
section[data-testid="stSidebar"] a:active {
    text-decoration: none !important;
    color: inherit !important;
}

/* Outer wrapper */
.profile-section {
    padding: 0 16px 10px;
}

/* ---- Profile identity card ---- */
.profile-card {
    width: 100%;
    box-sizing: border-box;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(212, 175, 55, 0.14);
    border-radius: 16px;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    padding: 22px 20px 18px;
    margin-bottom: 10px;
}

.profile-card:hover {
    border-color: rgba(212, 175, 55, 0.26);
    box-shadow: 0 6px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* Ambient top glow */
.profile-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 60px;
    background: radial-gradient(ellipse 90% 100% at 50% 0%, rgba(212,175,55,0.09) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Avatar - centered, 48px */
.profile-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 50%, #f0d060 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    font-weight: 700;
    color: #0a0800;
    letter-spacing: 0.5px;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.25), 0 0 20px rgba(212,175,55,0.25), 0 3px 12px rgba(0,0,0,0.4);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
    position: relative;
    z-index: 1;
    margin-bottom: 14px;
}

.profile-card:hover .profile-avatar {
    box-shadow: 0 0 0 2px rgba(212,175,55,0.45), 0 0 26px rgba(212,175,55,0.35), 0 3px 14px rgba(0,0,0,0.5);
    transform: scale(1.04);
}

/* Name */
.profile-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 19px;
    font-weight: 600;
    color: #f0e8cc;
    letter-spacing: 0.5px;
    line-height: 1.2;
    margin: 0 0 5px 0;
    position: relative;
    z-index: 1;
}

/* Role */
.profile-role {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(212, 175, 55, 0.45);
    font-weight: 500;
    line-height: 1;
    position: relative;
    z-index: 1;
}

/* ---- Contact card ---- */
.contact-card {
    width: 100%;
    box-sizing: border-box;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(212, 175, 55, 0.14);
    border-radius: 16px;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    padding: 8px 12px 12px;
}

.contact-card:hover {
    border-color: rgba(212, 175, 55, 0.22);
}

/* Each contact row */
.profile-link {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    text-decoration: none !important;
    padding: 9px 10px !important;
    border-radius: 9px !important;
    background: transparent !important;
    transition: background 0.2s ease, transform 0.2s ease !important;
    color: inherit !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

.profile-link:hover {
    background: rgba(212, 175, 55, 0.07) !important;
    transform: translateX(2px) !important;
    text-decoration: none !important;
}

/* Icon - plain, no badge box */
.profile-link-icon {
    font-size: 12px;
    color: rgba(212, 175, 55, 0.6);
    flex-shrink: 0;
    width: 14px;
    text-align: center;
    text-decoration: none !important;
    transition: color 0.2s ease;
}

.profile-link:hover .profile-link-icon {
    color: rgba(212, 175, 55, 0.9);
}

/* Link text */
.profile-link-text {
    font-size: 12px;
    color: rgba(200, 185, 140, 0.55);
    font-weight: 400;
    letter-spacing: 0.2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    transition: color 0.2s ease;
    flex: 1;
    min-width: 0;
    text-decoration: none !important;
}

.profile-link:hover .profile-link-text {
    color: rgba(212, 175, 55, 0.82);
}

/* Version footer */
.sidebar-version {
    text-align: center;
    padding: 16px 0 24px;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(200, 185, 140, 0.18);
    font-weight: 400;
    transition: color 0.3s ease;
}

.sidebar-version:hover {
    color: rgba(200, 185, 140, 0.3);
}

/* ---- PERFORMANCE REPORT & CONFUSION MATRIX ---- */
.matrix-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 32px;
    backdrop-filter: blur(20px);
}
.matrix-grid {
    display: grid;
    grid-template-columns: 120px 1fr 1fr;
    grid-gap: 16px;
    margin-top: 16px;
    align-items: center;
    text-align: center;
}
.matrix-header {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.6);
    font-weight: 500;
    padding: 10px 0;
}
.matrix-label {
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(220,210,185,0.4);
    text-align: left;
    font-weight: 500;
}
.matrix-cell {
    padding: 24px;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: all 0.25s ease;
}
.matrix-cell.correct {
    background: rgba(212,175,55,0.06);
    border: 1px solid rgba(212,175,55,0.25);
}
.matrix-cell.correct:hover {
    background: rgba(212,175,55,0.1);
    border-color: rgba(212,175,55,0.45);
    transform: translateY(-2px);
}
.matrix-cell.incorrect {
    background: rgba(214,40,40,0.03);
    border: 1px solid rgba(214,40,40,0.15);
}
.matrix-cell.incorrect:hover {
    background: rgba(214,40,40,0.06);
    border-color: rgba(214,40,40,0.3);
    transform: translateY(-2px);
}
.matrix-value {
    font-family: 'DM Mono', monospace;
    font-size: 32px;
    font-weight: 500;
    color: #f0e8cc;
}
.matrix-cell.incorrect .matrix-value {
    color: #e57373;
}
.matrix-cell-lbl {
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 6px;
    color: rgba(220,210,185,0.4);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TEAM DATA
# -----------------------------------
team_data = {
    "Chennai Super Kings": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/1-5.jpg",
        "abbr": "CSK", "color": "#facc15"
    },
    "Delhi Capitals": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/2-4.jpg",
        "abbr": "DC", "color": "#3b82f6"
    },
    "Punjab Kings": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/5-4.jpg",
        "abbr": "PBKS", "color": "#ef4444"
    },
    "Kolkata Knight Riders": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/3-4.jpg",
        "abbr": "KKR", "color": "#7c3aed"
    },
    "Mumbai Indians": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/4-4.jpg",
        "abbr": "MI", "color": "#3b82f6"
    },
    "Rajasthan Royals": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/6-4.jpg",
        "abbr": "RR", "color": "#ec4899"
    },
    "Royal Challengers Bangalore": {
        "logo": "https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/Untitled-4.jpg",
        "abbr": "RCB", "color": "#dc2626"
    },
    "Sunrisers Hyderabad": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/8-4.jpg",
        "abbr": "SRH", "color": "#f97316"
    }
}

# -----------------------------------
# MODEL
# -----------------------------------
def get_model(model_name='logistic'):
    if model_name == 'logistic':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'xgboost':
        return XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    return LogisticRegression(max_iter=1000)

@st.cache_resource
def train_model(model_name='logistic'):
    model_path = f"{model_name}_model.pkl"

    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            logging.error(f"Failed to load cached model from {model_path}: {e}")

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    
    balls_bowled = ((df['over'] - 1) * 6) + df['ball']
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    overs_bowled = (df['over'] - 1) + (df['ball'] / 6)
    df['crr'] = np.where(overs_bowled > 0, df['current_score'] / overs_bowled, 0.0)
    df['rrr'] = np.where(df['balls_left'] > 0, (df['runs_left'] * 6) / df['balls_left'], 0.0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', get_model(model_name))
    ])

    # Fit pipeline before evaluations to avoid UnboundLocalError
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    # Logging evaluations safely
    try:
        scores = cross_val_score(pipe, X_train, y_train, cv=5)
        logging.info(f"Model trained: {model_name}")
        logging.info(f"Cross Validation Scores: {scores}")
        logging.info(f"Average CV Accuracy: {scores.mean():.4f}")
        logging.info(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")
    except Exception as eval_error:
        logging.warning(f"Evaluation failed: {eval_error}")

    try:
        joblib.dump(pipe, model_path)
    except Exception as dump_error:
        logging.error(f"Failed to dump model to {model_path}: {dump_error}")

    return pipe

@st.cache_resource
def evaluate_model(model_name='logistic'):
    pipe = train_model(model_name)

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    
    balls_bowled = ((df['over'] - 1) * 6) + df['ball']
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    overs_bowled = (df['over'] - 1) + (df['ball'] / 6)
    df['crr'] = np.where(overs_bowled > 0, df['current_score'] / overs_bowled, 0.0)
    df['rrr'] = np.where(df['balls_left'] > 0, (df['runs_left'] * 6) / df['balls_left'], 0.0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    predictions = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    cv_mean = scores.mean()
    cv_std = scores.std()

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'cv_scores': scores.tolist()
    }

selected_model_key = st.session_state.get('selected_model', 'logistic')
pipe = train_model(selected_model_key)

def generate_ball_by_ball_df(pipe, batting_team, bowling_team, selected_city, target, score, overs, wickets):
    total_balls = int(overs * 6)
    if total_balls == 0:
        data = {
            'over': [0],
            'ball': [0],
            'batting_team_prob': [0.5],
            'bowling_team_prob': [0.5]
        }
        return pd.DataFrame(data)

    records = []
    for b in range(1, total_balls + 1):
        curr_over = (b - 1) // 6 + 1
        curr_ball = (b - 1) % 6 + 1
        
        fraction = b / total_balls
        curr_score = int(score * fraction)
        curr_wickets = int(wickets * fraction)
        
        runs_left = target - curr_score
        balls_left = max(120 - b, 0)
        crr = curr_score / (b / 6) if b > 0 else 0.0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
        
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - curr_wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })
        
        if runs_left <= 0:
            win = 1.0
            lose = 0.0
        elif balls_left <= 0:
            win = 0.0
            lose = 1.0
        else:
            try:
                proba = pipe.predict_proba(input_df)[0]
                if np.isnan(proba).any():
                    win, lose = 0.5, 0.5
                else:
                    win, lose = proba[1], proba[0]
            except Exception:
                win, lose = 0.5, 0.5
            
        records.append({
            'over': curr_over,
            'ball': curr_ball,
            'batting_team_prob': round(win, 4),
            'bowling_team_prob': round(lose, 4)
        })
        
    return pd.DataFrame(records)

def safe_calculate_rates(score, target, overs):
    runs_left = target - score
    balls_left = max(120 - (overs * 6), 0)
    crr = score / overs if overs > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
    return runs_left, balls_left, crr, rrr

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <span class="sidebar-logo-text">CRICSCOPE</span>
            <span class="sidebar-tagline">Match Intelligence Platform</span>
        </div>
    """, unsafe_allow_html=True)
    

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    if st.button("◈  Dashboard", key="nav_dash"):
        st.session_state.page = "Dashboard"

    if st.button("◉  Match Analysis", key="nav_analysis"):
        st.session_state.page = "Analysis"

    if st.button("⚖  Model Performance", key="nav_performance"):
        st.session_state.page = "Performance"

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Model Configuration</div>', unsafe_allow_html=True)

    model_options = {
        "Logistic Regression": "logistic",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }

    selected_model_name = st.selectbox(
        "Choose Prediction Model",
        options=list(model_options.keys()),
        index=list(model_options.values()).index(st.session_state.selected_model),
        key="selected_model_widget"
    )
    st.session_state.selected_model = model_options[selected_model_name]

    st.markdown('<div style="height:1px; background:rgba(212,175,55,0.08); margin:20px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Built By</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="profile-section">'
        '<div class="profile-card">'
        '<div class="profile-avatar">AS</div>'
        '<div class="profile-name">Arnav Singh</div>'
        '<div class="profile-role">ML &middot; Data &middot; Analytics</div>'
        '</div>'
        '<div class="contact-card">'
        '<a href="mailto:itsarnav.singh80@gmail.com" class="profile-link">'
        '<span class="profile-link-icon">&#9993;</span>'
        '<span class="profile-link-text">itsarnav.singh80@gmail.com</span>'
        '</a>'
        '<a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank" class="profile-link">'
        '<span class="profile-link-icon">in</span>'
        '<span class="profile-link-text">linkedin.com/in/arnav-singh</span>'
        '</a>'
        '<a href="https://github.com/Arnav-Singh-5080" target="_blank" class="profile-link">'
        '<span class="profile-link-icon">&#9670;</span>'
        '<span class="profile-link-text">Arnav-Singh-5080</span>'
        '</a>'
        '</div>'
        '</div>'
        '<div class="sidebar-version">CricScope v2.0 &middot; IPL Edition</div>',
        unsafe_allow_html=True
    )

# -----------------------------------
# DASHBOARD PAGE
# -----------------------------------
if st.session_state.page == "Dashboard":
    if "selected_team" not in st.session_state:
       st.session_state.selected_team = None

    st.markdown("""
        <div class="hero-wrapper">
            <div class="hero-eyebrow">IPL Match Intelligence · Season 2025</div>
            <div class="hero-badge">
                <div class="hero-dot"></div>
                Live Predictions Active
            </div>
            <div class="hero-title">CricScope</div>
            <div class="hero-subtitle">
                Precision match analytics engineered for modern cricket.
                Real-time win probability powered by machine learning.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stats-row">
            <div class="stat-pill">
                <div class="stat-value">8</div>
                <div class="stat-label">IPL Teams</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">ML</div>
                <div class="stat-label">Model Type</div>
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
        <div style="padding: 48px 72px;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:13px; letter-spacing:3px;
                        text-transform:uppercase; color:rgba(212,175,55,0.4); margin-bottom:28px;">
                IPL Teams
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:12px;">
    """, unsafe_allow_html=True)

    team_cols = st.columns(4)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        with team_cols[i % 4]:
            st.markdown(f"""
                <div style="
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;
                    padding:20px;
                    text-align:center;
                    transition:all 0.25s ease;
                    margin-bottom:12px;
                ">
                    <div style="width:72px;height:72px;border-radius:50%;margin:0 auto;
                                overflow:hidden;background:#111;
                                box-shadow:0 0 20px {tdata['color']}50;
                                display:flex;align-items:center;justify-content:center;">
                        <img src="{tdata['logo']}"
                             style="width:100%;height:100%;object-fit:cover;
                                    mix-blend-mode:screen;border-radius:50%;" />
                    </div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size:18px; font-weight:600;
                                color:{tdata['color']}; letter-spacing:2px; margin-top:12px;">
                        {tdata['abbr']}
                    </div>
                    <div style="font-size:10px; color:rgba(200,185,140,0.35); margin-top:4px;
                                letter-spacing:0.5px;">
                        {team_name}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"View {tdata['abbr']} Analysis", key=f"team_{i}"):
                 st.session_state.selected_team = team_name
                 st.session_state.page = "Team Analysis"
                 st.rerun()

    st.markdown("""
        <div style="padding:0 72px 32px; text-align:center;">
            <div style="display:inline-block; background:rgba(212,175,55,0.06); border:1px solid rgba(212,175,55,0.15);
                        border-radius:14px; padding:20px 36px;">
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:rgba(212,175,55,0.5);margin-bottom:8px;">Get Started</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:20px;color:#f0e8cc;font-weight:500;">
                    Open Match Analysis from the sidebar →
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# MODEL PERFORMANCE PAGE
# -----------------------------------
elif st.session_state.page == "Performance":

    st.markdown("""
        <div class="hero-wrapper" style="padding-bottom:32px;">
            <div class="hero-eyebrow">Classifier Diagnostic Metrics</div>
            <div class="hero-title" style="font-size:clamp(36px,4vw,56px); margin-bottom:10px;">Model Report</div>
            <div class="hero-subtitle">Comprehensive performance metrics, cross-validation scoring, and visual confusion matrix for the active model.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    with st.spinner("Analyzing active model parameters..."):
        metrics = evaluate_model(st.session_state.selected_model)

    # Convert model key to readable label
    model_name_map = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost"
    }
    active_model_name = model_name_map.get(st.session_state.selected_model, "Logistic Regression")

    # Metrics Row
    col_m1, col_m2, col_m3 = st.columns(3, gap="medium")
    
    with col_m1:
        st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value">{metrics['accuracy']:.2%}</div>
                <div class="stat-label">Test Accuracy</div>
                <div style="font-size:11px; color:rgba(220,210,185,0.45); margin-top:8px; line-height:1.4;">
                    Percentage of correct predictions on unseen test split data.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value">{metrics['cv_mean']:.2%}</div>
                <div class="stat-label">5-Fold CV Mean Accuracy</div>
                <div style="font-size:11px; color:rgba(220,210,185,0.45); margin-top:8px; line-height:1.4;">
                    Average validation score across 5 stratified folds. (SD: &plusmn;{metrics['cv_std']:.2%})
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value">{metrics['f1']:.2%}</div>
                <div class="stat-label">F1-Score</div>
                <div style="font-size:11px; color:rgba(220,210,185,0.45); margin-top:8px; line-height:1.4;">
                    Harmonic mean of precision and recall. Robust measure of model accuracy.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    # Detailed Analysis Columns
    col_det, col_cm = st.columns([1.1, 1.3], gap="medium")
    
    with col_det:
        st.markdown(f"""
            <div class="input-card" style="height: 100%;">
                <div class="input-label" style="font-size:11px;">Evaluation Deep Dive</div>
                <div style="margin-bottom: 24px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-family:'Cormorant Garamond',serif; font-size:20px; color:#f0e8cc; font-weight:500;">Precision</span>
                        <span style="font-family:'DM Mono',monospace; font-size:22px; color:#d4af37; font-weight:500;">{metrics['precision']:.2%}</span>
                    </div>
                    <p style="font-size:13px; color:rgba(220,210,185,0.5); line-height:1.5; margin:0;">
                        Out of all matches the model predicted as a win, how many were actual wins? High precision minimizes false positives.
                    </p>
                </div>
                <div style="margin-bottom: 24px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-family:'Cormorant Garamond',serif; font-size:20px; color:#f0e8cc; font-weight:500;">Recall (Sensitivity)</span>
                        <span style="font-family:'DM Mono',monospace; font-size:22px; color:#d4af37; font-weight:500;">{metrics['recall']:.2%}</span>
                    </div>
                    <p style="font-size:13px; color:rgba(220,210,185,0.5); line-height:1.5; margin:0;">
                        Out of all actual wins that occurred in the dataset, how many did the model correctly identify? High recall minimizes false negatives.
                    </p>
                </div>
                <div>
                    <div style="font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:rgba(212,175,55,0.35); margin-bottom:6px;">Model Settings</div>
                    <div style="font-family:'DM Mono',monospace; font-size:12px; color:rgba(220,210,185,0.6); background:rgba(0,0,0,0.2); padding:10px 14px; border-radius:8px; border:1px solid rgba(212,175,55,0.06); line-height:1.5;">
                        Active Classifier: {active_model_name}<br>
                        CV Strategy: 5-Fold Stratified K-Fold
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_cm:
        st.markdown(f"""
            <div class="matrix-wrapper">
                <div class="input-label" style="font-size:11px; margin-bottom: 8px;">Confusion Matrix</div>
                <div style="font-size:12px; color:rgba(220,210,185,0.45); margin-bottom: 20px; line-height:1.4;">
                    A tabular layout visualizing classification hits and misses. Gold-bordered diagonal cells represent correct predictions.
                </div>
                <div class="matrix-grid">
                    <div class="matrix-header">Actual \\ Pred</div>
                    <div class="matrix-header">Bowl Win (0)</div>
                    <div class="matrix-header">Bat Win (1)</div>
                    
                    <div class="matrix-label">Bowl Win (0)</div>
                    <div class="matrix-cell correct">
                        <div class="matrix-value">{metrics['tn']:,}</div>
                        <div class="matrix-cell-lbl">True Neg</div>
                    </div>
                    <div class="matrix-cell incorrect">
                        <div class="matrix-value">{metrics['fp']:,}</div>
                        <div class="matrix-cell-lbl">False Pos</div>
                    </div>
                    
                    <div class="matrix-label">Bat Win (1)</div>
                    <div class="matrix-cell incorrect">
                        <div class="matrix-value">{metrics['fn']:,}</div>
                        <div class="matrix-cell-lbl">False Neg</div>
                    </div>
                    <div class="matrix-cell correct">
                        <div class="matrix-value">{metrics['tp']:,}</div>
                        <div class="matrix-cell-lbl">True Pos</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Fold scores display
    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="input-label" style="font-size:11px; margin-bottom: 12px; padding-left: 4px;">Stratified 5-Fold Scores</div>', unsafe_allow_html=True)
    
    cv_cols = st.columns(5)
    for idx, score in enumerate(metrics['cv_scores']):
        with cv_cols[idx]:
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.015); border:1px solid rgba(255,255,255,0.05);
                            border-radius:10px; padding:12px; text-align:center;">
                    <div style="font-size:9px; letter-spacing:1px; text-transform:uppercase; color:rgba(220,210,185,0.35); margin-bottom:4px;">Fold {idx+1}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:15px; color:#e8d89a; font-weight:500;">{score:.2%}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Analysis":

    st.markdown("""
        <div class="hero-wrapper" style="padding-bottom:32px;">
            <div class="hero-eyebrow">Win Probability Engine</div>
            <div class="hero-title" style="font-size:clamp(36px,4vw,56px); margin-bottom:10px;">Match Analysis</div>
            <div class="hero-subtitle">Configure the match state below to compute real-time win probabilities.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    teams = list(team_data.keys())

    # ---- INPUT SECTION ----
    st.markdown("""
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    color:rgba(212,175,55,0.4);margin-bottom:20px;font-weight:500;">
            Match Configuration
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.2], gap="medium")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Teams</div>', unsafe_allow_html=True)
        batting_team = st.selectbox("Batting Team", teams, key="bat")
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], key="bowl")
        cities = [
            'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 
            'Cape Town', 'Centurion', 'Chandigarh', 'Chennai', 'Cuttack', 
            'Delhi', 'Dharamsala', 'Durban', 'East London', 'Hyderabad', 
            'Indore', 'Jaipur', 'Johannesburg', 'Kanpur', 'Kimberley', 
            'Kochi', 'Kolkata', 'Mohali', 'Mumbai', 'Nagpur', 
            'Port Elizabeth', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 
            'Sharjah', 'Visakhapatnam'
        ]
        selected_city = st.selectbox("Select Host City", cities, index=cities.index('Mumbai'), key="city")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Match State</div>', unsafe_allow_html=True)
        target = st.number_input("Target Score", min_value=50, max_value=300, value=180, step=1)
        score = st.number_input("Current Score", min_value=0, max_value=target - 1, value=50, step=1)
        col_ov, col_wk = st.columns(2)
        with col_ov:
            overs = st.slider("Overs Completed", min_value=0, max_value=20, value=10)
        with col_wk:
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

    # ---- TEAM VS DISPLAY ----
    t1 = team_data[batting_team]
    if bowling_team in team_data:
        t2 = team_data[bowling_team]
    else:
        t2 = team_data[teams[1]]

    st.markdown("""
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    color:rgba(212,175,55,0.4);margin-bottom:16px;font-weight:500;">
            Fixture
        </div>
    """, unsafe_allow_html=True)

    vs_col1, vs_col2, vs_col3 = st.columns([2.4, 0.8, 2.4], gap="medium")

    with vs_col1:
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                        border-radius:20px;padding:28px;text-align:center;
                        box-shadow:0 0 40px {t1['color']}12;">
                <div style="width:100px;height:100px;border-radius:50%;margin:0 auto;
                            overflow:hidden;background:#111;
                            box-shadow:0 0 28px {t1['color']}60;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t1['logo']}"
                         style="width:100%;height:100%;object-fit:cover;
                                mix-blend-mode:screen;" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:26px;font-weight:600;
                            color:{t1['color']};letter-spacing:3px;margin-top:14px;">
                    {t1['abbr']}
                </div>
                <div style="font-size:10px;color:rgba(200,185,140,0.3);margin-top:4px;letter-spacing:0.5px;">
                    BATTING
                </div>
            </div>
        """, unsafe_allow_html=True)

    with vs_col2:
        st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;height:100%;
                        font-family:'Cormorant Garamond',serif;font-size:52px;font-weight:300;
                        color:rgba(212,175,55,0.2);letter-spacing:-2px;padding:28px 0;">
                vs
            </div>
        """, unsafe_allow_html=True)

    with vs_col3:
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                        border-radius:20px;padding:28px;text-align:center;
                        box-shadow:0 0 40px {t2['color']}12;">
                <div style="width:100px;height:100px;border-radius:50%;margin:0 auto;
                            overflow:hidden;background:#111;
                            box-shadow:0 0 28px {t2['color']}60;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t2['logo']}"
                         style="width:100%;height:100%;object-fit:cover;
                                mix-blend-mode:screen;" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:26px;font-weight:600;
                            color:{t2['color']};letter-spacing:3px;margin-top:14px;">
                    {t2['abbr']}
                </div>
                <div style="font-size:10px;color:rgba(200,185,140,0.3);margin-top:4px;letter-spacing:0.5px;">
                    BOWLING
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

    # ---- ANALYZE BUTTON ----
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    analyze = st.button("Run Analysis", key="analyze_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- PREDICTION OUTPUT ----
    if analyze:
        runs_left, balls_left, crr, rrr = safe_calculate_rates(score, target, overs)

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # ---- VALIDATION LAYER (Issue #118) ----
        is_match_decided = False
        verdict_msg = ""
        verdict_type = "info" # "success" for chasing team win, "error" for chasing team loss, "warning" for tie/invalid
        
        if score >= target:
            is_match_decided = True
            verdict_msg = f"🏆 Match Decided: **{batting_team}** has already reached the target of {target} and won the match!"
            verdict_type = "success"
            win = 1.0
            lose = 0.0
        elif wickets >= 10:
            is_match_decided = True
            if score == target - 1:
                verdict_msg = f"🤝 Match Decided: **{batting_team}** is all out for {score}. The match is a **TIE**!"
                verdict_type = "warning"
                win = 0.5
                lose = 0.5
            else:
                verdict_msg = f"❌ Match Decided: **{batting_team}** is all out for {score} (target {target}). **{bowling_team}** won by {target - 1 - score} runs!"
                verdict_type = "error"
                win = 0.0
                lose = 1.0
        elif balls_left <= 0:
            is_match_decided = True
            if score == target - 1:
                verdict_msg = f"🤝 Match Decided: Overs completed. The match is a **TIE**!"
                verdict_type = "warning"
                win = 0.5
                lose = 0.5
            else:
                verdict_msg = f"❌ Match Decided: Overs completed. **{batting_team}** failed to reach the target of {target} and lost by {target - 1 - score} runs!"
                verdict_type = "error"
                win = 0.0
                lose = 1.0

        with st.spinner(""):
            if is_match_decided:
                pass
            else:
                if pipe is None:
                    st.error("Model not loaded. Please restart the app.")
                    st.stop()
                try:
                    proba = pipe.predict_proba(input_df)[0]
                except Exception as e:
                    logging.error(f"Prediction failed: {e}")
                    st.error("Prediction unavailable — model encountered an error. Adjust inputs and try again.")
                    st.stop()
                if np.isnan(proba).any():
                    st.error("Model returned invalid probabilities. The training pipeline may have produced corrupted coefficients. Restart the app to retrain.")
                    st.stop()
                win = proba[1]
                lose = proba[0]

        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                        color:rgba(212,175,55,0.4);margin-bottom:16px;font-weight:500;">
                Prediction Output
            </div>
        """, unsafe_allow_html=True)

        if is_match_decided:
            if verdict_type == "success":
                st.success(verdict_msg)
            elif verdict_type == "error":
                st.error(verdict_msg)
            else:
                st.warning(verdict_msg)
            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.1, 1.1], gap="medium")

        with res_col1:
            bat_pct = round(win * 100)
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Batting Team · {t1['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#c8b870;margin-bottom:16px;">
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
                            <div class="metric-chip-label">Score</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{runs_left}</div>
                            <div class="metric-chip-label">Needed</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{balls_left}</div>
                            <div class="metric-chip-label">Balls Left</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            bowl_pct = round(lose * 100)
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                            border-radius:24px;padding:36px 32px;position:relative;overflow:hidden;">
                    <div class="prediction-label">Bowling Team · {t2['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#c8b870;margin-bottom:16px;">
                        {bowling_team}
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:72px;font-weight:500;
                                color:rgba(200,185,140,0.55);line-height:1;margin-bottom:4px;">
                        {bowl_pct}%
                    </div>
                    <div class="win-prob-label">Win Probability</div>
                    <div class="prob-bar-track">
                        <div style="height:100%;border-radius:100px;
                                    background:rgba(200,185,140,0.2);
                                    width:{bowl_pct}%;transition:width 0.8s ease;"></div>
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
                            <div class="metric-chip-label">In Hand</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ---- SUMMARY ROW ----
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        verdict = batting_team if win > 0.5 else bowling_team
        conf = max(win, lose)
        conf_label = "High" if conf > 0.75 else "Moderate" if conf > 0.55 else "Close"

        st.markdown(f"""
            <div style="background:rgba(212,175,55,0.03);border:1px solid rgba(212,175,55,0.1);
                        border-radius:16px;padding:20px 28px;display:flex;
                        align-items:center;justify-content:space-between;">
                <div>
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                color:rgba(212,175,55,0.35);margin-bottom:6px;">Model Verdict</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:22px;
                                font-weight:500;color:#f0e8cc;">
                        {verdict} favoured to win
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                color:rgba(212,175,55,0.35);margin-bottom:6px;">Confidence</div>
                    <div style="font-family:'DM Mono',monospace;font-size:20px;color:#d4af37;">
                        {conf_label} · {round(conf*100)}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # ---- CSV EXPORT ----
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # Generate ball-by-ball predictions from current state to end of innings
        rows = []
        for ov in range(overs, 20):
            for bl in range(1, 7):
                total_balls_done = ov * 6 + bl
                if total_balls_done <= overs * 6:
                    continue  # skip already-played balls
                if total_balls_done > 120:
                    break

                b_left = 120 - total_balls_done
                c_score = score  # score stays same (projection from current state)
                r_left = target - c_score
                c_crr = c_score / (total_balls_done / 6) if total_balls_done > 0 else 0
                c_rrr = (r_left * 6) / b_left if b_left > 0 else 0

                proj_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': ['Mumbai'],
                    'runs_left': [r_left],
                    'balls_left': [b_left],
                    'wickets': [10 - wickets],
                    'target': [target],
                    'crr': [c_crr],
                    'rrr': [c_rrr]
                })

                try:
                    proj_proba = pipe.predict_proba(proj_df)[0]
                    bat_prob = round(proj_proba[1] * 100, 2)
                    bowl_prob = round(proj_proba[0] * 100, 2)
                except Exception:
                    bat_prob, bowl_prob = 50.0, 50.0
                rows.append({
                    "over": ov + 1,
                    "ball": bl,
                    "batting_team_prob": bat_prob,
                    "bowling_team_prob": bowl_prob
                })

        export_df = pd.DataFrame(rows)

        if not export_df.empty:
            st.download_button(
                label="⬇️ Download Ball-by-Ball Predictions (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"{batting_team}_vs_{bowling_team}_predictions.csv",
                mime="text/csv"
            )

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        
        # Generate ball-by-ball predictions for export
        with st.spinner("Generating export data..."):
            export_df = generate_ball_by_ball_df(
                pipe, batting_team, bowling_team, selected_city,
                target, score, overs, wickets
            )
            csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Ball-by-Ball Prediction Data (CSV)",
            data=csv_data,
            file_name=f"cricscope_predictions_{batting_team.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        render_venue_intelligence(batting_team, bowling_team, selected_city, target)
        # ---- CSV EXPORT ----
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # Generate ball-by-ball predictions from current state to end of innings
        rows = []
        for ov in range(overs, 20):
            for bl in range(1, 7):
                total_balls_done = ov * 6 + bl
                if total_balls_done <= overs * 6:
                    continue  # skip already-played balls
                if total_balls_done > 120:
                    break

                b_left = 120 - total_balls_done
                c_score = score  # score stays same (projection from current state)
                r_left = target - c_score
                c_crr = c_score / (total_balls_done / 6) if total_balls_done > 0 else 0
                c_rrr = (r_left * 6) / b_left if b_left > 0 else 0

                proj_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': ['Mumbai'],
                    'runs_left': [r_left],
                    'balls_left': [b_left],
                    'wickets': [10 - wickets],
                    'target': [target],
                    'crr': [c_crr],
                    'rrr': [c_rrr]
                })

                try:
                    proj_proba = pipe.predict_proba(proj_df)[0]
                    bat_prob = round(proj_proba[1] * 100, 2)
                    bowl_prob = round(proj_proba[0] * 100, 2)
                except Exception:
                    bat_prob, bowl_prob = 50.0, 50.0
                rows.append({
                    "over": ov + 1,
                    "ball": bl,
                    "batting_team_prob": bat_prob,
                    "bowling_team_prob": bowl_prob
                })

        export_df = pd.DataFrame(rows)

        if not export_df.empty:
            st.download_button(
                label="⬇️ Download Ball-by-Ball Predictions (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"{batting_team}_vs_{bowling_team}_predictions.csv",
                mime="text/csv"
            )

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        
        # Generate ball-by-ball predictions for export
        with st.spinner("Generating export data..."):
            export_df = generate_ball_by_ball_df(
                pipe, batting_team, bowling_team, selected_city,
                target, score, overs, wickets
            )
            csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Ball-by-Ball Prediction Data (CSV)",
            data=csv_data,
            file_name=f"cricscope_predictions_{batting_team.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close main-pad
    
# -----------------------------------
# TEAM ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Team Analysis":
    
    if "selected_team" not in st.session_state:
        st.warning("Please select a team from Dashboard.")
        st.stop()
    
    st.markdown("""
<div style="padding: 30px 50px;">
""", unsafe_allow_html=True)

    team = st.session_state.selected_team
    
    matches_df = pd.read_csv("matches.csv")

    team_matches = matches_df[
        (matches_df["team1"] == team) |
        (matches_df["team2"] == team)
    ]

    matches_played = len(team_matches)

    wins = len(
        team_matches[
            team_matches["winner"] == team
        ]
    )

    losses = matches_played - wins

    win_rate = round((wins / matches_played) * 100, 1) if matches_played > 0 else 0

    st.title("🏏 Team Analysis")

    if team:
        st.markdown(
    f"""
    <h2 style="
        color:{team_data[team]['color']};
        text-align:center;
        margin-bottom:20px;
    ">
        {team}
    </h2>
    """,
    unsafe_allow_html=True
)
         # Team Logo
        if team in team_data:
                c1, c2, c3 = st.columns([1,2,1])

    with c2:
        st.image(team_data[team]["logo"], width=180)
        
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Matches", matches_played)

        with col2:
            st.metric("Wins", wins)

        with col3:
            st.metric("Losses", losses)

        with col4:
            st.metric("Win Rate", f"{win_rate}%")
        
    # Performance Overview
        st.subheader("📊 Performance Overview")
        
        winning_matches = team_matches[
        team_matches["winner"] == team
]
        best_venue = winning_matches["venue"].mode()[0]

        seasons_played = team_matches["Season"].nunique()
        
        pom_count = winning_matches["player_of_match"].value_counts()
        top_player = pom_count.index[0]
        top_player_awards = pom_count.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
                st.metric("🏟 Best Venue", best_venue)

        with col2:
                st.metric("Seasons Played", seasons_played)

        with col3:
              st.metric(
        "🏆 Top Performer",
        top_player,
        f"{top_player_awards} Awards"
)

        st.markdown("---")

     # Team Strength Analysis
        st.subheader("📈 Team Statistics")
        
        deliveries_df = pd.read_csv("deliveries.csv")

        team_batting = deliveries_df[
        deliveries_df["batting_team"] == team
]
        total_runs = team_batting["total_runs"].sum()

        team_bowling = deliveries_df[
        deliveries_df["bowling_team"] == team
]   
        total_wickets = team_bowling[
        team_bowling["player_dismissed"].notna()
].shape[0]
        
        fielding_events = team_bowling[
        team_bowling["dismissal_kind"].isin(
        ["caught", "run out", "stumped"]
    )
]

        fielding_count = len(fielding_events)
            
        batting_strength = min(round(total_runs / 40000 * 100), 100)

        bowling_strength = min(round(total_wickets / 1200 * 100), 100)

        fielding_strength = min(round(fielding_count / 800 * 100), 100)
       
        # Batting
        st.markdown(
            f"🏏 **Total Runs** : {total_runs:,} ({batting_strength}%)"
        )
        st.progress(batting_strength / 100)

        # Bowling
        st.markdown(
            f"🎯 **Wickets Taken** : {total_wickets} ({bowling_strength}%)"
        )
        st.progress(bowling_strength / 100)

        # Fielding
        st.markdown(
            f"🧤 **Fielding Dismissals** : {fielding_count} ({fielding_strength}%)"
        )
        st.progress(fielding_strength / 100)
        
    
        if st.button("⬅ Back to Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
            
st.markdown("</div>", unsafe_allow_html=True)