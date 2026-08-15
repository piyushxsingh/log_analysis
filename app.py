"""
app.py
------
AI-Powered Intelligent Log Analysis System — Streamlit Dashboard

Run:
    streamlit run app.py
"""

import io
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Make sure sibling modules are importable regardless of cwd ────────────────
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_and_preprocess
from predict    import (
    load_models,
    run_full_prediction,
    generate_summary,
    get_recommendations,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Log Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark-card aesthetic with accent colours
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global font & base theme ───────────────────── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainContent"], .block-container {
        background: var(--background-color, #0a0f1e) !important;
        color: var(--text-color, #e2e8f0) !important;
    }
    .stMarkdown, .css-1kyxreq, .css-1d391kg {
        color: var(--text-color, #e2e8f0) !important;
    }
    .upload-hint {
        color: var(--secondary-text-color, #94a3b8) !important;
        font-size: 0.82rem;
        line-height: 1.4;
        margin-top: 0.25rem;
    }

    /* ── Toolbar / navigation bar ─────────────────── */
    header, [data-testid="stToolbar"], [data-testid="stToolbar"] * {
        background: var(--background-color, #0a0f1e) !important;
        color: var(--text-color, #e2e8f0) !important;
        border-color: transparent !important;
    }

    /* ── Inputs and widget fields ─────────────────── */
    input, textarea, select, button, [role="textbox"], [role="combobox"] {
        background-color: var(--secondary-background-color, #0f172a) !important;
        color: var(--text-color, #e2e8f0) !important;
        border-color: rgba(148, 163, 184, 0.32) !important;
    }
    input::placeholder, textarea::placeholder {
        color: rgba(226, 232, 240, 0.64) !important;
    }
    .stApp input,
    .stApp textarea,
    .stApp select,
    .stApp button,
    [data-testid="stMainContent"] input,
    [data-testid="stMainContent"] textarea,
    [data-testid="stMainContent"] select,
    [data-testid="stMainContent"] button,
    [data-testid="stMainContent"] [role="textbox"],
    [data-testid="stMainContent"] [role="combobox"] {
        background-color: var(--secondary-background-color, rgba(15, 23, 42, 0.85)) !important;
        color: var(--text-color, #e2e8f0) !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] button {
        background-color: var(--secondary-background-color, rgba(15, 23, 42, 0.85)) !important;
        color: var(--text-color, #e2e8f0) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stNumberInput"],
    section[data-testid="stSidebar"] [data-testid="stSlider"],
    section[data-testid="stSidebar"] [data-testid="stSelectbox"],
    [data-testid="stMainContent"] [data-testid="stFileUploader"],
    [data-testid="stMainContent"] [data-testid="stNumberInput"],
    [data-testid="stMainContent"] [data-testid="stSlider"],
    [data-testid="stMainContent"] [data-testid="stSelectbox"] {
        background-color: var(--secondary-background-color, rgba(15, 23, 42, 0.85)) !important;
    }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(2, 6, 23, 0.98)),
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 34%);
        background-color: var(--secondary-background-color, #0f172a) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    section[data-testid="stSidebar"] * { color: var(--text-color, #e2e8f0) !important; }
    section[data-testid="stSidebar"] > div {
        padding: 1.35rem 1rem 1.5rem;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.16);
        margin: 1rem 0;
    }
    section[data-testid="stSidebar"] label {
        font-weight: 650 !important;
        color: var(--text-color, #cbd5e1) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: var(--secondary-background-color, rgba(15, 23, 42, 0.72)) !important;
        border: 1px dashed rgba(20, 184, 166, 0.5);
        border-radius: 8px;
        padding: 0.85rem;
        color: var(--text-color, #e2e8f0) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: var(--text-color, #e2e8f0) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    section[data-testid="stSidebar"] .stDownloadButton button,
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px;
        border: 1px solid rgba(20, 184, 166, 0.45);
        background: rgba(20, 184, 166, 0.1);
        color: var(--text-color, #e2e8f0) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] {
        background: var(--secondary-background-color, rgba(30, 41, 59, 0.58)) !important;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
    }
    .sidebar-brand {
        display: flex;
        gap: 0.8rem;
        align-items: center;
        padding: 0.95rem;
        background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(99,102,241,0.12));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    }
    .sidebar-logo {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        flex: 0 0 42px;
        border-radius: 8px;
        background: #14b8a6;
        color: #02111f !important;
        font-size: 1.35rem;
        font-weight: 800;
    }
    .sidebar-title {
        margin: 0;
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: 0;
        line-height: 1.1;
    }
    .sidebar-subtitle {
        margin-top: 0.2rem;
        color: #94a3b8 !important;
        font-size: 0.78rem;
        line-height: 1.35;
    }
    .sidebar-section-label {
        margin: 1rem 0 0.45rem;
        color: #94a3b8 !important;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .sidebar-card {
        background: var(--secondary-background-color, rgba(30, 41, 59, 0.58));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.45rem 0 0.8rem;
    }
    .sidebar-card-title {
        margin-bottom: 0.3rem;
        font-size: 0.88rem;
        font-weight: 750;
    }
    .sidebar-card-copy {
        color: var(--secondary-text-color, #94a3b8) !important;
        font-size: 0.78rem;
        line-height: 1.45;
    }
    .sidebar-status-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem;
        margin: 0.75rem 0 0.95rem;
    }
    .sidebar-status {
        background: var(--secondary-background-color, rgba(15, 23, 42, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        padding: 0.65rem;
    }
    .sidebar-status-value {
        display: block;
        font-size: 0.98rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .sidebar-status-label {
        display: block;
        margin-top: 0.25rem;
        color: var(--secondary-text-color, #94a3b8) !important;
        font-size: 0.7rem;
    }
    .sidebar-footer {
        color: var(--secondary-text-color, #64748b) !important;
        font-size: 0.72rem;
        line-height: 1.5;
        padding-top: 0.4rem;
    }

    /* ── Metric cards ────────────────────────────── */
    div[data-testid="metric-container"] {
        background: var(--secondary-background-color, #1e293b);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label { color: var(--secondary-text-color, #94a3b8) !important; }
    div[data-testid="metric-container"] div   { color: var(--text-color, #f1f5f9) !important; }

    /* ── Section headers ─────────────────────────── */
    .section-header {
        background: linear-gradient(90deg, var(--secondary-background-color, #1e293b), var(--background-color, #0a0f1e));
        border-left: 5px solid #6366f1;
        padding: 10px 18px;
        border-radius: 8px;
        margin: 12px 0 8px 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-color, #e2e8f0);
    }

    /* ── Anomaly row highlight ───────────────────── */
    .anomaly-row { background-color: rgba(239,68,68,0.12); }

    /* ── Summary box ─────────────────────────────── */
    .summary-box {
        background: var(--secondary-background-color, #1e293b);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 20px;
        color: var(--text-color, #e2e8f0);
        line-height: 1.8;
    }

    /* ── Recommendation card ─────────────────────── */
    .rec-card {
        background: var(--secondary-background-color, #1e293b);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #6366f1;
        color: var(--text-color, #cbd5e1);
    }

    /* ── Dataframe headers ───────────────────────── */
    thead tr th { background: var(--secondary-background-color, #1e293b) !important; color: var(--text-color, #94a3b8) !important; }

    /* ── Hide Streamlit branding ──────────────────── */
    #MainMenu, footer { visibility: hidden; }

    /* ── Page background ─────────────────────────── */
    .stApp { background: var(--background-color, #0a0f1e); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root {
        --bg: #050b14;
        --panel: rgba(12, 22, 38, 0.88);
        --panel-soft: rgba(16, 31, 54, 0.72);
        --line: rgba(101, 130, 180, 0.24);
        --muted: #9aa8bd;
        --text: #f4f7fb;
        --blue: #5b7cff;
        --cyan: #22d3ee;
        --green: #35d979;
        --amber: #f6b73c;
        --red: #ff4b55;
        --purple: #8b5cf6;
    }
    .stApp {
        background:
            linear-gradient(145deg, rgba(37, 99, 235, 0.08), transparent 28%),
            linear-gradient(180deg, #06101f 0%, #03070d 100%) !important;
    }
    .block-container {
        max-width: 1560px;
        padding: 1rem 1.35rem 1.4rem !important;
    }
    header, [data-testid="stToolbar"] {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(10, 18, 34, 0.98), rgba(3, 9, 18, 0.98)),
            linear-gradient(145deg, rgba(34, 211, 238, 0.12), transparent 34%) !important;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] > div {
        padding: 1rem 0.9rem !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.9rem 0.85rem;
        margin-bottom: 0.8rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(145deg, rgba(15, 29, 54, 0.92), rgba(4, 10, 20, 0.92));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px rgba(0,0,0,0.26);
    }
    .sidebar-logo {
        width: 52px;
        height: 52px;
        flex: 0 0 52px;
        display: grid;
        place-items: center;
        border: 1px solid rgba(34, 211, 238, 0.7);
        border-radius: 8px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.22), rgba(20, 37, 64, 0.9));
        color: var(--cyan) !important;
        font-size: 1.65rem;
        font-weight: 900;
        box-shadow: 0 0 22px rgba(34, 211, 238, 0.22);
    }
    .sidebar-title {
        font-size: 1.15rem;
        line-height: 1.1;
        font-weight: 800;
        color: var(--text);
    }
    .sidebar-subtitle {
        margin-top: 0.25rem;
        color: var(--muted) !important;
        font-size: 0.82rem;
    }
    .sidebar-nav {
        display: grid;
        gap: 0.25rem;
        padding: 0.2rem 0 0.75rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 0.9rem;
    }
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        min-height: 36px;
        padding: 0.45rem 0.7rem;
        border-radius: 8px;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .sidebar-nav-item.active {
        color: #ffffff;
        background: linear-gradient(90deg, rgba(91, 124, 255, 0.9), rgba(91, 124, 255, 0.28));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .sidebar-section-label {
        margin: 0.9rem 0 0.45rem;
        color: #d4dbe8 !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-testid="stCheckbox"],
    section[data-testid="stSidebar"] [data-testid="stNumberInput"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] {
        background: rgba(12, 25, 45, 0.72) !important;
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border-style: dashed;
        padding: 0.75rem;
    }
    .app-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.45rem 1.7rem;
        margin-bottom: 0.9rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(13, 30, 55, 0.94), rgba(5, 12, 24, 0.96)),
            linear-gradient(90deg, rgba(34, 211, 238, 0.08), transparent);
        box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    .hero-icon {
        width: 44px;
        height: 44px;
        display: grid;
        place-items: center;
        color: var(--cyan);
        font-size: 1.55rem;
        border-radius: 8px;
        background: rgba(34, 211, 238, 0.1);
    }
    .app-hero h1 {
        margin: 0;
        font-size: 1.8rem;
        line-height: 1.15;
        color: var(--text);
        letter-spacing: 0;
    }
    .app-hero p {
        margin: 0.45rem 0 0 3.55rem;
        color: var(--muted);
        font-size: 0.92rem;
    }
    .hero-actions {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .hero-pill, .hero-avatar {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(11, 22, 39, 0.92);
        color: #e8eef9;
        padding: 0.58rem 0.8rem;
        font-size: 0.88rem;
    }
    .hero-export {
        border-color: rgba(91, 124, 255, 0.8);
        background: linear-gradient(135deg, #5b7cff, #6c63ff);
        color: white;
        font-weight: 700;
    }
    .hero-avatar {
        width: 42px;
        height: 42px;
        padding: 0;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #64748b, #1e293b);
    }
    .kpi-card {
        min-height: 96px;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(145deg, rgba(17, 32, 55, 0.88), rgba(8, 17, 31, 0.92));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 14px 36px rgba(0,0,0,0.2);
    }
    .kpi-icon {
        width: 54px;
        height: 54px;
        flex: 0 0 54px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        font-size: 1.55rem;
        font-weight: 900;
        background: rgba(91, 124, 255, 0.14);
        color: var(--blue);
    }
    .kpi-label {
        color: #d6deeb;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .kpi-value {
        color: white;
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 850;
    }
    .kpi-sub {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }
    .kpi-sub.hot { color: var(--red); }
    .kpi-sub.good { color: var(--green); }
    .dashboard-panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(145deg, rgba(15, 29, 50, 0.88), rgba(6, 13, 25, 0.92));
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .panel-title {
        color: var(--text);
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.65rem;
    }
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0;
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        margin-top: 0.7rem;
    }
    .stat-strip-item {
        padding: 0.85rem 1rem;
        background: rgba(13, 25, 43, 0.72);
        border-right: 1px solid var(--line);
    }
    .stat-strip-item:last-child { border-right: 0; }
    .stat-label { color: var(--muted); font-size: 0.8rem; }
    .stat-value { color: var(--text); font-size: 1.15rem; font-weight: 850; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(17, 32, 55, 0.88), rgba(8, 17, 31, 0.92));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] p { color: var(--muted) !important; }
    div[data-testid="stMetricValue"] { color: #fff !important; }
    .section-header {
        background: transparent !important;
        border-left: 0 !important;
        padding: 0.35rem 0 0.7rem !important;
        margin: 0 !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 0.65rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 0.75rem;
    }
    div[data-testid="stTabs"] [role="tab"] {
        color: #c7d2e4;
        padding: 0.75rem 0.45rem;
        font-size: 0.9rem;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        color: #7fa0ff !important;
        border-bottom-color: #5b7cff !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 8px !important;
        border: 1px solid rgba(91, 124, 255, 0.55) !important;
        background: linear-gradient(135deg, rgba(91, 124, 255, 0.95), rgba(108, 99, 255, 0.9)) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    @media (max-width: 900px) {
        .app-hero { align-items: flex-start; flex-direction: column; }
        .app-hero p { margin-left: 0; }
        .stat-strip { grid-template-columns: 1fr; }
        .stat-strip-item { border-right: 0; border-bottom: 1px solid var(--line); }
        .stat-strip-item:last-child { border-bottom: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SCHEME  — used by Matplotlib charts
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "INFO":     "#22c55e",   # green
    "WARNING":  "#f59e0b",   # amber
    "ERROR":    "#ef4444",   # red
    "CRITICAL": "#7c3aed",   # purple
    "anomaly":  "#ef4444",
    "normal":   "#22c55e",
    "bg":       "#0f172a",
    "card":     "#1e293b",
    "text":     "#e2e8f0",
    "grid":     "#334155",
}

def _style_fig(fig, ax_list=None):
    """Apply dark theme to a Matplotlib figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = ax_list or fig.get_axes()
    for ax in axes:
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">AI</div>
            <div>
                <div class="sidebar-title">AI Log Analyzer</div>
                <div class="sidebar-subtitle">Intelligent log monitoring</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV or TXT log file",
        type=["csv", "txt"],
        help="Drag-and-drop or click to browse. Supports CSV and plain-text syslog formats.",
    )
    st.markdown(
        "<div class='upload-hint'>Up to 200 MB per CSV file.</div>",
        unsafe_allow_html=True,
    )

    # Demo mode toggle
    use_demo = st.checkbox("▶ Use demo dataset", value=(uploaded_file is None))

    st.markdown('<div class="sidebar-section-label">Settings</div>', unsafe_allow_html=True)

    contamination = st.slider(
        "Anomaly sensitivity",
        min_value=0.05, max_value=0.40, value=0.15, step=0.05,
        help="Higher value → more logs flagged as anomalous.",
    )

    max_display = st.number_input(
        "Max rows to display", min_value=20, max_value=2000, value=200, step=50
    )

    st.markdown('<div class=''sidebar-section-label''>Built with</div><small style=''color:#64748b''>Python - Scikit-learn - Streamlit</small>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS  (cached so they're not reloaded on every rerun)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()

@st.cache_data(show_spinner=False)
def get_demo_data():
    return load_and_preprocess("data/system_logs.csv")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def severity_colour(sev: str) -> str:
    return {
        "INFO": "🟢", "WARNING": "🟡",
        "ERROR": "🟠", "CRITICAL": "🔴",
    }.get(sev, "⚪")


def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def render_kpi(icon: str, label: str, value: str, subtext: str, tone: str = "blue"):
    tone_colors = {
        "blue": "#5b7cff",
        "red": "#ff4b55",
        "green": "#35d979",
        "amber": "#f6b73c",
        "purple": "#8b5cf6",
        "cyan": "#22d3ee",
    }
    color = tone_colors.get(tone, tone_colors["blue"])
    sub_class = "good" if tone == "green" else "hot" if tone == "red" else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon" style="color:{color}; background:{color}22;">{icon}</div>
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub {sub_class}">{subtext}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_open(title: str):
    st.markdown(f'<div class="dashboard-panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def chart_severity_pie(df: pd.DataFrame):
    """Donut chart of log severity distribution."""
    counts = df["severity"].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values
    colors = [PALETTE.get(l, "#94a3b8") for l in labels]
    explode = [0.05] * len(labels)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
        pctdistance=0.80,
        wedgeprops=dict(width=0.46, linewidth=1.5, edgecolor=PALETTE["bg"]),
    )
    for t in texts:
        t.set_color(PALETTE["text"])
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax.text(0, 0.08, f"{len(df):,}", ha="center", va="center", color=PALETTE["text"], fontsize=18, fontweight="bold")
    ax.text(0, -0.13, "Total Logs", ha="center", va="center", color="#cbd5e1", fontsize=9)
    ax.set_title("Severity Distribution", color=PALETTE["text"], pad=14, fontweight="bold")
    return _style_fig(fig, [ax])


def chart_anomaly_bar(df: pd.DataFrame):
    """Stacked bar: normal vs anomaly per severity."""
    sev_order = ["INFO", "WARNING", "ERROR", "CRITICAL"]
    df_grp = (
        df.groupby(["severity", "anomaly"])
        .size()
        .unstack(fill_value=0)
        .reindex(sev_order, fill_value=0)
    )
    normal_vals  = df_grp.get(1,  pd.Series([0]*4, index=sev_order)).values
    anomaly_vals = df_grp.get(-1, pd.Series([0]*4, index=sev_order)).values

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(sev_order))
    w = 0.45

    ax.bar(x - w/2, normal_vals,  w, label="Normal",  color=PALETTE["normal"],  alpha=0.85)
    ax.bar(x + w/2, anomaly_vals, w, label="Anomaly", color=PALETTE["anomaly"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sev_order)
    ax.set_ylabel("Count")
    ax.set_title("Anomaly vs Normal Logs", fontweight="bold")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.6)
    return _style_fig(fig, [ax])


def chart_timeline(df: pd.DataFrame):
    """Line chart: log count per hour of day."""
    if "hour" not in df.columns or df["hour"].max() < 0:
        return None

    sev_order = ["INFO", "WARNING", "ERROR", "CRITICAL"]
    df_h = df[df["hour"] >= 0].groupby(["hour", "severity"]).size().unstack(fill_value=0)
    for sev in sev_order:
        if sev not in df_h.columns:
            df_h[sev] = 0
    df_h = df_h[sev_order]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for sev in sev_order:
        ax.plot(
            df_h.index, df_h[sev],
            label=sev, color=PALETTE[sev],
            linewidth=2, marker="o", markersize=4,
        )

    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Log Count")
    ax.set_title("Log Timeline  — Entries per Hour", fontweight="bold")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.set_xlim(0, 23)
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.5)
    return _style_fig(fig, [ax])


def chart_risk_donut(df: pd.DataFrame):
    """Donut chart for risk level distribution."""
    if "risk_label" not in df.columns:
        return None

    counts = df["risk_label"].value_counts()
    risk_colors = {
        "🔴 CRITICAL": "#7c3aed",
        "🟠 HIGH":     "#ef4444",
        "🟡 MEDIUM":   "#f59e0b",
        "🟢 LOW":      "#22c55e",
    }
    labels = counts.index.tolist()
    sizes  = counts.values
    colors = [risk_colors.get(l, "#64748b") for l in labels]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
    )
    for t in texts:
        t.set_color(PALETTE["text"])
        t.set_fontsize(9)
    for at in autotexts:
        at.set_color("#0f172a")
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax.set_title("Risk Level Distribution", color=PALETTE["text"], pad=14, fontweight="bold")
    return _style_fig(fig, [ax])


def chart_top_sources(df: pd.DataFrame):
    """Horizontal bar of top log-generating sources."""
    if "source" not in df.columns:
        return None

    top = df["source"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color="#6366f1", alpha=0.85)
    ax.set_xlabel("Log Count")
    ax.set_title("Top Log Sources", fontweight="bold")
    ax.bar_label(bars, padding=4, color=PALETTE["text"], fontsize=8)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.5)
    return _style_fig(fig, [ax])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────


# ─── Load / preprocess data ───────────────────────────────────────────────────
models_loaded = False
df_result     = None

with st.spinner("Loading AI models …"):
    try:
        models = get_models()
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"⚠️ Models not found. Please run `python train.py` first.\n\n{e}")
        st.stop()

# Determine data source
if uploaded_file is not None:
    with st.spinner("Parsing uploaded file …"):
        try:
            df_raw = load_and_preprocess(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            st.stop()
elif use_demo:
    with st.spinner("Loading demo dataset …"):
        df_raw = get_demo_data()
else:
    st.info("👈  Upload a log file or enable the demo dataset from the sidebar to get started.")
    st.stop()

# Run ML predictions
with st.spinner("Running AI analysis …"):
    # Temporarily override contamination from sidebar slider
    models["iso_forest"].set_params(contamination=contamination)
    models["iso_forest"].fit(
        models["vectorizer"].transform(df_raw["clean_message"].fillna(""))
    )
    df_result = run_full_prediction(df_raw, models)


# ─── Timestamp range ───────────────────────────────────────────────────
def parse_timestamp_range(df: pd.DataFrame):
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()
            return min_ts, max_ts
        except Exception:
            pass
    return None, None

# First, inject CSS separately (do this once, ideally at app start)
st.markdown("""
<style>
.app-hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    background: #0f172a;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-left h1 { color: #e2e8f0; margin: 0; font-size: 2rem; }
.hero-left p  { color: #94a3b8; margin: 6px 0 0; font-size: 0.95rem; }
.hero-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.hero-pill {
    background: #1e293b;
    color: #94a3b8;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
    border: 1px solid #334155;
    white-space: nowrap;
}
.hero-export {
    background: #1d4ed8;
    color: #fff;
    border-color: #2563eb;
    cursor: pointer;
}
.hero-export:hover { background: #2563eb; }
.hero-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #7c3aed;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# Then render the hero safely
try:
    ts_start, ts_end = parse_timestamp_range(df_result)
except Exception:
    ts_start, ts_end = "N/A", "N/A"

st.markdown(
    f"""
    <div class="app-hero">
        <div class="hero-left">
            <h1>🔍 AI-Powered Log Analysis System</h1>
            <p>· Machine Learning · Anomaly Detection · Severity Classification · Real-time Insights</p>
        </div>
        <div class="hero-actions">
            <div class="hero-pill">📅 {ts_start} – {ts_end}</div>
            <div class="hero-pill hero-export">Export Report</div>
            <div class="hero-avatar">AD</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS  (top row)
# ─────────────────────────────────────────────────────────────────────────────
total     = len(df_result)
n_anom    = (df_result["anomaly"] == -1).sum()
n_crit    = (df_result["severity"] == "CRITICAL").sum()
n_err     = (df_result["severity"] == "ERROR").sum()
anom_pct  = n_anom / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("LOG", "Total Logs", f"{total:,}", "Processed", "cyan")
with c2:
    render_kpi("AN", "Anomalies", f"{n_anom:,}", f"+ {anom_pct:.1f}%", "red")
with c3:
    render_kpi("CR", "Critical", f"{n_crit:,}", "Highest severity", "purple")
with c4:
    render_kpi("ER", "Errors", f"{n_err:,}", "Needs review", "amber")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_logs, tab_anomalies, tab_charts, tab_summary, tab_recs, tab_realtime = st.tabs([
    "📊 Overview",
    "📄 Log Viewer",
    "🚨 Anomalies",
    "📈 Charts",
    "📝 Summary",
    "💡 Recommendations",
    "⚡ Real-time Monitor",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        # Severity breakdown table
        sev_counts = df_result["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        sev_counts["Icon"]    = sev_counts["Severity"].map(severity_colour)
        sev_counts["% Share"] = (sev_counts["Count"] / total * 100).round(1).astype(str) + "%"
        sev_counts = sev_counts[["Icon", "Severity", "Count", "% Share"]]
        st.dataframe(sev_counts, use_container_width=True, hide_index=True)

        # Anomaly breakdown table
        st.markdown("**Anomaly Breakdown by Severity**")
        anom_df = (
            df_result[df_result["anomaly"] == -1]["severity"]
            .value_counts()
            .reset_index()
        )
        anom_df.columns = ["Severity", "Anomaly Count"]
        anom_df["Icon"] = anom_df["Severity"].map(severity_colour)
        st.dataframe(anom_df[["Icon", "Severity", "Anomaly Count"]],
                     use_container_width=True, hide_index=True)

    with col_r:
        fig_pie = chart_severity_pie(df_result)
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)
        st.markdown(
            f"""
            <div class="stat-strip">
                <div class="stat-strip-item">
                    <div class="stat-label">Total Anomalies</div>
                    <div class="stat-value" style="color:#ff4b55;">{n_anom:,}</div>
                </div>
                <div class="stat-strip-item">
                    <div class="stat-label">Anomaly Rate</div>
                    <div class="stat-value" style="color:#ff4b55;">{anom_pct:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Log Activity Over Time</div>', unsafe_allow_html=True)
    fig_timeline = chart_timeline(df_result)
    if fig_timeline:
        st.pyplot(fig_timeline, use_container_width=True)
        plt.close(fig_timeline)
    else:
        st.info("Timeline unavailable - timestamps could not be parsed.")

    # Risk level row
    st.markdown('<div class="section-header">Risk Level Summary</div>', unsafe_allow_html=True)
    risk_counts = df_result["risk_label"].value_counts()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🔴 CRITICAL Risk", risk_counts.get("🔴 CRITICAL", 0))
    r2.metric("🟠 HIGH Risk",     risk_counts.get("🟠 HIGH",     0))
    r3.metric("🟡 MEDIUM Risk",   risk_counts.get("🟡 MEDIUM",   0))
    r4.metric("🟢 LOW Risk",      risk_counts.get("🟢 LOW",      0))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOG VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown('<div class="section-header">Log Viewer</div>', unsafe_allow_html=True)

    # Filters
    flt1, flt2, flt3 = st.columns(3)
    with flt1:
        sev_filter = st.multiselect(
            "Filter by Severity",
            options=["INFO", "WARNING", "ERROR", "CRITICAL"],
            default=["INFO", "WARNING", "ERROR", "CRITICAL"],
        )
    with flt2:
        show_only_anomalies = st.checkbox("Show anomalies only", value=False)
    with flt3:
        search_text = st.text_input("🔎 Search messages", placeholder="e.g. database, timeout …")

    # Apply filters
    view_df = df_result[df_result["severity"].isin(sev_filter)].copy()
    if show_only_anomalies:
        view_df = view_df[view_df["anomaly"] == -1]
    if search_text.strip():
        view_df = view_df[
            view_df["message"].str.contains(search_text, case=False, na=False)
        ]

    st.caption(f"Showing {min(len(view_df), max_display):,} of {len(view_df):,} matching logs")

    # Display columns
    display_cols = ["timestamp", "severity", "source", "message", "anomaly", "risk_label", "confidence"]
    display_cols = [c for c in display_cols if c in view_df.columns]

    st.dataframe(
        view_df[display_cols].head(max_display),
        use_container_width=True,
        hide_index=True,
        column_config={
            "anomaly":    st.column_config.NumberColumn("Anomaly", help="1=Normal, -1=Anomaly"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "risk_label": st.column_config.TextColumn("Risk"),
            "severity":   st.column_config.TextColumn("Severity"),
        },
    )

    # Download button
    st.download_button(
        label="⬇️ Download filtered logs (CSV)",
        data=df_to_csv(view_df),
        file_name="filtered_logs.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_anomalies:
    st.markdown('<div class="section-header">🚨 Detected Anomalies</div>', unsafe_allow_html=True)

    anomaly_df = df_result[df_result["anomaly"] == -1].copy()
    anomaly_df = anomaly_df.sort_values("anomaly_score", ascending=True)  # most anomalous first

    st.info(
        f"**{len(anomaly_df):,} anomalous log entries detected** "
        f"({len(anomaly_df)/total*100:.1f}% of total). "
        "Sorted by anomaly score — most suspicious first."
    )

    if not anomaly_df.empty:
        acol1, acol2 = st.columns([2, 1])
        with acol1:
            a_cols = ["timestamp", "severity", "message", "anomaly_score", "risk_label"]
            a_cols = [c for c in a_cols if c in anomaly_df.columns]
            st.dataframe(
                anomaly_df[a_cols].head(max_display),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "anomaly_score": st.column_config.NumberColumn(
                        "Anomaly Score", format="%.4f",
                        help="More negative = more anomalous",
                    ),
                },
            )
        with acol2:
            # Top anomalous sources
            if "source" in anomaly_df.columns:
                st.markdown("**Top Anomalous Sources**")
                src = anomaly_df["source"].value_counts().head(6).reset_index()
                src.columns = ["Source", "Count"]
                st.dataframe(src, use_container_width=True, hide_index=True)

            # Severity split among anomalies
            st.markdown("**Severity of Anomalies**")
            sev_anom = anomaly_df["severity"].value_counts().reset_index()
            sev_anom.columns = ["Severity", "Count"]
            st.dataframe(sev_anom, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Download anomaly report (CSV)",
            data=df_to_csv(anomaly_df),
            file_name="anomaly_report.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown('<div class="section-header">📈 Visual Analytics</div>', unsafe_allow_html=True)

    # Row 1: pie + anomaly bar
    ch1, ch2 = st.columns(2)
    with ch1:
        fig1 = chart_severity_pie(df_result)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with ch2:
        fig2 = chart_anomaly_bar(df_result)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Row 2: timeline (full width)
    st.markdown("---")
    fig3 = chart_timeline(df_result)
    if fig3:
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    else:
        st.info("Timeline unavailable — timestamps could not be parsed.")

    # Row 3: risk donut + top sources
    st.markdown("---")
    ch3, ch4 = st.columns(2)
    with ch3:
        fig4 = chart_risk_donut(df_result)
        if fig4:
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)
    with ch4:
        fig5 = chart_top_sources(df_result)
        if fig5:
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.markdown('<div class="section-header">📝 AI-Generated Analysis Summary</div>', unsafe_allow_html=True)

    summary_text = generate_summary(df_result)

    st.markdown(
        f'<div class="summary-box">{summary_text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Predicted vs actual severity comparison
    if "predicted_severity" in df_result.columns:
        st.markdown("**Model Prediction Accuracy (sample)**")
        cmp_df = df_result[["severity", "predicted_severity", "confidence"]].head(100)
        cmp_df["Match"] = cmp_df["severity"] == cmp_df["predicted_severity"]
        match_pct = cmp_df["Match"].mean() * 100
        st.metric("Prediction Match Rate (first 100 logs)", f"{match_pct:.1f}%")
        st.dataframe(cmp_df.head(20), use_container_width=True, hide_index=True)

    # Export full report
    st.download_button(
        label="⬇️ Download full analysis (CSV)",
        data=df_to_csv(df_result),
        file_name="full_analysis.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    st.markdown('<div class="section-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)

    recs = get_recommendations(df_result)

    ICONS = {"CRITICAL": "🔴", "ERROR": "🟠", "WARNING": "🟡", "INFO": "🟢"}

    for sev, actions in recs.items():
        with st.expander(f"{ICONS[sev]}  {sev} — {len(actions)} recommendations", expanded=(sev == "CRITICAL")):
            for action in actions:
                st.markdown(
                    f'<div class="rec-card">{action}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 📌 Overall Risk Assessment")

    max_risk = df_result["risk_score"].max() if "risk_score" in df_result.columns else 1
    if max_risk == 4:
        st.error("🔴 **CRITICAL RISK** — Immediate action required. Escalate to on-call team now.")
    elif max_risk == 3:
        st.warning("🟠 **HIGH RISK** — Active errors detected. Investigate and remediate within 1 hour.")
    elif max_risk == 2:
        st.warning("🟡 **MEDIUM RISK** — Warnings present. Monitor closely and plan maintenance.")
    else:
        st.success("🟢 **LOW RISK** — System is healthy. Continue standard monitoring.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — REAL-TIME MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_realtime:
    st.markdown('<div class="section-header">⚡ Real-time Log Monitor (Simulation)</div>', unsafe_allow_html=True)

    st.info(
        "This tab simulates a live log stream by replaying entries from the loaded dataset. "
        "In production, connect a Kafka/Fluentd/syslog stream to this pipeline."
    )

    col_start, col_stop, col_speed = st.columns([1, 1, 2])
    start_btn = col_start.button("▶ Start Stream")
    speed     = col_speed.select_slider(
        "Stream speed", options=["Slow (2s)", "Normal (1s)", "Fast (0.3s)"],
        value="Normal (1s)"
    )
    delay_map = {"Slow (2s)": 2.0, "Normal (1s)": 1.0, "Fast (0.3s)": 0.3}
    delay     = delay_map[speed]

    if start_btn:
        log_placeholder   = st.empty()
        metric_placeholder = st.empty()

        shown_rows = []
        colour_map = {
            "INFO": "🟢", "WARNING": "🟡", "ERROR": "🟠", "CRITICAL": "🔴"
        }

        # Stream first 30 entries
        for _, row in df_result.head(30).iterrows():
            shown_rows.append(row)
            stream_df = pd.DataFrame(shown_rows)

            with metric_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Logs processed", len(shown_rows))
                m2.metric("Anomalies found", (stream_df["anomaly"] == -1).sum())
                m3.metric("Critical events", (stream_df["severity"] == "CRITICAL").sum())

            # Show last 10 entries
            recent = pd.DataFrame(shown_rows[-10:])
            disp   = recent[["timestamp", "severity", "message", "anomaly", "risk_label"]].copy()
            log_placeholder.dataframe(disp, use_container_width=True, hide_index=True)

            time.sleep(delay)

        st.success(f"✅ Stream complete — processed {len(shown_rows)} log entries.")


