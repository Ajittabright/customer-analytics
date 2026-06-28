# =============================================================================
# app/streamlit_app.py - Professional Corporate Version 3.0
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import joblib

from src.data_loader import load_config
from src.predictor import predict_single_customer, load_model_artifacts

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ChurnGuard | Customer Intelligence Platform",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CACHE
# =============================================================================
@st.cache_resource
def get_config():
    return load_config()

@st.cache_resource
def get_model_artifacts():
    config = get_config()
    return load_model_artifacts(config)

# =============================================================================
# PROFESSIONAL CSS - NO EMOJIS, CLEAN CORPORATE STYLE
# =============================================================================
def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e;
    }

    .nav-bar {
        background: #161b22;
        border-bottom: 1px solid #30363d;
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        border-radius: 8px;
    }

    .nav-logo {
        font-size: 1.3rem;
        font-weight: 800;
        color: #58a6ff;
        letter-spacing: -0.5px;
    }

    .nav-tagline {
        font-size: 0.8rem;
        color: #8b949e;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .nav-badge {
        background: #1f6feb;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        transition: border-color 0.2s;
    }

    .stat-card:hover {
        border-color: #58a6ff;
    }

    .stat-label {
        font-size: 0.72rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6edf3;
    }

    .stat-sub {
        font-size: 0.75rem;
        color: #58a6ff;
        margin-top: 0.2rem;
    }

    .result-panel-churn {
        background: linear-gradient(135deg, #3d1a1a 0%, #2d0f0f 100%);
        border: 1px solid #f85149;
        border-left: 4px solid #f85149;
        border-radius: 10px;
        padding: 2rem;
    }

    .result-panel-stay {
        background: linear-gradient(135deg, #0d2a1f 0%, #051f14 100%);
        border: 1px solid #3fb950;
        border-left: 4px solid #3fb950;
        border-radius: 10px;
        padding: 2rem;
    }

    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
    }

    .result-desc {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 0.5rem;
    }

    .risk-pill-high {
        display: inline-block;
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid #f85149;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .risk-pill-medium {
        display: inline-block;
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid #d29922;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .risk-pill-low {
        display: inline-block;
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid #3fb950;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8b949e;
        font-weight: 600;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #30363d;
        margin-bottom: 1.2rem;
        margin-top: 2rem;
    }

    .profile-table {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        overflow: hidden;
        width: 100%;
    }

    .profile-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 1.2rem;
        border-bottom: 1px solid #21262d;
    }

    .profile-row:last-child {
        border-bottom: none;
    }

    .profile-key {
        font-size: 0.82rem;
        color: #8b949e;
        font-weight: 400;
    }

    .profile-val {
        font-size: 0.85rem;
        color: #e6edf3;
        font-weight: 600;
    }

    .action-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem;
    }

    .action-header-red {
        color: #f85149;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }

    .action-header-yellow {
        color: #d29922;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }

    .action-header-green {
        color: #3fb950;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }

    .action-item {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        padding: 0.4rem 0;
        font-size: 0.85rem;
        color: #c9d1d9;
    }

    .action-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-top: 0.45rem;
        flex-shrink: 0;
    }

    .dot-red { background: #f85149; }
    .dot-yellow { background: #d29922; }
    .dot-green { background: #3fb950; }

    .insight-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        height: 100%;
        transition: border-color 0.2s;
    }

    .insight-panel:hover {
        border-color: #58a6ff;
    }

    .insight-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
    }

    .insight-text {
        font-size: 0.82rem;
        color: #8b949e;
        line-height: 1.5;
    }

    .sidebar-section-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 0.6rem;
        margin-top: 1rem;
    }

    .welcome-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 2rem;
    }

    .welcome-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 1.2rem;
    }

    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.7rem 0;
        border-bottom: 1px solid #21262d;
    }

    .step-row:last-child {
        border-bottom: none;
    }

    .step-num {
        background: #1f6feb;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 0.1rem;
    }

    .step-text {
        font-size: 0.85rem;
        color: #c9d1d9;
        line-height: 1.4;
    }

    .prob-display {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }

    .prob-number {
        font-size: 2.8rem;
        font-weight: 800;
        display: block;
        letter-spacing: -1px;
    }

    .prob-label-text {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #8b949e;
        margin-top: 0.3rem;
    }

    .footer-bar {
        margin-top: 3rem;
        padding: 1.2rem 2rem;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .footer-left {
        font-size: 0.78rem;
        color: #8b949e;
        font-weight: 600;
    }

    .footer-right {
        font-size: 0.72rem;
        color: #484f58;
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }

    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    .stButton button {
        background: #1f6feb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        width: 100% !important;
        transition: background 0.2s !important;
    }

    .stButton button:hover {
        background: #388bfd !important;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] {
        background: #0d1117 !important;
        border-color: #30363d !important;
    }

    hr { border-color: #30363d !important; }

    .stDownloadButton button {
        background: #161b22 !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        width: 100% !important;
        transition: border-color 0.2s !important;
    }

    .stDownloadButton button:hover {
        border-color: #58a6ff !important;
    }

    .stSpinner > div {
        border-top-color: #58a6ff !important;
    }

    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# GAUGE CHART - CLEAN VERSION
# =============================================================================
def create_gauge(probability):
    fig = plt.figure(figsize=(5, 2.8), facecolor='#161b22')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('#161b22')

    num_segments = 100
    theta = np.linspace(0, np.pi, num_segments)
    width = np.pi / num_segments

    for i, t in enumerate(theta):
        pct = i / num_segments * 100
        if pct < 30:
            color = '#3fb950'
        elif pct < 60:
            color = '#d29922'
        else:
            color = '#f85149'
        alpha = 0.3 + 0.7 * (i / num_segments)
        ax.bar(t, 0.35, width=width, bottom=0.6,
               color=color, alpha=alpha, edgecolor='none')

    needle_pct = probability / 100
    needle_angle = np.pi - needle_pct * np.pi
    ax.plot([needle_angle, needle_angle], [0.5, 0.92],
            color='#e6edf3', linewidth=2.5, solid_capstyle='round', zorder=5)
    ax.scatter([needle_angle], [0.5], color='#e6edf3', s=40, zorder=6)

    if probability < 30:
        gauge_color = '#3fb950'
    elif probability < 60:
        gauge_color = '#d29922'
    else:
        gauge_color = '#f85149'

    ax.text(np.pi / 2, 0.05, f'{probability:.1f}%',
            ha='center', va='center',
            fontsize=20, fontweight='800',
            color=gauge_color,
            transform=ax.transData,
            fontfamily='Inter')

    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(1)
    ax.set_thetalim(0, np.pi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)

    labels = ['0%', '30%', '60%', '100%']
    angles = [np.pi, np.pi * 0.7, np.pi * 0.4, 0]
    for label, angle in zip(labels, angles):
        x = 0.98 * np.cos(angle)
        y = 0.98 * np.sin(angle)
        ax.text(angle, 1.02, label,
                ha='center', va='center',
                fontsize=7, color='#484f58',
                fontfamily='Inter')

    plt.tight_layout(pad=0.2)
    return fig


# =============================================================================
# PROBABILITY CHART - CLEAN VERSION
# =============================================================================
def create_probability_chart(stay_prob, churn_prob):
    fig, ax = plt.subplots(figsize=(6, 2.2), facecolor='#161b22')
    ax.set_facecolor('#161b22')

    categories = ['Stay Probability', 'Churn Probability']
    values = [stay_prob, churn_prob]
    colors = ['#3fb950', '#f85149']

    bars = ax.barh(categories, values, color=colors,
                   height=0.4, edgecolor='none')

    for bar, val, color in zip(bars, values, colors):
        ax.text(
            min(val + 1.5, 108),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center',
            color=color,
            fontsize=11,
            fontweight='700',
            fontfamily='Inter'
        )

    ax.set_xlim(0, 115)
    ax.axvline(x=50, color='#30363d', linestyle='--', linewidth=1)
    ax.text(50, 1.55, '50%', ha='center', fontsize=7, color='#484f58')

    ax.set_xlabel('Probability (%)', color='#484f58', fontsize=8, labelpad=8)
    ax.tick_params(colors='#8b949e', labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


# =============================================================================
# FEATURE CHART - CLEAN VERSION
# =============================================================================
def create_feature_chart(customer_data):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#161b22')
    ax.set_facecolor('#161b22')

    features = ['Age', 'Balance', 'Credit Score', 'Num Products', 'Active Member']
    raw = [
        customer_data['Age'] / 92 * 100,
        customer_data['Balance'] / 300000 * 100,
        customer_data['CreditScore'] / 850 * 100,
        customer_data['NumOfProducts'] / 4 * 100,
        customer_data['IsActiveMember'] * 100
    ]

    risk_flags = [
        raw[0] > 55,
        raw[1] < 15,
        raw[2] < 50,
        raw[3] > 60,
        raw[4] < 50
    ]

    colors = ['#f85149' if flag else '#58a6ff' for flag in risk_flags]

    bars = ax.barh(features, raw, color=colors,
                   height=0.45, edgecolor='none', alpha=0.85)

    for bar, val, color in zip(bars, raw, colors):
        ax.text(
            min(val + 1.5, 108),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}%',
            va='center',
            color=color,
            fontsize=8.5,
            fontweight='600',
            fontfamily='Inter'
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel('Relative Score (%)', color='#484f58', fontsize=8, labelpad=8)
    ax.set_title('Feature Risk Indicators', color='#8b949e',
                 fontsize=9, fontweight='600', pad=10,
                 fontfamily='Inter')
    ax.tick_params(colors='#8b949e', labelsize=8.5)

    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    red_patch = mpatches.Patch(color='#f85149', label='Risk Factor')
    blue_patch = mpatches.Patch(color='#58a6ff', label='Normal Range')
    ax.legend(handles=[red_patch, blue_patch],
              facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#8b949e', fontsize=7.5,
              loc='lower right')

    plt.tight_layout()
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    st.sidebar.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:1.1rem; font-weight:800; color:#58a6ff;
                    letter-spacing:-0.3px;'>ChurnGuard</div>
        <div style='font-size:0.7rem; color:#484f58; text-transform:uppercase;
                    letter-spacing:1px; margin-top:2px;'>Customer Analysis Panel</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-section-label">Personal Information</div>',
                        unsafe_allow_html=True)

    age = st.sidebar.slider("Age", 18, 92, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-label">Financial Information</div>',
                        unsafe_allow_html=True)

    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

    if credit_score < 580:
        cs_label, cs_color = "Poor", "#f85149"
    elif credit_score < 670:
        cs_label, cs_color = "Fair", "#d29922"
    elif credit_score < 740:
        cs_label, cs_color = "Good", "#d29922"
    elif credit_score < 800:
        cs_label, cs_color = "Very Good", "#3fb950"
    else:
        cs_label, cs_color = "Excellent", "#3fb950"

    st.sidebar.markdown(
        f'<div style="text-align:right; margin-top:-0.8rem; margin-bottom:0.5rem;">'
        f'<span style="color:{cs_color}; font-size:0.75rem; font-weight:600;">{cs_label}</span></div>',
        unsafe_allow_html=True
    )

    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0, 1000.0)
    salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 300000.0, 80000.0, 1000.0)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-label">Banking Relationship</div>',
                        unsafe_allow_html=True)

    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_card = st.sidebar.checkbox("Has Credit Card", value=True)
    is_active = st.sidebar.checkbox("Active Member", value=True)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    predict_btn = st.sidebar.button(
        "Run Analysis",
        use_container_width=True,
        type="primary"
    )

    customer_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary
    }

    return customer_data, predict_btn


# =============================================================================
# HEADER
# =============================================================================
def render_header():
    st.markdown("""
    <div class="nav-bar">
        <div>
            <div class="nav-logo">ChurnGuard</div>
            <div class="nav-tagline">Customer Intelligence Platform</div>
        </div>
        <div class="nav-badge">ML POWERED</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        artifacts = get_model_artifacts()
        model_name = artifacts["metadata"]["model_name"]
        feature_count = artifacts["metadata"]["feature_count"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Model", model_name)
        with col2:
            st.metric("Feature Count", str(feature_count))
        with col3:
            st.metric("Training Samples", "8,000")
        with col4:
            st.metric("Session", datetime.now().strftime("%H:%M:%S"))
    except Exception:
        st.warning("Model not loaded. Please run model_trainer.py first.")


# =============================================================================
# RESULTS
# =============================================================================
def render_results(result, customer_data):

    st.markdown('<div class="section-title">Prediction Output</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        if result["will_churn"]:
            st.markdown("""
            <div class="result-panel-churn">
                <div class="result-label" style="color:#f85149;">Churn Alert</div>
                <div class="result-title" style="color:#f85149;">Customer at Risk</div>
                <div class="result-desc">
                    This customer shows strong indicators of churning.
                    Immediate retention action is recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-panel-stay">
                <div class="result-label" style="color:#3fb950;">Retention Signal</div>
                <div class="result-title" style="color:#3fb950;">Customer Retained</div>
                <div class="result-desc">
                    This customer shows low churn probability.
                    Standard engagement protocols apply.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if result["churn_probability"] < 30:
            pill = f'<span class="risk-pill-low">Low Risk</span>'
        elif result["churn_probability"] < 60:
            pill = f'<span class="risk-pill-medium">Medium Risk</span>'
        else:
            pill = f'<span class="risk-pill-high">High Risk</span>'

        st.markdown(pill, unsafe_allow_html=True)

    with col2:
        try:
            gauge_fig = create_gauge(result["churn_probability"])
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close()
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="prob-display">
                <span class="prob-number" style="color:#f85149;">{result['churn_probability']}%</span>
                <div class="prob-label-text">Churn Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="prob-display">
                <span class="prob-number" style="color:#3fb950;">{result['stay_probability']}%</span>
                <div class="prob-label-text">Stay Score</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Analysis Breakdown</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.5rem; font-weight:600;">PROBABILITY DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        prob_fig = create_probability_chart(
            result["stay_probability"],
            result["churn_probability"]
        )
        st.pyplot(prob_fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.5rem; font-weight:600;">FEATURE RISK INDICATORS</div>',
                    unsafe_allow_html=True)
        impact_fig = create_feature_chart(customer_data)
        st.pyplot(impact_fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">Customer Profile</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="profile-table">
            <div class="profile-row">
                <span class="profile-key">Age</span>
                <span class="profile-val">{customer_data['Age']} years</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Country</span>
                <span class="profile-val">{customer_data['Geography']}</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Gender</span>
                <span class="profile-val">{customer_data['Gender']}</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Credit Score</span>
                <span class="profile-val">{customer_data['CreditScore']}</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Account Balance</span>
                <span class="profile-val">${customer_data['Balance']:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="profile-table">
            <div class="profile-row">
                <span class="profile-key">Estimated Salary</span>
                <span class="profile-val">${customer_data['EstimatedSalary']:,.2f}</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Tenure</span>
                <span class="profile-val">{customer_data['Tenure']} years</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Products Held</span>
                <span class="profile-val">{customer_data['NumOfProducts']}</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Credit Card</span>
                <span class="profile-val">{'Yes' if customer_data['HasCrCard'] else 'No'}</span>
            </div>
            <div class="profile-row">
                <span class="profile-key">Active Member</span>
                <span class="profile-val">{'Yes' if customer_data['IsActiveMember'] else 'No'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recommended Actions</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="action-panel">', unsafe_allow_html=True)

    if result["churn_probability"] >= 60:
        st.markdown("""
        <div class="action-header-red">Urgent — Immediate Intervention Required</div>
        <div class="action-item">
            <div class="action-dot dot-red"></div>
            <div>Contact customer within 24 hours via phone or personal email</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-red"></div>
            <div>Offer personalized retention package including fee waivers or rate improvements</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-red"></div>
            <div>Escalate to senior relationship manager for high-value accounts</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-red"></div>
            <div>Schedule follow-up review within 7 days to confirm retention</div>
        </div>
        """, unsafe_allow_html=True)

    elif result["churn_probability"] >= 30:
        st.markdown("""
        <div class="action-header-yellow">Monitor — Proactive Engagement Advised</div>
        <div class="action-item">
            <div class="action-dot dot-yellow"></div>
            <div>Schedule a check-in call within the next 2 weeks</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-yellow"></div>
            <div>Send targeted email campaign with loyalty rewards or product offers</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-yellow"></div>
            <div>Review account activity for signs of disengagement</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="action-header-green">Stable — Standard Relationship Maintenance</div>
        <div class="action-item">
            <div class="action-dot dot-green"></div>
            <div>Continue standard engagement and quarterly communication</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-green"></div>
            <div>Consider cross-selling additional banking products</div>
        </div>
        <div class="action-item">
            <div class="action-dot dot-green"></div>
            <div>Enroll in long-term loyalty and rewards programme</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

    export_data = {
        **customer_data,
        "Predicted_Churn": result["prediction"],
        "Churn_Probability_%": result["churn_probability"],
        "Stay_Probability_%": result["stay_probability"],
        "Risk_Level": result["risk_level"],
        "Recommendation": result["recommendation"],
        "Model_Used": result["model_name"],
        "Prediction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    csv = pd.DataFrame([export_data]).to_csv(index=False)

    st.download_button(
        "Download Analysis Report (CSV)",
        data=csv,
        file_name=f"churnguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# =============================================================================
# WELCOME SCREEN
# =============================================================================
def render_welcome():

    st.markdown('<div class="section-title">Platform Overview</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div class="welcome-panel">
            <div class="welcome-title">How to Use This Platform</div>
            <div class="step-row">
                <div class="step-num">1</div>
                <div class="step-text">Enter customer personal details in the left panel including age, gender, and country</div>
            </div>
            <div class="step-row">
                <div class="step-num">2</div>
                <div class="step-text">Input financial information including credit score, balance, and estimated salary</div>
            </div>
            <div class="step-row">
                <div class="step-num">3</div>
                <div class="step-text">Provide banking relationship data such as tenure, products held, and activity status</div>
            </div>
            <div class="step-row">
                <div class="step-num">4</div>
                <div class="step-text">Click Run Analysis to generate the churn prediction and risk assessment</div>
            </div>
            <div class="step-row">
                <div class="step-num">5</div>
                <div class="step-text">Review the prediction results, feature analysis, and recommended actions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="welcome-panel">
            <div class="welcome-title">Model Information</div>
        """, unsafe_allow_html=True)

        model_info = {
            "Parameter": [
                "Algorithm",
                "Training Samples",
                "Test Samples",
                "Total Features",
                "Primary Metric",
                "Dataset"
            ],
            "Value": [
                "Gradient Boosting",
                "8,000",
                "2,000",
                "22",
                "F1-Score",
                "Bank Churn Dataset"
            ]
        }
        st.dataframe(
            pd.DataFrame(model_info),
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Key Findings from Data Analysis</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    insights = [
        ("Germany Region",
         "Customers from Germany show a churn rate of approximately 32 percent, significantly higher than France and Spain."),
        ("Age Group Risk",
         "Customers aged 45 to 60 demonstrate the highest churn tendency across all demographic segments."),
        ("Product Paradox",
         "Customers holding 3 or more products show unexpectedly elevated churn rates compared to 2-product holders."),
        ("Member Activity",
         "Inactive members are significantly more likely to churn within the next quarter than active members.")
    ]

    for col, (title, text) in zip([col1, col2, col3, col4], insights):
        with col:
            st.markdown(f"""
            <div class="insight-panel">
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    apply_styles()
    render_header()
    customer_data, predict_btn = render_sidebar()

    if predict_btn:
        with st.spinner("Running analysis..."):
            try:
                config = get_config()
                result = predict_single_customer(customer_data, config)
                render_results(result, customer_data)
            except FileNotFoundError:
                st.error("Model files not found. Please run: python main.py")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
    else:
        render_welcome()

    st.markdown("""
    <div class="footer-bar">
        <div class="footer-left">ChurnGuard v3.0 | Customer Intelligence Platform</div>
        <div class="footer-right">
            Built with Python 3.11 | Scikit-learn | Streamlit |
            Gradient Boosting Classifier | 10,000 Customer Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
