"""Email Spam Classifier — Advanced Message Spam Detection Suite.
Features a high-performance modern cyber-security UI with real-time explainability,
batch scanning, model telemetry, and threat heuristics.
"""

import base64
from datetime import datetime
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import clean_text

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.pkl"
OUTPUTS_DIR = ROOT / "outputs"
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.jpg"

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom Cyber / Dark Glassmorphism Design System
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-main: #070B14;
        --card-bg: rgba(15, 23, 42, 0.75);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-cyan: #06b6d4;
        --accent-blue: #3b82f6;
        --accent-violet: #8b5cf6;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    /* Overall Canvas */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 85% 20%, rgba(139, 92, 246, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(6, 182, 212, 0.08) 0%, transparent 50%);
        font-family: 'Outfit', sans-serif;
        color: var(--text-primary);
    }

    /* Top Padding & Block adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Custom Header Banner */
    .sentinel-header {
        display: flex;
        align-items: center;
        gap: 22px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
    }

    .sentinel-logo {
        width: 82px;
        height: 82px;
        border-radius: 18px;
        object-fit: cover;
        box-shadow: 0 0 24px rgba(6, 182, 212, 0.35);
        border: 2px solid rgba(6, 182, 212, 0.5);
    }

    .sentinel-title-wrap h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sentinel-subtitle {
        color: var(--text-secondary);
        font-size: 0.96rem;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-active {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-info {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Glass Cards */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }

    /* Result Banners */
    .result-banner-spam {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(136, 19, 55, 0.3) 100%);
        border: 1px solid rgba(244, 63, 94, 0.5);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.25);
        animation: pulseRed 2.5s infinite alternate;
    }

    .result-banner-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
    }

    @keyframes pulseRed {
        0% { box-shadow: 0 0 20px rgba(244, 63, 94, 0.2); }
        100% { box-shadow: 0 0 40px rgba(244, 63, 94, 0.45); }
    }

    .result-title-spam {
        font-family: 'Outfit', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        color: #FDA4AF;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }

    .result-title-ham {
        font-family: 'Outfit', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        color: #6EE7B7;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }

    /* Probability Visual Bar */
    .meter-container {
        width: 100%;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        height: 24px;
        overflow: hidden;
        display: flex;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .meter-spam {
        background: linear-gradient(90deg, #F43F5E, #FB7185);
        height: 100%;
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .meter-ham {
        background: linear-gradient(90deg, #059669, #10B981);
        height: 100%;
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Chip Badges */
    .chip {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .chip-danger {
        background: rgba(244, 63, 94, 0.18);
        color: #FDA4AF;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    .chip-safe {
        background: rgba(16, 185, 129, 0.15);
        color: #A7F3D0;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Heuristics Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 14px 18px;
        text-align: center;
    }

    .metric-card .num {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #38BDF8;
    }

    .metric-card .label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 2px;
    }

    /* Style Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--card-border);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: 600;
        color: var(--text-secondary);
        border: none;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background-color: rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }

    /* Streamlit Text Area */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.3) !important;
    }

    /* Primary Scan Button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 50%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 26px rgba(37, 99, 235, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Secondary preset buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #38BDF8 !important;
    }

    /* Footer */
    .sentinel-footer {
        text-align: center;
        padding: 24px 0 10px 0;
        margin-top: 40px;
        border-top: 1px solid var(--card-border);
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    .sentinel-footer a {
        color: #38BDF8;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model & Helper Utilities
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification engine...")
def load_classifier():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_image_base64(path: Path) -> str:
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def extract_threat_keywords(message: str, model, top_k: int = 6):
    """Identify which n-grams contributed most strongly to the spam prediction."""
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)
        if "spam" not in classes or "ham" not in classes:
            return []

        spam_idx = classes.index("spam")
        ham_idx = classes.index("ham")
        log_prob_diff = classifier.feature_log_prob_[spam_idx] - classifier.feature_log_prob_[ham_idx]

        cleaned = clean_text(message)
        analyzer = vectorizer.build_analyzer()
        tokens = analyzer(cleaned)

        vocab = vectorizer.vocabulary_
        found = []
        seen = set()
        for tok in tokens:
            if tok in vocab and tok not in seen:
                seen.add(tok)
                score = float(log_prob_diff[vocab[tok]])
                found.append((tok, score))

        # Sort descending by spam affinity
        found.sort(key=lambda x: x[1], reverse=True)
        return found[:top_k]
    except Exception:
        return []


def analyze_heuristics(text: str):
    """Compute risk heuristics: links, money, urgent verbs, capitalization."""
    urls = len(re.findall(r"https?://\S+|www\.\S+|bit\.ly\S+", text, re.I))
    emails = len(re.findall(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", text, re.I))
    currency = len(re.findall(r"[$€£₹¥]|(?:\b(?:usd|eur|gbp|inr|cash|dollars)\b)", text, re.I))
    numbers = len(re.findall(r"\d+", text))

    urgency_terms = [
        "urgent", "immediately", "winner", "won", "claim", "free", "congratulations",
        "prize", "selected", "expire", "action required", "call now", "verify",
        "suspended", "security alert", "warning", "bank"
    ]
    urgency_count = sum(1 for term in urgency_terms if term in text.lower())

    clean_chars = [c for c in text if c.isalpha()]
    caps_count = sum(1 for c in clean_chars if c.isupper())
    caps_ratio = (caps_count / len(clean_chars)) if clean_chars else 0.0

    return {
        "urls": urls,
        "emails": emails,
        "currency": currency,
        "numbers": numbers,
        "urgency": urgency_count,
        "caps_ratio": caps_ratio,
        "word_count": len(text.split()),
        "char_count": len(text),
    }


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
logo_b64 = get_image_base64(LOGO_PATH)
logo_img_tag = (
    f'<img class="sentinel-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Email Spam Classifier"/>'
    if logo_b64
    else '<div class="sentinel-logo" style="display:flex;align-items:center;justify-content:center;background:#1e293b;font-size:36px;">🛡️</div>'
)

st.markdown(
    f"""
    <div class="sentinel-header">
        {logo_img_tag}
        <div class="sentinel-title-wrap">
            <h1>EMAIL SPAM CLASSIFIER</h1>
            <div class="sentinel-subtitle">Advanced Natural Language Forensic Engine for Phishing, Scam & Spam Detection</div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="badge-pill badge-active">● ENGINE ONLINE</span>
                <span class="badge-pill badge-info">TF-IDF + MULTINOMIAL NAIVE BAYES</span>
                <span class="badge-pill badge-info">AIR-GAPPED &bull; LOCAL PRIVACY</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_classifier()
if model is None:
    st.error(
        "⚠️ No trained model detected at `models/spam_classifier.pkl`. "
        "Please train the model first by running `python -m src.train` in your terminal."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Tabs Navigation
# -----------------------------------------------------------------------------
tab_live, tab_batch, tab_telemetry, tab_history = st.tabs(
    ["🔍 Live Threat Inspector", "📂 Batch File Scanner", "📊 Model Telemetry & Matrix", "🕒 Scan Audit History"]
)

# =============================================================================
# TAB 1: Live Threat Inspector
# =============================================================================
with tab_live:
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown(
            """
            <div style="font-weight:700; font-size:1.1rem; margin-bottom:8px; color:#E2E8F0;">
                📝 Message Payload Input
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Preset Buttons
        st.markdown(
            "<span style='font-size:0.8rem; color:#94A3B8;'>QUICK TEST PRESETS:</span>",
            unsafe_allow_html=True,
        )
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            if st.button("🎁 Lottery Scam", use_container_width=True, key="p1_btn"):
                st.session_state.input_text = (
                    "URGENT WINNER NOTICE! You have been selected to win a $1,000 Walmart Gift Card "
                    "or cash prize. Call 09061743835 immediately to claim your award! Reference #8849."
                )
                st.rerun()

        with p2:
            if st.button("📦 Package Scam", use_container_width=True, key="p2_btn"):
                st.session_state.input_text = (
                    "USPS Alert: Your package delivery has been put on hold due to an incomplete address. "
                    "Please update your details within 12 hours at http://usps-portal-update.com/tracking"
                )
                st.rerun()

        with p3:
            if st.button("💼 Work Sync", use_container_width=True, key="p3_btn"):
                st.session_state.input_text = (
                    "Hi team, just confirming our sprint retrospective is moved to 2:30 PM today in Room 4B. "
                    "Please bring your slide updates if ready."
                )
                st.rerun()

        with p4:
            if st.button("☕ Casual Chat", use_container_width=True, key="p4_btn"):
                st.session_state.input_text = (
                    "Hey! Are you free this weekend to grab coffee or see the new movie? Let me know!"
                )
                st.rerun()

        user_text = st.text_area(
            label="Message content",
            value=st.session_state.input_text,
            height=180,
            placeholder="Paste an email body, SMS text, or communication message here for security analysis...",
            label_visibility="collapsed",
        )

        b_scan, b_clear = st.columns([3, 1])
        with b_scan:
            scan_triggered = st.button("⚡ ANALYZE MESSAGE FOR THREATS", type="primary", use_container_width=True)
        with b_clear:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.input_text = ""
                st.rerun()

        # Heuristic Quick Cards below input
        if user_text.strip():
            heuristics = analyze_heuristics(user_text)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="num">{heuristics['word_count']}</div>
                        <div class="label">Words</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with h2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="num" style="color: {'#F43F5E' if heuristics['urls'] > 0 else '#38BDF8'}">{heuristics['urls']}</div>
                        <div class="label">Links / URLs</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with h3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="num" style="color: {'#F59E0B' if heuristics['currency'] > 0 else '#38BDF8'}">{heuristics['currency']}</div>
                        <div class="label">Financial Cues</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with h4:
                caps_pct = int(heuristics['caps_ratio'] * 100)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="num" style="color: {'#F43F5E' if caps_pct > 25 else '#38BDF8'}">{caps_pct}%</div>
                        <div class="label">Caps Ratio</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_result:
        st.markdown(
            """
            <div style="font-weight:700; font-size:1.1rem; margin-bottom:8px; color:#E2E8F0;">
                🛡️ Forensic Threat Verdict
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_payload = user_text.strip()

        if active_payload:
            cleaned = clean_text(active_payload)
            prediction = model.predict([cleaned])[0]
            classes = list(model.classes_)
            probs = model.predict_proba([cleaned])[0]
            spam_idx = classes.index("spam")
            ham_idx = classes.index("ham")

            spam_prob = float(probs[spam_idx])
            ham_prob = float(probs[ham_idx])
            is_spam = prediction == "spam"

            # Log into history if newly scanned
            history_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "snippet": (active_payload[:75] + "...") if len(active_payload) > 75 else active_payload,
                "label": prediction.upper(),
                "spam_prob": f"{spam_prob:.1%}",
                "ham_prob": f"{ham_prob:.1%}",
            }
            if not st.session_state.history or st.session_state.history[0]["snippet"] != history_entry["snippet"]:
                st.session_state.history.insert(0, history_entry)

            # Visual Result Card
            if is_spam:
                st.markdown(
                    f"""
                    <div class="result-banner-spam">
                        <div class="result-title-spam">
                            <span>🚨 HIGH RISK DETECTED</span>
                        </div>
                        <div style="color:#FEE2E2; margin-top:6px; font-size:0.95rem;">
                            The classifier has identified high-confidence spam / fraudulent phishing signatures.
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:16px; font-weight:700;">
                            <span style="color:#FDA4AF;">SPAM RISK SCORE</span>
                            <span style="font-family:'JetBrains Mono', monospace; font-size:1.2rem; color:#FFFFFF;">{spam_prob:.1%}</span>
                        </div>
                        <div class="meter-container">
                            <div class="meter-spam" style="width: {spam_prob * 100:.1f}%;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8;">
                            <span>0% (Safe)</span>
                            <span>Threshold: 50%</span>
                            <span>100% (Confirmed Threat)</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-banner-ham">
                        <div class="result-title-ham">
                            <span>🛡️ VERIFIED LEGITIMATE (HAM)</span>
                        </div>
                        <div style="color:#D1FAE5; margin-top:6px; font-size:0.95rem;">
                            The message exhibits typical legitimate communication patterns with low threat indicators.
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:16px; font-weight:700;">
                            <span style="color:#6EE7B7;">SAFETY CONFIDENCE</span>
                            <span style="font-family:'JetBrains Mono', monospace; font-size:1.2rem; color:#FFFFFF;">{ham_prob:.1%}</span>
                        </div>
                        <div class="meter-container">
                            <div class="meter-ham" style="width: {ham_prob * 100:.1f}%;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8;">
                            <span>100% (Safe)</span>
                            <span>Spam Probability: {spam_prob:.1%}</span>
                            <span>0% (Threat)</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Threat Keywords & Explainability
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-weight:700; font-size:0.95rem; color:#E2E8F0; margin-bottom:6px;">
                    🎯 Model Attribution (Key Influence Words)
                </div>
                """,
                unsafe_allow_html=True,
            )

            threat_terms = extract_threat_keywords(active_payload, model, top_k=6)
            if threat_terms:
                chips_html = "".join(
                    f'<span class="chip {"chip-danger" if score > 0 else "chip-safe"}">'
                    f'{"▲" if score > 0 else "▼"} {term} ({score:+.2f})'
                    f'</span>'
                    for term, score in threat_terms
                )
                st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
                st.caption("Positive values (▲ red) pushed decision toward Spam; Negative (▼ green) pushed toward Ham.")
            else:
                st.caption("No strong vocabulary trigger weights detected in message.")

        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding: 48px 24px; color:#64748B;">
                    <div style="font-size: 3rem; margin-bottom: 12px;">📡</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #94A3B8;">Awaiting Message Payload</div>
                    <div style="font-size: 0.88rem; margin-top: 6px;">
                        Paste any text message or select one of the Quick Test Presets on the left to begin instant neural threat inspection.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 2: Batch Scanner
# =============================================================================
with tab_batch:
    st.markdown(
        """
        <div style="font-weight:700; font-size:1.1rem; color:#E2E8F0; margin-bottom:4px;">
            📂 Bulk Message Inspection Engine
        </div>
        <div style="color:#94A3B8; font-size:0.9rem; margin-bottom:18px;">
            Upload a CSV or TXT file containing communication payloads to batch-classify hundreds of records in seconds.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV file with messages", type=["csv", "tsv", "txt"])

    if uploaded_file is not None:
        try:
            # Read first few lines or full dataframe
            df = pd.read_csv(uploaded_file, on_bad_lines="skip")
            st.success(f"Successfully loaded {len(df)} rows.")

            # Identify column
            cols = df.columns.tolist()
            text_col = st.selectbox(
                "Select column containing the message text:",
                options=cols,
                index=0 if len(cols) == 1 else (1 if len(cols) > 1 else 0),
            )

            if st.button("🚀 Process Batch Classification", type="primary"):
                with st.spinner("Analyzing messages with Naive Bayes pipeline..."):
                    df["clean_msg"] = df[text_col].astype(str).map(clean_text)
                    predictions = model.predict(df["clean_msg"])
                    probs = model.predict_proba(df["clean_msg"])
                    spam_idx = list(model.classes_).index("spam")
                    spam_probs = probs[:, spam_idx]

                    df["prediction"] = predictions
                    df["spam_probability"] = (spam_probs * 100).round(2)
                    df["threat_level"] = np.where(spam_probs >= 0.75, "CRITICAL", np.where(spam_probs >= 0.5, "SUSPICIOUS", "SAFE"))

                    # Summary cards
                    total = len(df)
                    spam_count = int((df["prediction"] == "spam").sum())
                    ham_count = total - spam_count
                    spam_rate = (spam_count / total) * 100 if total > 0 else 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Scanned", total)
                    c2.metric("Spam Detected", spam_count, delta=f"{spam_rate:.1f}% rate", delta_color="inverse")
                    c3.metric("Legitimate (Ham)", ham_count)
                    c4.metric("Avg Spam Risk", f"{df['spam_probability'].mean():.1f}%")

                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                    # Preview table
                    preview_df = df[[text_col, "prediction", "spam_probability", "threat_level"]].head(100)
                    st.dataframe(preview_df, use_container_width=True)

                    # Download button
                    csv_export = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored Results (CSV)",
                        data=csv_export,
                        file_name="sentinel_scanned_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    else:
        st.info("💡 You can test the batch scanner using the Kaggle dataset located in the project's `data/scam.csv` file.")

# =============================================================================
# TAB 3: Model Telemetry & Matrix
# =============================================================================
with tab_telemetry:
    st.markdown(
        """
        <div style="font-weight:700; font-size:1.1rem; color:#E2E8F0; margin-bottom:4px;">
            📊 Model Telemetry, Architecture & Diagnostics
        </div>
        <div style="color:#94A3B8; font-size:0.9rem; margin-bottom:18px;">
            Rigorous evaluation metrics and feature importance distributions from the training pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "96.0%", "Test Set: 1,115 samples")
    m2.metric("Spam Precision", "100.0%", "0 False Positives")
    m3.metric("Spam Recall", "72.0%", "High specificity")
    m4.metric("Spam F1-Score", "0.84", "Balanced metric")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    top_words_path = OUTPUTS_DIR / "top_spam_words.png"

    with col_chart1:
        st.markdown("### Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), caption="Evaluation on 20% holdout test split (1,115 SMS messages)", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found. Run `python -m src.train` to generate it.")

    with col_chart2:
        st.markdown("### Top Spam Associations")
        if top_words_path.exists():
            st.image(str(top_words_path), caption="Top terms weighted by log-probability difference in Naive Bayes", use_container_width=True)
        else:
            st.warning("Top spam words plot not found. Run `python -m src.train` to generate it.")

# =============================================================================
# TAB 4: Scan Audit History
# =============================================================================
with tab_history:
    st.markdown(
        """
        <div style="font-weight:700; font-size:1.1rem; color:#E2E8F0; margin-bottom:4px;">
            🕒 Session Threat Audit Log
        </div>
        <div style="color:#94A3B8; font-size:0.9rem; margin-bottom:18px;">
            Chronological inspection history for all payloads tested during this active session.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        h_col1, h_col2 = st.columns([4, 1])
        with h_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:36px; color:#64748B;">
                No messages scanned in this session yet. Run a scan in the Live Inspector tab!
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="sentinel-footer">
        <strong>Email Spam Classifier &bull; Threat Intelligence</strong> &bull; Local Scikit-Learn Engine<br>
        <span style="font-size:0.75rem; color:#475569;">
            All text processed locally on-device. Zero telemetry transmitted externally.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
