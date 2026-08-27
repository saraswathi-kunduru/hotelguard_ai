import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64

# ============================================================
# HOTELGUARD AI — PREMIUM STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="HotelGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("hotelguard_final_model.pkl")


model = load_model()

# Load the generated AI assistant visual from the same project folder.
# The image is embedded as base64 so Streamlit can display it reliably.
ASSISTANT_IMAGE_PATH = Path(__file__).with_name("hotelguard_ai_assistant_visual.png")
if ASSISTANT_IMAGE_PATH.exists():
    ASSISTANT_IMAGE_DATA = base64.b64encode(ASSISTANT_IMAGE_PATH.read_bytes()).decode("utf-8")
    ASSISTANT_IMAGE_SRC = f"data:image/png;base64,{ASSISTANT_IMAGE_DATA}"
else:
    ASSISTANT_IMAGE_SRC = ""

THRESHOLD = 0.32

# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #07111f;
    --navy-2: #0b1728;
    --card: #101e31;
    --card-2: #13243a;
    --line: rgba(148,163,184,.16);
    --text: #f8fafc;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --blue: #2563eb;
    --green: #22c55e;
    --amber: #f59e0b;
    --orange: #f97316;
    --red: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 0%, rgba(37,99,235,.13), transparent 28%),
        radial-gradient(circle at 20% 20%, rgba(56,189,248,.06), transparent 24%),
        var(--navy);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081321 0%, #0d1b2d 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stRadio label {
    border-radius: 10px;
    padding: 6px 8px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56,189,248,.08);
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

p, label, .stCaption {
    color: #cbd5e1 !important;
}

hr {
    border-color: var(--line) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(19,36,58,.95), rgba(12,27,45,.95));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 12px 30px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
}

.stButton > button {
    border: 1px solid rgba(56,189,248,.28);
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    color: white;
    box-shadow: 0 10px 24px rgba(37,99,235,.20);
    transition: .2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(37,99,235,.30);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #0d1b2d !important;
    color: #f8fafc !important;
    border-color: var(--line) !important;
    border-radius: 10px !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 300px;
    border-radius: 28px;
    padding: 48px;
    margin-bottom: 26px;

    /* Keep the original hotel photo clearly visible.
       The gradient is only a gentle dark tint for text readability. */
    background:
        linear-gradient(
            90deg,
            rgba(2, 10, 20, 0.72) 0%,
            rgba(3, 14, 27, 0.58) 42%,
            rgba(3, 14, 27, 0.40) 72%,
            rgba(3, 14, 27, 0.34) 100%
        ),
        url("https://media.istockphoto.com/id/903417402/photo/luxury-construction-hotel-with-swimming-pool-at-sunset.jpg?s=612x612&w=0&k=20&c=NyPC_c-wE3W_CImA4t57FpyGy6f428CYROd80jxVC4A=");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    border: 1px solid rgba(56,189,248,.20);
    box-shadow: 0 25px 60px rgba(0,0,0,.24);
}


.hero-assistant {
    position: absolute;
    right: 0;
    top: 0;
    width: 52%;
    height: 100%;
    object-fit: cover;
    object-position: 62% center;
    border-radius: 0 28px 28px 0;
    opacity: .94;
    filter: saturate(1.05) contrast(1.02) drop-shadow(0 20px 35px rgba(0,0,0,.38));
    pointer-events: none;
    z-index: 1;
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.55) 16%, #000 34%, #000 100%);
    mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.55) 16%, #000 34%, #000 100%);
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(4,12,23,.18) 0%, transparent 48%, rgba(4,12,23,.08) 100%);
    z-index: 2;
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 3;
    max-width: 58%;
}

@media (max-width: 900px) {
    .hero-assistant {
        opacity: .20;
        width: 70%;
        right: -12%;
        top: 0;
        height: 100%;
        -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.25) 35%, #000 65%);
        mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.25) 35%, #000 65%);
    }
    .hero-content {
        max-width: 100%;
    }
}

.hero h1 {
    font-size: clamp(38px, 5vw, 64px) !important;
    margin: 0 0 10px 0;
    font-weight: 800;
}

.hero .eyebrow {
    color: #7dd3fc !important;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-weight: 800;
}

.hero .subtitle {
    color: #cbd5e1 !important;
    max-width: 680px;
    font-size: 18px;
    line-height: 1.65;
}

.pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(34,197,94,.12);
    border: 1px solid rgba(34,197,94,.24);
    color: #86efac !important;
    font-size: 12px;
    font-weight: 700;
    margin-top: 12px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    margin: 22px 0 14px 0;
}

.glass-card {
    background: linear-gradient(145deg, rgba(19,36,58,.92), rgba(10,24,40,.92));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 12px 30px rgba(0,0,0,.12);
}

.glass-card h3 {
    margin: 0 0 8px 0;
    font-size: 17px;
}

.glass-card p {
    color: var(--muted) !important;
    margin: 0;
    line-height: 1.55;
}

.kpi-card {
    background: linear-gradient(145deg, #13243a, #0d1b2d);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 12px 30px rgba(0,0,0,.14);
}

.kpi-label {
    color: var(--muted) !important;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.kpi-value {
    color: var(--text) !important;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-note {
    color: #7dd3fc !important;
    font-size: 12px;
    margin-top: 5px;
}

.result-card {
    border-radius: 24px;
    padding: 28px;
    border: 1px solid var(--line);
    background: linear-gradient(145deg, #12243a, #091827);
    box-shadow: 0 20px 45px rgba(0,0,0,.20);
}

.risk-critical { border-color: rgba(239,68,68,.45); }
.risk-high { border-color: rgba(249,115,22,.45); }
.risk-medium { border-color: rgba(245,158,11,.45); }
.risk-low { border-color: rgba(34,197,94,.45); }

.risk-number {
    font-size: 52px;
    line-height: 1;
    font-weight: 800;
    margin: 8px 0;
}

.risk-badge {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: .06em;
}

.badge-critical { background: rgba(239,68,68,.14); color: #fca5a5 !important; }
.badge-high { background: rgba(249,115,22,.14); color: #fdba74 !important; }
.badge-medium { background: rgba(245,158,11,.14); color: #fcd34d !important; }
.badge-low { background: rgba(34,197,94,.14); color: #86efac !important; }

.form-section {
    background: rgba(12,27,45,.70);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
}

.small-muted {
    color: #94a3b8 !important;
    font-size: 13px;
}

.feature-tag {
    display: inline-block;
    margin: 5px 5px 0 0;
    padding: 6px 10px;
    border-radius: 8px;
    background: rgba(56,189,248,.08);
    border: 1px solid rgba(56,189,248,.16);
    color: #bae6fd !important;
    font-size: 12px;
}

.footer {
    text-align: center;
    padding: 25px 0 5px;
    color: #64748b !important;
    font-size: 12px;
}

@media (max-width: 900px) {
    .hero {
        padding: 30px;
    }
}



/* ============================================================
   PREMIUM GLOWING AI BUBBLES
   ============================================================ */

.ai-particle-layer {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    pointer-events: none;
    z-index: 1;
}

.ai-particle {
    position: absolute;
    bottom: -70px;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: rgba(125, 211, 252, 0.95);
    border: 1px solid rgba(186, 230, 253, 0.35);
    box-shadow:
        0 0 7px rgba(56, 189, 248, 0.95),
        0 0 18px rgba(56, 189, 248, 0.58),
        0 0 32px rgba(56, 189, 248, 0.22);
    opacity: 0;
    animation: bubbleRise linear infinite;
    will-change: transform, opacity;
}

.ai-particle.small {
    width: 3px;
    height: 3px;
    box-shadow:
        0 0 5px rgba(125, 211, 252, 0.9),
        0 0 12px rgba(56, 189, 248, 0.48);
}

.ai-particle.medium {
    width: 7px;
    height: 7px;
    box-shadow:
        0 0 10px rgba(56, 189, 248, 1),
        0 0 24px rgba(56, 189, 248, 0.62),
        0 0 42px rgba(56, 189, 248, 0.25);
}

.ai-particle.large {
    width: 12px;
    height: 12px;
    background: rgba(125, 211, 252, 0.28);
    border: 1px solid rgba(125, 211, 252, 0.28);
    box-shadow:
        0 0 12px rgba(56, 189, 248, 0.95),
        0 0 30px rgba(56, 189, 248, 0.65),
        0 0 55px rgba(56, 189, 248, 0.28);
}

.ai-particle.xlarge {
    width: 18px;
    height: 18px;
    background: rgba(125, 211, 252, 0.14);
    border: 1px solid rgba(125, 211, 252, 0.22);
    box-shadow:
        0 0 16px rgba(56, 189, 248, 0.8),
        0 0 38px rgba(56, 189, 248, 0.48),
        0 0 70px rgba(56, 189, 248, 0.2);
}

@keyframes bubbleRise {
    0% {
        transform: translate3d(0, 40px, 0) scale(0.55);
        opacity: 0;
    }

    10% {
        opacity: 0.12;
    }

    22% {
        opacity: 0.58;
    }

    48% {
        transform: translate3d(45px, -42vh, 0) scale(0.95);
        opacity: 0.42;
    }

    72% {
        transform: translate3d(-35px, -76vh, 0) scale(1.08);
        opacity: 0.24;
    }

    100% {
        transform: translate3d(28px, -118vh, 0) scale(1.18);
        opacity: 0;
    }
}

.main .block-container {
    position: relative;
    z-index: 2;
}

[data-testid="stSidebar"] {
    position: relative;
    z-index: 3;
}

@media (prefers-reduced-motion: reduce) {
    .ai-particle-layer {
        display: none;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PREMIUM AI PARTICLES
# ============================================================

st.markdown(
    """
    <div class="ai-particle-layer" aria-hidden="true">
        <span class="ai-particle small" style="left:3%; animation-duration:16s; animation-delay:-4s;"></span>
        <span class="ai-particle medium" style="left:8%; animation-duration:23s; animation-delay:-13s;"></span>
        <span class="ai-particle small" style="left:13%; animation-duration:19s; animation-delay:-8s;"></span>
        <span class="ai-particle large" style="left:19%; animation-duration:27s; animation-delay:-18s;"></span>
        <span class="ai-particle small" style="left:25%; animation-duration:15s; animation-delay:-6s;"></span>
        <span class="ai-particle medium" style="left:31%; animation-duration:21s; animation-delay:-11s;"></span>
        <span class="ai-particle small" style="left:37%; animation-duration:18s; animation-delay:-2s;"></span>
        <span class="ai-particle xlarge" style="left:44%; animation-duration:31s; animation-delay:-22s;"></span>
        <span class="ai-particle medium" style="left:50%; animation-duration:24s; animation-delay:-15s;"></span>
        <span class="ai-particle small" style="left:56%; animation-duration:17s; animation-delay:-9s;"></span>
        <span class="ai-particle large" style="left:62%; animation-duration:29s; animation-delay:-20s;"></span>
        <span class="ai-particle small" style="left:68%; animation-duration:14s; animation-delay:-5s;"></span>
        <span class="ai-particle medium" style="left:73%; animation-duration:22s; animation-delay:-14s;"></span>
        <span class="ai-particle small" style="left:79%; animation-duration:18s; animation-delay:-7s;"></span>
        <span class="ai-particle xlarge" style="left:84%; animation-duration:32s; animation-delay:-25s;"></span>
        <span class="ai-particle medium" style="left:90%; animation-duration:20s; animation-delay:-10s;"></span>
        <span class="ai-particle small" style="left:96%; animation-duration:16s; animation-delay:-3s;"></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 4px 18px 4px;">
            <div style="font-size:30px;">🛡️</div>
            <div style="font-size:23px;font-weight:800;color:#f8fafc!important;">
                HotelGuard AI
            </div>
            <div style="font-size:12px;color:#94a3b8!important;margin-top:4px;">
                HOTEL REVENUE INTELLIGENCE
            </div>
            <div class="pill">● MODEL ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Navigation",
        [
    "🏠 Executive Home",
    "🎯 Booking Risk Prediction",
    "📊 Analytics Dashboard",
    "🧠 Explainable AI",
    "🏨 Risk Command Center",
    "🕒 Smart Waiting List",
    "📈 Model Performance",
    "💬 HotelGuard AI Chatbot",
],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="glass-card" style="min-height:auto;padding:16px;">
            <div style="font-size:11px;color:#94a3b8!important;
                        text-transform:uppercase;letter-spacing:.08em;">
                Production Model
            </div>
            <div style="font-size:17px;font-weight:800;margin-top:5px;">
                Tuned XGBoost
            </div>
            <div style="font-size:12px;color:#94a3b8!important;margin-top:8px;">
                Threshold: <b style="color:#7dd3fc!important;">0.32</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HELPERS
# ============================================================

def page_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div style="margin-bottom:24px;">
            <div style="font-size:12px;color:#38bdf8!important;
                        text-transform:uppercase;letter-spacing:.14em;
                        font-weight:800;">{eyebrow}</div>
            <h1 style="margin:6px 0 6px 0;font-size:38px;">{title}</h1>
            <div style="color:#94a3b8!important;font-size:15px;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_info(probability):
    if probability >= 0.75:
        return "CRITICAL", "critical", "Immediate intervention recommended."
    if probability >= 0.50:
        return "HIGH", "high", "Prioritize confirmation and proactive contact."
    if probability >= 0.32:
        return "MEDIUM", "medium", "Monitor and send a confirmation reminder."
    return "LOW", "low", "No immediate intervention is required."


# ============================================================
# HOME
# ============================================================

if page == "🏠 Executive Home":

    st.markdown(
        f"""
        <div class="hero">
            <img class="hero-assistant"
                 src="{ASSISTANT_IMAGE_SRC}"
                 alt="HotelGuard AI assistant"
            />
            <div class="hero-content">
                <div class="eyebrow">AI-POWERED HOTEL REVENUE PROTECTION</div>
                <h1>HotelGuard AI</h1>
                <div class="subtitle">
                    Predict booking cancellations before they happen,
                    understand the drivers behind risk, and turn predictions
                    into proactive revenue-protection decisions.
                </div>
                <div class="pill">● XGBoost MODEL ONLINE &nbsp; • &nbsp; 0.32 DECISION THRESHOLD</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("ROC-AUC", "92.38%", "Strong ranking performance")
    with c2:
        kpi("RECALL", "85.04%", "Cancellation detection")
    with c3:
        kpi("PR-AUC", "82.67%", "Imbalanced-class quality")
    with c4:
        kpi("F1 SCORE", "74.94%", "Balanced classification")

    st.markdown('<div class="section-title">What HotelGuard Does</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🎯 Predict Risk</h3>
                <p>Estimate cancellation probability for an individual hotel booking.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🧠 Explain Risk</h3>
                <p>Surface the booking characteristics that are most important to the model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <h3>💰 Protect Revenue</h3>
                <p>Translate cancellation probability into estimated booking revenue exposure.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Decision Flow</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("01", "Booking", "Customer & stay data"),
        ("02", "Model", "Tuned XGBoost"),
        ("03", "Probability", "Cancellation risk"),
        ("04", "Revenue", "Exposure estimate"),
        ("05", "Action", "Business response"),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4, c5], steps):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="min-height:145px;">
                    <div style="color:#38bdf8!important;font-weight:800;">{num}</div>
                    <h3 style="margin-top:8px;">{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer">
            HotelGuard AI • Explainable Hotel Cancellation Intelligence • Revenue Protection System
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PREDICTION
# ============================================================

elif page == "🎯 Booking Risk Prediction":

    page_header(
        "AI DECISION ENGINE",
        "Booking Risk Prediction",
        "Enter booking characteristics and receive a cancellation-risk assessment.",
    )

    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("### 🏨 Booking Profile")
    st.markdown(
        '<div class="small-muted">Core booking, guest, stay and commercial attributes</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("Lead Time", 0, 1000, 100)
        adults = st.number_input("Adults", 0, 20, 2)
        children = st.number_input("Children", 0.0, 10.0, 0.0)
        babies = st.number_input("Babies", 0, 10, 0)
        is_repeated_guest = st.selectbox("Repeated Guest", [0, 1])
        previous_cancellations = st.number_input("Previous Cancellations", 0, 100, 0)
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Cancelled", 0, 100, 0
        )
        booking_changes = st.number_input("Booking Changes", 0, 100, 0)

    with col2:
        arrival_year = st.number_input("Arrival Year", 2015, 2030, 2017)
        arrival_week_number = st.number_input("Arrival Week Number", 1, 53, 30)
        arrival_day = st.number_input("Arrival Day of Month", 1, 31, 15)
        arrival_month = st.selectbox(
            "Arrival Month",
            [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
        )
        stays_weekend = st.number_input("Weekend Nights", 0, 50, 1)
        stays_week = st.number_input("Week Nights", 0, 50, 3)
        days_waiting = st.number_input("Days in Waiting List", 0, 500, 0)
        adr = st.number_input("ADR", 0.0, 10000.0, 100.0)
        required_parking = st.number_input("Required Car Parking Spaces", 0, 10, 0)

    with col3:
        special_requests = st.number_input("Special Requests", 0, 20, 1)
        agent = st.number_input("Agent", 0, 1000, 0)
        company = st.number_input("Company", 0, 1000, 0)
        meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC", "Undefined"])
        market_segment = st.selectbox(
            "Market Segment",
            [
                "Online TA", "Offline TA/TO", "Direct", "Groups",
                "Corporate", "Complementary", "Aviation", "Undefined"
            ],
        )
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
        )
        deposit_type = st.selectbox(
            "Deposit Type",
            ["No Deposit", "Non Refund", "Refundable"],
        )
        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Transient-Party", "Contract", "Group"],
        )

    st.markdown("### 🛏️ Room & Guest Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        reserved_room_type = st.selectbox(
            "Reserved Room Type", list("ABCDEFGHIL")
        )
    with c2:
        assigned_room_type = st.selectbox(
            "Assigned Room Type", list("ABCDEFGHIL")
        )
    with c3:
        country = st.text_input("Country Code", value="PRT")

    st.markdown("</div>", unsafe_allow_html=True)

    total_guests = adults + children + babies
    total_nights = stays_weekend + stays_week
    estimated_booking_value = adr * total_nights
    is_long_stay = int(total_nights >= 7)
    is_high_value_booking = int(estimated_booking_value >= 1000)
    has_previous_cancellation = int(previous_cancellations > 0)
    has_booking_changes = int(booking_changes > 0)
    has_special_request = int(special_requests > 0)
    has_weekend_stay = int(stays_weekend > 0)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Estimated Guests", f"{total_guests:.0f}")
    with c2:
        st.metric("Total Nights", f"{total_nights:.0f}")
    with c3:
        st.metric("Estimated Booking Value", f"₹{estimated_booking_value:,.0f}")

    predict_button = st.button(
        "🎯 RUN AI RISK ASSESSMENT",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        input_data = pd.DataFrame(
            [
                {
                    "lead_time": lead_time,
                    "arrival_date_year": arrival_year,
                    "arrival_date_week_number": arrival_week_number,
                    "arrival_date_day_of_month": arrival_day,
                    "stays_in_weekend_nights": stays_weekend,
                    "stays_in_week_nights": stays_week,
                    "adults": adults,
                    "children": children,
                    "babies": babies,
                    "is_repeated_guest": is_repeated_guest,
                    "previous_cancellations": previous_cancellations,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled,
                    "booking_changes": booking_changes,
                    "agent": agent,
                    "company": company,
                    "days_in_waiting_list": days_waiting,
                    "adr": adr,
                    "required_car_parking_spaces": required_parking,
                    "total_of_special_requests": special_requests,
                    "total_guests": total_guests,
                    "total_nights": total_nights,
                    "estimated_booking_value": estimated_booking_value,
                    "is_long_stay": is_long_stay,
                    "is_high_value_booking": is_high_value_booking,
                    "has_previous_cancellation": has_previous_cancellation,
                    "has_booking_changes": has_booking_changes,
                    "has_special_request": has_special_request,
                    "has_weekend_stay": has_weekend_stay,
                    "hotel": hotel,
                    "arrival_date_month": arrival_month,
                    "meal": meal,
                    "country": country,
                    "market_segment": market_segment,
                    "distribution_channel": distribution_channel,
                    "reserved_room_type": reserved_room_type,
                    "assigned_room_type": assigned_room_type,
                    "deposit_type": deposit_type,
                    "customer_type": customer_type,
                }
            ]
        )

        try:
            probability = float(model.predict_proba(input_data)[0][1])
        except Exception as e:
            st.error("The model could not process these inputs.")
            st.exception(e)
            st.stop()

        prediction = int(probability >= THRESHOLD)
        risk, risk_class, recommendation = risk_info(probability)
        revenue_at_risk = probability * estimated_booking_value

        badge_class = f"badge-{risk_class}"

        st.markdown("## 🎯 AI Assessment Result")

        c1, c2 = st.columns([1, 1.4])

        with c1:
            st.markdown(
                f"""
                <div class="result-card risk-{risk_class}">
                    <div class="small-muted">CANCELLATION PROBABILITY</div>
                    <div class="risk-number">{probability:.1%}</div>
                    <span class="risk-badge {badge_class}">{risk} RISK</span>
                    <div style="margin-top:18px;color:#94a3b8!important;">
                        Decision threshold: <b style="color:#e2e8f0!important;">32%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="small-muted">REVENUE EXPOSURE</div>
                    <h2 style="font-size:36px;margin:5px 0;">
                        ₹{revenue_at_risk:,.0f}
                    </h2>
                    <div style="color:#94a3b8!important;">
                        Estimated revenue at risk from this booking.
                    </div>
                    <hr>
                    <div style="color:#94a3b8!important;">
                        Booking value:
                        <b style="color:#f8fafc!important;">
                            ₹{estimated_booking_value:,.0f}
                        </b>
                    </div>
                    <div style="color:#94a3b8!important;margin-top:8px;">
                        Model decision:
                        <b style="color:#f8fafc!important;">
                            {"CANCELLATION RISK" if prediction else "LOWER CANCELLATION RISK"}
                        </b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 💡 Recommended Business Action")

        if risk == "CRITICAL":
            st.error(
                "**Immediate intervention recommended.** Contact the customer, "
                "confirm the booking and consider additional retention measures."
            )
        elif risk == "HIGH":
            st.warning(
                "**High cancellation risk detected.** Prioritize this booking "
                "for confirmation and proactive customer communication."
            )
        elif risk == "MEDIUM":
            st.info(
                "**Moderate cancellation risk.** Monitor the booking and "
                "consider sending a confirmation reminder."
            )
        else:
            st.success(
                "**Low cancellation risk.** No immediate intervention is required."
            )

# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    page_header(
        "EXECUTIVE INTELLIGENCE",
        "Analytics Dashboard",
        "A management view of model quality, risk thresholds and revenue-protection strategy.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("ROC-AUC", "92.38%", "Model ranking quality")
    with c2:
        kpi("RECALL", "85.04%", "Cancellation detection")
    with c3:
        kpi("PR-AUC", "82.67%", "Precision-recall quality")
    with c4:
        kpi("THRESHOLD", "32%", "Decision boundary")

    st.markdown('<div class="section-title">Model Health</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🤖 Tuned XGBoost</h3>
                <p>
                    The production model uses a tuned XGBoost pipeline to
                    estimate the probability of hotel booking cancellation.
                </p>
                <br>
                <span class="feature-tag">ROC-AUC 92.38%</span>
                <span class="feature-tag">Recall 85.04%</span>
                <span class="feature-tag">PR-AUC 82.67%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>💰 Revenue Protection Strategy</h3>
                <p>
                    Risk scores can be converted into expected revenue exposure,
                    allowing hotel teams to prioritize high-risk bookings.
                </p>
                <br>
                <span class="feature-tag">Predict</span>
                <span class="feature-tag">Prioritize</span>
                <span class="feature-tag">Intervene</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
            ],
            "Score": [
                0.8427,
                0.6699,
                0.8504,
                0.7494,
                0.9238,
                0.8267,
            ],
        }
    )
    metrics["Percentage"] = metrics["Score"] * 100

    st.bar_chart(
        metrics.set_index("Metric")["Percentage"],
        use_container_width=True,
    )

    st.markdown('<div class="section-title">Risk Framework</div>', unsafe_allow_html=True)

    risk_table = pd.DataFrame(
        {
            "Risk Level": ["🟢 Low", "🟡 Medium", "🟠 High", "🔴 Critical"],
            "Probability": ["< 32%", "32% – 49%", "50% – 74%", "≥ 75%"],
            "Recommended Action": [
                "No immediate action",
                "Send reminder",
                "Prioritize confirmation",
                "Immediate intervention",
            ],
        }
    )

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("CANCELLATION DETECTION", "85.04%", "Recall")
    with c2:
        kpi("DECISION THRESHOLD", "0.32", "Optimized classification point")
    with c3:
        kpi("PRIMARY OBJECTIVE", "Revenue", "Protection")

# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    page_header(
        "MODEL TRANSPARENCY",
        "Explainable AI",
        "Understand which booking variables have the strongest influence on model predictions.",
    )

    st.markdown(
        """
        <div class="glass-card">
            <h3>🧠 Why Explainability Matters</h3>
            <p>
                Hotel teams need more than a probability score. Feature importance
                helps communicate which variables are most influential in the model.
                Importance alone does not establish whether a feature increases or
                decreases risk for every individual booking.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_data = pd.DataFrame(
        {
            "Feature": [
                "country_PRT",
                "required_car_parking_spaces",
                "agent",
                "lead_time",
                "total_of_special_requests",
                "market_segment_Online TA",
                "arrival_date_year",
                "booking_changes",
                "customer_type_Transient",
                "assigned_room_type_A",
            ],
            "Importance": [
                0.840541,
                0.640617,
                0.521985,
                0.487464,
                0.420155,
                0.305730,
                0.233624,
                0.158651,
                0.144573,
                0.122108,
            ],
        }
    )

    st.markdown("### ⭐ Top Model Features")

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.dataframe(
            feature_data,
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.bar_chart(
            feature_data.set_index("Feature")["Importance"],
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="glass-card" style="margin-top:18px;">
            <h3>🔎 Interpretation</h3>
            <p>
                The feature-importance view identifies variables that contribute
                strongly to the model's decisions. For a production explainability
                layer, individual SHAP values can be added to show the direction
                and magnitude of each feature for a particular booking.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    page_header(
        "MODEL GOVERNANCE",
        "Model Performance",
        "Validation metrics and the final classification threshold used by HotelGuard AI.",
    )

    performance = pd.DataFrame(
        {
            "Metric": [
                "ROC-AUC",
                "PR-AUC",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
            ],
            "Score": [
                0.9238267600570835,
                0.8266723909729606,
                0.8426998674275175,
                0.6698949441891004,
                0.8503854969785372,
                0.74942613166835,
            ],
        }
    )
    performance["Percentage"] = performance["Score"] * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("ROC-AUC", "92.38%", "Discrimination")
    with c2:
        kpi("RECALL", "85.04%", "Cancellation capture")
    with c3:
        kpi("F1 SCORE", "74.94%", "Balance")

    st.markdown("### 📊 Validation Metrics")

    st.bar_chart(
        performance.set_index("Metric")["Percentage"],
        use_container_width=True,
    )

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🎯 Final Decision Threshold")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric("Classification Threshold", "0.32")

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Why 0.32?</h3>
                <p>
                    The selected threshold is designed to improve cancellation
                    detection and achieve approximately 85.04% recall. In a real
                    deployment, the threshold should be monitored and recalibrated
                    as business costs and booking patterns change.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🏆 Production Summary")

    st.success(
        "HotelGuard AI combines cancellation prediction, explainability and "
        "revenue-at-risk estimation into a decision-support workflow for hotel teams."
    )
    # ============================================================
# HOTELGUARD AI CHATBOT
# ============================================================

elif page == "💬 HotelGuard AI Chatbot":

    page_header(
        "AI HOTEL ASSISTANT",
        "HotelGuard AI Chatbot",
        "Ask questions about hotel cancellations, booking risk, revenue protection and hotel operations.",
    )

    # --------------------------------------------------------
    # CHATBOT INTRO
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="glass-card">
            <h3>🤖 HotelGuard AI Assistant</h3>
            <p>
                Ask the assistant about cancellation risk, hotel bookings,
                revenue protection, customer communication and operational
                decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if gemini_client is None:

        st.error(
            "Gemini API key was not found. "
            "Please check your .env file."
        )

    else:

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        if "hotelguard_chat_history" not in st.session_state:
            st.session_state.hotelguard_chat_history = []

        # ----------------------------------------------------
        # DISPLAY PREVIOUS MESSAGES
        # ----------------------------------------------------

        for message in st.session_state.hotelguard_chat_history:

            if message["role"] == "user":

                with st.chat_message("user"):
                    st.markdown(message["content"])

            else:

                with st.chat_message("assistant"):
                    st.markdown(message["content"])

        # ----------------------------------------------------
        # CHAT INPUT
        # ----------------------------------------------------

        user_question = st.chat_input(
            "Ask HotelGuard AI something..."
        )

        if user_question:

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_question)

            st.session_state.hotelguard_chat_history.append(
                {
                    "role": "user",
                    "content": user_question,
                }
            )

            # ------------------------------------------------
            # HOTELGUARD SYSTEM CONTEXT
            # ------------------------------------------------

            system_context = """
You are HotelGuard AI, an intelligent hotel revenue and
cancellation-risk assistant.

HotelGuard AI is a hotel decision-support system.

The system uses a tuned XGBoost model to predict hotel booking
cancellation probability.

Important system information:

- Production model: Tuned XGBoost
- Decision threshold: 0.32
- ROC-AUC: 92.38%
- Recall: 85.04%
- PR-AUC: 82.67%
- F1 Score: 74.94%

Risk framework:

- Below 32%: Low Risk
- 32% to below 50%: Medium Risk
- 50% to below 75%: High Risk
- 75% or above: Critical Risk

Revenue at risk is estimated as:

cancellation probability × estimated booking value

Your job is to help hotel staff understand cancellation risk,
booking operations, revenue protection and customer communication.

Give practical, concise and professional answers.

Do not claim that you personally made a prediction unless
booking data has actually been provided.

If the user asks about a specific booking without providing
booking information, explain what information is needed.

Never invent booking data, hotel policies, prices or customer
information.

You can explain machine-learning concepts in simple language.

When recommending actions, focus on realistic hotel operations
such as confirmation messages, proactive communication,
monitoring, flexible modification options and prioritization.

You are an assistant for decision support, not a replacement
for hotel management.
"""

            # ------------------------------------------------
            # CONVERSATION
            # ------------------------------------------------

            conversation_text = ""

            for message in st.session_state.hotelguard_chat_history:

                role = message["role"].upper()

                conversation_text += (
                    f"\n{role}: {message['content']}\n"
                )

            prompt = (
                system_context
                + "\n\nConversation so far:"
                + conversation_text
                + "\n\nRespond to the latest user question."
            )

            # ------------------------------------------------
            # GEMINI REQUEST
            # ------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner("HotelGuard AI is thinking..."):

                    try:

                        response = gemini_client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=prompt,
                        )

                        answer = response.text

                        st.markdown(answer)

                        st.session_state.hotelguard_chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                    except Exception as e:

                        st.error(
                            "HotelGuard AI could not generate a response."
                        )

                        st.caption(
                            "Please check your internet connection "
                            "and Gemini API availability."
                        )

                        st.exception(e)

        # ----------------------------------------------------
        # CLEAR CHAT
        # ----------------------------------------------------

        if st.session_state.hotelguard_chat_history:

            if st.button(
                "🗑️ Clear Chat",
                use_container_width=True,
            ):

                st.session_state.hotelguard_chat_history = []

                st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ HotelGuard AI &nbsp;•&nbsp; Explainable Hotel Cancellation Intelligence
        &nbsp;•&nbsp; Revenue Protection System
    </div>
    """,
    unsafe_allow_html=True,
)
