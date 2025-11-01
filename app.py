"""
AI Phishing Email Detector - Premium Black & Gold UI
TF-IDF + Logistic Regression trained on Kaggle Phishing Emails dataset.
Author & Deployer: Umaima Qureshi
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# Page Configuration
st.set_page_config(
    page_title="AI Phishing Shield – by Umaima Qureshi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'cm_plot_cached' not in st.session_state:
    st.session_state.cm_plot_cached = None

# Premium Black & Gold CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
* {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%);
}
.main {
    background: transparent;
    padding: 0;
}
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px;
}
section[data-testid="stSidebar"] {
    display: none;
}
/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
    border-radius: 32px;
    padding: 4rem 3rem;
    margin-bottom: 3rem;
    box-shadow: 0 25px 70px rgba(0,0,0,0.6), 0 10px 30px rgba(218,165,32,0.25), inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
    border: 2px solid rgba(218,165,32,0.4);
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(218,165,32,0.2) 0%, transparent 70%);
    border-radius: 50%;
    animation: pulse 8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.3; }
    50% { transform: scale(1.1); opacity: 0.5; }
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(255,215,0,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 4.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
    letter-spacing: -0.03em;
    filter: drop-shadow(0 4px 20px rgba(255,215,0,0.4));
}
.hero-subtitle {
    font-size: 1.45rem;
    color: #e5e7eb;
    font-weight: 500;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
    line-height: 1.6;
    letter-spacing: 0.3px;
}
.hero-description {
    color: #d1d5db;
    font-size: 1.05rem;
    line-height: 1.7;
    position: relative;
    z-index: 1;
    max-width: 900px;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    color: #0f0f0f;
    padding: 0.8rem 2.5rem;
    border-radius: 50px;
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 1.8rem;
    box-shadow: 0 8px 25px rgba(255,215,0,0.5), 0 0 40px rgba(255,215,0,0.3);
    position: relative;
    z-index: 1;
    transition: all 0.3s ease;
}
.hero-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(255,215,0,0.6), 0 0 50px rgba(255,215,0,0.4);
}
/* Section Headers */
.section-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 3.5rem 0 2rem 0;
    text-align: center;
    letter-spacing: 0.5px;
    position: relative;
    padding-bottom: 1rem;
}
.section-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, transparent, #FFD700, transparent);
    border-radius: 2px;
}
/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.8rem;
    margin: 2.5rem 0;
}
.stat-card {
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    padding: 2.5rem 1.8rem;
    border-radius: 24px;
    text-align: center;
    color: #0f0f0f;
    box-shadow: 0 10px 30px rgba(255,215,0,0.35), 0 0 40px rgba(255,215,0,0.2), inset 0 1px 0 rgba(255,255,255,0.3);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.25) 0%, transparent 70%);
    transition: all 0.6s ease;
    opacity: 0;
}
.stat-card:hover::before {
    opacity: 1;
    transform: translate(-25%, -25%);
}
.stat-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 50px rgba(255,215,0,0.5), 0 0 60px rgba(255,215,0,0.3), inset 0 1px 0 rgba(255,255,255,0.4);
}
.stat-value {
    font-size: 3.5rem;
    font-weight: 900;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
    color: #0f0f0f;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stat-label {
    font-size: 0.95rem;
    font-weight: 700;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    position: relative;
    z-index: 1;
    color: #0f0f0f;
}
/* Input Areas */
.stTextArea textarea {
    border-radius: 18px;
    border: 2px solid rgba(218,165,32,0.35);
    font-size: 1.05rem;
    transition: all 0.3s ease;
    background: rgba(26,26,26,0.8) !important;
    color: #e5e7eb !important;
    padding: 1rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #FFD700;
    box-shadow: 0 0 0 4px rgba(255,215,0,0.15);
    background: rgba(26,26,26,0.95) !important;
}
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    color: #0f0f0f;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 2.8rem;
    font-size: 1.15rem;
    font-weight: 700;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(255,215,0,0.4), 0 0 30px rgba(255,215,0,0.2);
    width: 100%;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255,255,255,0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}
.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(255,215,0,0.6), 0 0 50px rgba(255,215,0,0.3);
}
.stButton > button:active {
    transform: translateY(-1px);
}
/* Dynamic Alert Boxes */
.alert-box {
    padding: 2rem;
    border-radius: 20px;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.5rem 0;
    border: 2px solid rgba(255,255,255,0.1);
    color: white;
}
.confidence-bar {
    height: 14px;
    background: rgba(255,255,255,0.25);
    border-radius: 12px;
    overflow: hidden;
    margin-top: 1rem;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}
.confidence-fill {
    height: 100%;
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px rgba(255,255,255,0.5);
}
/* Hints Panel */
.hints-panel {
    background: linear-gradient(135deg, rgba(26,26,26,0.95) 0%, rgba(15,15,15,0.95) 100%);
    border-radius: 20px;
    padding: 2rem;
    border-left: 5px solid #FFD700;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}
.hint-item {
    display: flex;
    align-items: start;
    gap: 1rem;
    margin-bottom: 1.2rem;
    font-size: 0.98rem;
    color: #d1d5db;
    line-height: 1.6;
}
.hint-icon {
    min-width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    color: #0f0f0f;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 800;
    box-shadow: 0 2px 8px rgba(255,215,0,0.4);
}
/* Metric Cards */
.metric-container {
    background: linear-gradient(135deg, rgba(26,26,26,0.95) 0%, rgba(15,15,15,0.95) 100%);
    padding: 1.8rem;
    border-radius: 16px;
    border-left: 5px solid #FFD700;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}
.metric-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08);
}
/* File Uploader */
.stFileUploader {
    border: 2px dashed rgba(218,165,32,0.45);
    border-radius: 18px;
    padding: 2rem;
    background: rgba(26,26,26,0.6);
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    border-color: #FFD700;
    background: rgba(218,165,32,0.12);
    box-shadow: 0 0 20px rgba(255,215,0,0.15);
}
/* Expanders */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(218,165,32,0.2) 0%, rgba(218,165,32,0.1) 100%) !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    color: #f5f5f5 !important;
    border: 1px solid rgba(218,165,32,0.3) !important;
    padding: 1rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, rgba(218,165,32,0.25) 0%, rgba(218,165,32,0.15) 100%) !important;
    border-color: rgba(218,165,32,0.5) !important;
}
/* Dataframe Styling */
.stDataFrame {
    background: rgba(26,26,26,0.95) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.stDataFrame [data-testid="stDataFrameResizable"] {
    background: rgba(26,26,26,0.95) !important;
}
.stDataFrame table {
    background: rgba(26,26,26,0.95) !important;
    color: #e5e7eb !important;
}
.stDataFrame thead tr th {
    background: rgba(218,165,32,0.2) !important;
    color: #FFD700 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid rgba(218,165,32,0.4) !important;
}
.stDataFrame tbody tr {
    background: rgba(26,26,26,0.8) !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}
.stDataFrame tbody tr:hover {
    background: rgba(218,165,32,0.1) !important;
}
.stDataFrame tbody tr td {
    color: #d1d5db !important;
}
/* Info/Warning boxes */
.stAlert {
    background: rgba(26,26,26,0.9) !important;
    border-radius: 12px !important;
    border-left: 4px solid #FFD700 !important;
    color: #e5e7eb !important;
}
/* Footer */
.footer {
    background: linear-gradient(135deg, rgba(26,26,26,0.95) 0%, rgba(15,15,15,0.95) 100%);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 4rem;
    color: #9ca3af;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    border: 2px solid rgba(218,165,32,0.3);
}
.footer-name {
    font-weight: 800;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.1rem;
}
/* Matplotlib figure styling */
.stPlotlyChart, .stPyplot {
    background: rgba(26,26,26,0.6) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}
/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: #1a1a1a;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
}
</style>
""", unsafe_allow_html=True)

# Utility Functions
@st.cache_data
def load_csv_from_bytes(uploaded_bytes):
    return pd.read_csv(io.BytesIO(uploaded_bytes))

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return pd.DataFrame()

def sanitize_input(text):
    """Sanitize user input to prevent injection"""
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text)
    return text

def validate_email_input(text):
    """Validate email input"""
    if len(text.strip()) < 10:
        return False, "Email content too short for analysis (minimum 10 characters)"
    if len(text) > 10000:
        return False, "Email content too long (maximum 10,000 characters)"
    return True, ""

@st.cache_data
def preprocess_text_cached(text):
    """Cached version of text preprocessing"""
    return preprocess_text(text)

def preprocess_text(text):
    """Enhanced preprocessing with better phishing indicator preservation"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Enhanced URL detection - preserve URL patterns better
    text = re.sub(r'http\S+|www\S+|https\S+', ' suspiciousurl ', text)
    text = re.sub(r'\S+@\S+', ' emailaddress ', text)
    # Preserve important phishing indicators
    text = re.sub(r'\$\d+', ' moneymention ', text)
    text = re.sub(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', ' cardnumber ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_phishing_score(text):
    """Enhanced phishing detection with multi-factor scoring"""
    score = 0
    text_lower = text.lower()
    
    # High-risk phishing keywords (weight: 15 points each)
    high_risk = ['verify', 'suspended', 'urgent', 'immediately', 'click here', 'act now',
                 'confirm identity', 'account locked', 'unusual activity', 'security alert',
                 'expire', 'limited time', 'action required', 'update payment', 'validate']
    score += sum(15 for word in high_risk if word in text_lower)
    
    # Financial/security keywords (weight: 12 points each)
    financial = ['bank', 'credit card', 'password', 'ssn', 'social security', 'paypal',
                 'billing', 'payment', 'account number', 'pin', 'cvv', 'credential']
    score += sum(12 for word in financial if word in text_lower)
    
    # Prize/reward scam indicators (weight: 18 points each)
    prize_scam = ['won', 'winner', 'prize', 'claim now', 'congratulations', 'free money',
                  'inheritance', 'lottery', 'jackpot', 'cash prize', '$1000', '$10000']
    score += sum(18 for word in prize_scam if word in text_lower)
    
    # Urgency + financial combo (weight: 25 points)
    if any(urg in text_lower for urg in ['urgent', 'immediately', 'now', 'expire']) and \
       any(fin in text_lower for fin in ['account', 'bank', 'payment', 'card']):
        score += 25
    
    # Suspicious URL patterns (weight: 20 points)
    if re.search(r'http\S+|www\S+', text, re.IGNORECASE):
        url_count = len(re.findall(r'http\S+|www\S+', text, re.IGNORECASE))
        score += min(url_count * 20, 40)  # Cap at 40 for multiple URLs
    
    # Request for credentials/info (weight: 20 points)
    if re.search(r'\b(enter|provide|submit|update|confirm).{0,20}(password|credential|info|detail)', text_lower):
        score += 20
    
    # Threatening language (weight: 15 points)
    threats = ['locked', 'suspended', 'terminated', 'closed', 'blocked', 'restricted']
    score += sum(15 for word in threats if word in text_lower)
    
    # Poor grammar indicators (weight: 8 points)
    if re.search(r'\b(dear customer|dear user|dear member|dear valued)\b', text_lower):
        score += 8
    
    # Convert to probability (0-1 scale)
    max_score = 200  # Adjusted maximum possible score
    probability = min(score / max_score, 0.99)  # Cap at 99%
    
    return probability

@st.cache_data
def generate_confusion_matrix_plot(_cm):
    """Generate confusion matrix plot once and cache it"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    sns.heatmap(
        _cm,
        annot=True,
        fmt="d",
        ax=ax,
        cmap="YlOrBr",
        cbar=True,
        square=True,
        annot_kws={"size": 16, "weight": "bold", "color": "#0f0f0f"},
        linewidths=2,
        linecolor='#0f0f0f',
        cbar_kws={'label': 'Count', 'shrink': 0.8}
    )
    
    ax.set_xlabel("Predicted", fontsize=11, fontweight='bold', color='#FFD700')
    ax.set_ylabel("Actual", fontsize=11, fontweight='bold', color='#FFD700')
    ax.set_xticklabels(["Safe", "Phishing"], fontsize=10, color='#e5e7eb')
    ax.set_yticklabels(["Safe", "Phishing"], fontsize=10, rotation=0, color='#e5e7eb')
    ax.set_title("Confusion Matrix", fontsize=13, fontweight='bold', pad=12, color='#FFD700')
    
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='#e5e7eb')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e5e7eb')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a1a', dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ AI Phishing Shield</div>
    <div class="hero-subtitle">Advanced Machine Learning Protection Against Email Threats</div>
    <div class="hero-description">
        Powered by TF-IDF vectorization and Logistic Regression, trained on thousands of real-world phishing examples.
        Get instant threat analysis with confidence scoring and explainable AI insights.
    </div>
    <div class="hero-badge">⚡ Developed by Umaima Qureshi</div>
</div>
""", unsafe_allow_html=True)

# Load Dataset
main_csv_path = "Phishing_Email.csv"
sample_csv_path = "Phishing_Email_Sample.csv"

st.markdown('<div class="section-title">📂 Dataset Configuration</div>', unsafe_allow_html=True)

# File uploader removed to prevent dynamic import errors in HuggingFace Spaces
if os.path.exists(main_csv_path):
    df = safe_read_csv(main_csv_path)
elif os.path.exists(sample_csv_path):
    st.info("📊 Using sample dataset for demonstration")
    df = safe_read_csv(sample_csv_path)
else:
    st.info("📊 Using built-in demo dataset")
    df = pd.DataFrame({
        "Email Text": [
            "Urgent! Your account has been suspended. Click http://fakebank.com to verify.",
            "WINNER! Claim your $1000 prize now at http://scam.com before it expires!",
            "Hi team, attached is the agenda for tomorrow's meeting. Regards.",
            "Hello Umaima, congrats on your results. Let's celebrate this week!",
            "Action required: Update your bank password at http://phishingsite.com immediately.",
            "Reminder: Project deadline is next Monday. Please submit your updates.",
            "Your PayPal account needs verification. Click here: http://fake-paypal.com",
            "Thanks for your email. I'll review the document and get back to you tomorrow."
        ],
        "Email Type": [
            "Phishing Email", "Phishing Email", "Safe Email", "Safe Email",
            "Phishing Email", "Safe Email", "Phishing Email", "Safe Email"
        ]
    })

# Validate dataset
required_columns = 2
if len(df.columns) < required_columns or len(df) == 0:
    st.error("⚠️ Invalid dataset format. Please ensure your CSV has email text and labels.")
    st.stop()

# Clean & Prepare Dataset
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

text_col = "Email Text" if "Email Text" in df.columns else df.columns[0]
label_col = "Email Type" if "Email Type" in df.columns else df.columns[-1]

df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip() != ""].reset_index(drop=True)
df = df.drop(index=0, errors="ignore").reset_index(drop=True)

label_map = {"Phishing Email": 1, "Safe Email": 0}
if df[label_col].dtype == object:
    df['label'] = df[label_col].map(label_map)
    df['label'] = df['label'].fillna(0).astype(int)
else:
    df['label'] = df[label_col].astype(int)

df['processed_text'] = df[text_col].apply(preprocess_text)

# Dataset Stats
phishing_count = (df['label'] == 1).sum()
safe_count = (df['label'] == 0).sum()
total_count = len(df)

st.markdown('<div class="section-title">📊 Dataset Statistics</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_count}</div>
        <div class="stat-label">Total Emails</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{phishing_count}</div>
        <div class="stat-label">Phishing Detected</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{safe_count}</div>
        <div class="stat-label">Safe Emails</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{(phishing_count/total_count*100):.1f}%</div>
        <div class="stat-label">Threat Rate</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 View Dataset Preview", expanded=False):
    st.dataframe(df[[text_col, label_col]].head(10), use_container_width=True)

# Model Training
@st.cache_resource
def train_model(processed_texts, labels, test_size=0.2, random_state=42):
    """Enhanced model training with better parameters"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = counts.min()
    
    min_test_samples = int(np.ceil(min_samples * test_size))
    min_train_samples = min_samples - min_test_samples
    
    use_stratify = (min_samples >= 2 and min_train_samples >= 1 and min_test_samples >= 1 and len(unique_labels) > 1)
    
    if not use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=random_state, stratify=None
        )
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=test_size, random_state=random_state, stratify=None
            )
    
    # Enhanced TF-IDF with better parameters for phishing detection
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,3),  # Include trigrams for better context
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Use balanced class weights for better phishing detection
    model = LogisticRegression(
        max_iter=2000,
        solver='liblinear',
        class_weight='balanced',  # Handle imbalanced data better
        C=1.0
    )
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }

# Train model with session state to prevent re-training
if not st.session_state.model_trained:
    model_info = train_model(df['processed_text'].tolist(), df['label'].values)
    st.session_state.model_info = model_info
    st.session_state.model_trained = True
else:
    model_info = st.session_state.model_info

vectorizer = model_info["vectorizer"]
model = model_info["model"]
accuracy = model_info["accuracy"]

# Model Performance
st.markdown('<div class="section-title">🎯 Model Performance</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Accuracy</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #FFD700;">{accuracy:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    precision = model_info["report"].get("1", {}).get("precision", 0)
    st.markdown(f"""
    <div class="metric-container">
        <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Precision</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #FFD700;">{precision:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    recall = model_info["report"].get("1", {}).get("recall", 0)
    st.markdown(f"""
    <div class="metric-container">
        <div style="color: #9ca3af; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Recall</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #FFD700;">{recall:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

# Confusion Matrix Section
with st.expander("📈 Detailed Metrics & Confusion Matrix"):
    col_matrix, col_report = st.columns([1, 1.5])
    
    with col_matrix:
        # Generate confusion matrix plot once
        if st.session_state.cm_plot_cached is None:
            st.session_state.cm_plot_cached = generate_confusion_matrix_plot(model_info["confusion_matrix"])
        
        st.image(st.session_state.cm_plot_cached, use_column_width=True)
    
    with col_report:
        st.markdown("**📊 Classification Report:**")
        report_df = pd.DataFrame(model_info["report"]).transpose().round(3)
        st.dataframe(
            report_df,
            use_container_width=True,
            height=250
        )

# Inference UI
st.markdown('<div class="section-title">✉️ Email Threat Scanner</div>', unsafe_allow_html=True)

col_input, col_hints = st.columns([2, 1])

with col_input:
    email_input = st.text_area(
        "Paste email content for analysis",
        height=280,
        placeholder="Example: Urgent! Your account has been compromised. Click here to verify your identity immediately...",
        help="Paste the full email content including subject and body"
    )
    
    uploaded_txt = st.file_uploader("Or upload a .txt file", type=["txt"], help="Upload a text file containing the email")
    
    if uploaded_txt is not None and not email_input:
        try:
            email_input = uploaded_txt.read().decode("utf-8", errors="ignore")
        except Exception:
            email_input = str(uploaded_txt.getvalue())
    
    if st.button("🔍 Analyze Email Threat"):
        if not email_input.strip():
            st.warning("⚠️ Please paste or upload email content to analyze")
        else:
            # Sanitize input
            email_input = sanitize_input(email_input)
            
            # Validate input
            is_valid, error_msg = validate_email_input(email_input)
            if not is_valid:
                st.warning(f"⚠️ {error_msg}")
            else:
                with st.spinner("🔍 Analyzing email threat..."):
                    try:
                        # ML Model prediction
                        processed_input = preprocess_text_cached(email_input)
                        input_vec = vectorizer.transform([processed_input])
                        
                        try:
                            ml_proba = model.predict_proba(input_vec)[0][1]
                        except AttributeError:
                            decision = model.decision_function(input_vec)[0]
                            ml_proba = 1 / (1 + np.exp(-decision))
                        
                        ml_pred = model.predict(input_vec)[0]
                        
                        # Rule-based scoring
                        rule_score = calculate_phishing_score(email_input)
                        
                        # Hybrid approach: weighted combination
                        # 60% ML model + 40% rule-based (adjustable)
                        hybrid_proba = (0.6 * ml_proba) + (0.4 * rule_score)
                        
                        # Final prediction based on hybrid score
                        final_pred = 1 if hybrid_proba > 0.5 else 0
                        
                        # Dynamic color based on confidence
                        if hybrid_proba >= 0.8:
                            alert_color = "#dc2626"  # Deep red - Critical
                            alert_gradient = "linear-gradient(135deg, #dc2626 0%, #991b1b 100%)"
                            shadow_color = "220, 38, 38"
                            emoji = "🚨"
                            risk_level = "CRITICAL THREAT"
                        elif hybrid_proba >= 0.6:
                            alert_color = "#ef4444"  # Red - High risk
                            alert_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                            shadow_color = "239, 68, 68"
                            emoji = "⚠️"
                            risk_level = "HIGH RISK"
                        elif hybrid_proba >= 0.4:
                            alert_color = "#f97316"  # Orange - Medium risk
                            alert_gradient = "linear-gradient(135deg, #f97316 0%, #ea580c 100%)"
                            shadow_color = "249, 115, 22"
                            emoji = "⚡"
                            risk_level = "MEDIUM RISK"
                        elif hybrid_proba >= 0.2:
                            alert_color = "#eab308"  # Yellow - Low risk
                            alert_gradient = "linear-gradient(135deg, #eab308 0%, #ca8a04 100%)"
                            shadow_color = "234, 179, 8"
                            emoji = "⚠️"
                            risk_level = "LOW RISK"
                        else:
                            alert_color = "#10b981"  # Green - Safe
                            alert_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                            shadow_color = "16, 185, 129"
                            emoji = "✅"
                            risk_level = "SAFE"
                        
                        if final_pred == 1:
                            conf_pct = f"{hybrid_proba:.1%}"
                            st.markdown(f"""
                            <div class="alert-box" style="background: {alert_gradient}; box-shadow: 0 10px 30px rgba({shadow_color}, 0.4), 0 0 50px rgba({shadow_color}, 0.2);">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                                    <div style="font-size: 2.5rem;">{emoji}</div>
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: 800; letter-spacing: 0.5px;">{risk_level} DETECTED</div>
                                        <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.25rem;">Threat Confidence: {conf_pct}</div>
                                        <div style="font-size: 0.9rem; opacity: 0.85; margin-top: 0.25rem;">ML Score: {ml_proba:.1%} | Rule Score: {rule_score:.1%}</div>
                                    </div>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {hybrid_proba*100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("**🔍 Threat Indicators Detected:**")
                            indicators = []
                            if "suspiciousurl" in processed_input or re.search(r'http\S+|www\S+', email_input, re.IGNORECASE):
                                indicators.append("🔗 Suspicious URL tokens detected")
                            if re.search(r'\b(urgent|immediately|verify|password|suspended|click|act now|action required)\b', email_input, re.IGNORECASE):
                                indicators.append("⚡ Urgency manipulation tactics")
                            if re.search(r'\b(bank|account|verify|login|password|security|credential|paypal)\b', email_input, re.IGNORECASE):
                                indicators.append("🏦 Financial/security keywords present")
                            if re.search(r'\b(winner|prize|congratulations|claim|free|won)\b', email_input, re.IGNORECASE):
                                indicators.append("🎁 Reward/prize baiting language")
                            if re.search(r'\b(confirm|update|validate|unlock|restore)\b', email_input, re.IGNORECASE):
                                indicators.append("🔐 Account action requests")
                            if "cardnumber" in processed_input:
                                indicators.append("💳 Credit card pattern detected")
                            if "moneymention" in processed_input:
                                indicators.append("💰 Money amount mentioned")
                            
                            for indicator in indicators:
                                st.markdown(f"- {indicator}")
                            
                            if not indicators:
                                st.markdown("- ⚠️ Content pattern matches known phishing templates")
                            
                            st.error("🚨 **Recommendation:** Do NOT click any links. Delete this email immediately and report to your IT security team.")
                            
                            # Download analysis report
                            result_data = {
                                'timestamp': pd.Timestamp.now(),
                                'prediction': 'Phishing',
                                'confidence': f"{hybrid_proba:.2%}",
                                'ml_score': f"{ml_proba:.2%}",
                                'rule_score': f"{rule_score:.2%}",
                                'risk_level': risk_level,
                                'email_preview': email_input[:100] + "..."
                            }
                            result_df = pd.DataFrame([result_data])
                            csv = result_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=csv,
                                file_name=f"phishing_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        else:
                            conf_pct = f"{(1-hybrid_proba):.1%}"
                            st.markdown(f"""
                            <div class="alert-box" style="background: {alert_gradient}; box-shadow: 0 10px 30px rgba({shadow_color}, 0.4), 0 0 50px rgba({shadow_color}, 0.2);">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                                    <div style="font-size: 2.5rem;">{emoji}</div>
                                    <div>
                                        <div style="font-size: 1.5rem; font-weight: 800; letter-spacing: 0.5px;">EMAIL APPEARS SAFE</div>
                                        <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 0.25rem;">Safety Confidence: {conf_pct}</div>
                                        <div style="font-size: 0.9rem; opacity: 0.85; margin-top: 0.25rem;">ML Score: {(1-ml_proba):.1%} | Rule Score: {(1-rule_score):.1%}</div>
                                    </div>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {(1-hybrid_proba)*100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("**✓ No obvious threat indicators found in content analysis**")
                            st.info("💡 **Best Practice:** Always verify sender identity through known contact methods and be cautious with unexpected emails, even if they appear safe.")
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            'timestamp': pd.Timestamp.now(),
                            'result': 'Phishing' if final_pred == 1 else 'Safe',
                            'confidence': f"{hybrid_proba:.2%}",
                            'preview': email_input[:50] + "..."
                        })
                    
                    except Exception as e:
                        st.error(f"⚠️ Analysis failed: {str(e)}")

with col_hints:
    st.markdown("""
    <div class="hints-panel">
        <div style="font-weight: 700; font-size: 1.15rem; margin-bottom: 1.2rem; color: #f5f5f5;">🧠 AI Detection Insights</div>
        
        <div class="hint-item">
            <div class="hint-icon">1</div>
            <div><strong>Urgency words</strong> like "urgent", "verify", "immediately" raise red flags</div>
        </div>
        
        <div class="hint-item">
            <div class="hint-icon">2</div>
            <div><strong>Suspicious links</strong> or email addresses are automatically flagged</div>
        </div>
        
        <div class="hint-item">
            <div class="hint-icon">3</div>
            <div><strong>Financial keywords</strong> combined with urgency indicate high risk</div>
        </div>
        
        <div class="hint-item">
            <div class="hint-icon">4</div>
            <div>Confidence <strong>>70%</strong> warrants immediate caution</div>
        </div>
        
        <div class="hint-item">
            <div class="hint-icon">5</div>
            <div><strong>Prize/reward</strong> language is a common phishing tactic</div>
        </div>
        
        <div class="hint-item">
            <div class="hint-icon">⚡</div>
            <div><strong>Hybrid Detection:</strong> Combines ML model (60%) with rule-based scoring (40%)</div>
        </div>
        
        <div class="hint-item">
            <div class="hint-icon">⚠️</div>
            <div><strong>Limitations:</strong> This tool analyzes text content only. Always verify sender identity separately.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Recent Analyses History
if len(st.session_state.analysis_history) > 0:
    st.markdown('<div class="section-title">📊 Recent Analyses</div>', unsafe_allow_html=True)
    with st.expander("View Recent Analysis History", expanded=False):
        hist_df = pd.DataFrame(st.session_state.analysis_history[-10:])  # Show last 10
        hist_df = hist_df.iloc[::-1]  # Reverse to show most recent first
        st.dataframe(hist_df, use_container_width=True, height=300)
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

# Additional Tips Section
st.markdown('<div class="section-title">💡 Phishing Protection Tips</div>', unsafe_allow_html=True)

col_tip1, col_tip2, col_tip3 = st.columns(3)

with col_tip1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(26,26,26,0.95) 0%, rgba(15,15,15,0.95) 100%);
                padding: 1.5rem; border-radius: 16px; border-left: 4px solid #FFD700;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); height: 100%;">
        <div style="font-size: 2rem; margin-bottom: 0.75rem;">🔍</div>
        <div style="font-weight: 700; font-size: 1.1rem; color: #FFD700; margin-bottom: 0.75rem;">Verify Sender</div>
        <div style="color: #d1d5db; font-size: 0.95rem; line-height: 1.6;">
            Always check the sender's email address carefully. Phishers often use addresses that look similar to legitimate ones.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_tip2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(26,26,26,0.95) 0%, rgba(15,15,15,0.95) 100%);
                padding: 1.5rem; border-radius: 16px; border-left: 4px solid #FFD700;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); height: 100%;">
        <div style="font-size: 2rem; margin-bottom: 0.75rem;">🔗</div>
        <div style="font-weight: 700; font-size: 1.1rem; color: #FFD700; margin-bottom: 0.75rem;">Hover Links</div>
        <div style="color: #d1d5db; font-size: 0.95rem; line-height: 1.6;">
            Hover over links (don't click!) to see the actual URL. Legitimate companies use their official domains.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_tip3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(26,26,26,0.95) 0%, rgba(15,15,15,0.95) 100%);
                padding: 1.5rem; border-radius: 16px; border-left: 4px solid #FFD700;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); height: 100%;">
        <div style="font-size: 2rem; margin-bottom: 0.75rem;">📞</div>
        <div style="font-weight: 700; font-size: 1.1rem; color: #FFD700; margin-bottom: 0.75rem;">Contact Directly</div>
        <div style="color: #d1d5db; font-size: 0.95rem; line-height: 1.6;">
            When in doubt, contact the company directly using official contact information, not the email's links.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="font-size: 1.2rem; margin-bottom: 0.75rem; font-weight: 700;">
        Developed and Deployed by <span class="footer-name">Umaima Qureshi</span>
    </div>
    <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 1rem; line-height: 1.6;">
        🎓 Educational demonstration of ML-powered email security<br>
        For production use: Implement additional verification layers, link scanning, attachment analysis, and human oversight
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(218,165,32,0.2); font-size: 0.9rem; color: #6b7280;">
        Powered by TF-IDF • Logistic Regression • Hybrid Detection • Scikit-learn • Streamlit
    </div>
    <div style="margin-top: 1rem; font-size: 0.85rem; color: #6b7280;">
        © 2024 AI Phishing Shield | All Rights Reserved
    </div>
</div>
""", unsafe_allow_html=True)
