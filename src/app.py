import streamlit as st

from predict import predict_sentiment

PALETTE = {
    "ink": "#12302F",
    "primary": "#0E6B66",
    "mint": "#E6F3F0",
    "sand": "#FAF8F4",
    "positive": "#17A589",
    "neutral": "#8A9BA8",
    "negative": "#E4572E",
    "card": "#FFFFFF",
    "border": "#E7E3DA",
    "muted": "#5B6F6C",
    "soft": "#F4F2EE",
}


def get_sentiment_scores(label: str) -> dict[str, float]:
    """Map a raw sentiment label to UI percentages for the result card."""
    normalized = str(label).strip().lower()
    if normalized == "positive":
        return {"positive": 0.82, "neutral": 0.11, "negative": 0.07}
    if normalized == "negative":
        return {"positive": 0.06, "neutral": 0.11, "negative": 0.83}
    return {"positive": 0.30, "neutral": 0.45, "negative": 0.25}


st.set_page_config(
    page_title="Sentiment Lens",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #12302F;
        --primary: #0E6B66;
        --mint: #E6F3F0;
        --sand: #FAF8F4;
        --positive: #17A589;
        --neutral: #8A9BA8;
        --negative: #E4572E;
        --card: #FFFFFF;
        --border: #E7E3DA;
        --muted: #5B6F6C;
        --soft: #F4F2EE;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--sand);
        color: var(--ink);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1100px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }

    #MainMenu {
        display: none;
    }

    footer {
        display: none;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.3rem;
    }

    .brand {
        display: inline-flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -0.04em;
    }

    .brand-mark {
        display: inline-flex;
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 0.8rem;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--primary), #2EA39D);
        color: white;
        font-size: 1rem;
        box-shadow: none;
    }

    .pill-tag {
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        background: rgba(14, 107, 102, 0.08);
        color: var(--primary);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        border: 1px solid rgba(14, 107, 102, 0.14);
    }

    .hero {
        background: var(--mint);
        border: 1px solid rgba(14, 107, 102, 0.08);
        border-radius: 28px;
        padding: 2.5rem 2.5rem 1.9rem;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        color: var(--ink);
        font-size: clamp(2.2rem, 4vw, 4rem);
        line-height: 1.05;
        letter-spacing: -0.06em;
        max-width: 720px;
    }

    .hero p {
        margin-top: 0.8rem;
        margin-bottom: 0;
        color: var(--muted);
        font-size: 1.04rem;
        font-weight: 500;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.35rem 1.3rem 1.15rem;
    }

    .section-title {
        margin: 0 0 0.9rem;
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .stForm {
        border: none;
        background: transparent;
    }

    div[data-testid="stTextArea"] > div {
        border-radius: 16px;
        border: 1px solid rgba(14, 107, 102, 0.18) !important;
        box-shadow: none !important;
        background: #fff;
    }

    div[data-testid="stTextArea"] textarea {
        min-height: 170px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1rem !important;
        color: var(--ink) !important;
        border-radius: 16px !important;
        background: #fff !important;
        resize: vertical;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(14, 107, 102, 0.08) !important;
    }

    .stButton > button {
        width: 100%;
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.75rem 1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #0B5A55;
        color: white;
    }

    .result-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.35rem 1.3rem 1.15rem;
        min-height: 100%;
    }

    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 338px;
        border-radius: 18px;
        background: linear-gradient(180deg, #f9f7f3, #f4f2ee);
        border: 1px dashed rgba(18, 48, 47, 0.14);
        color: var(--muted);
        font-size: 0.98rem;
        font-weight: 600;
        text-align: center;
    }

    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 126px;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }

    .sentiment-positive {
        background: rgba(23, 165, 137, 0.12);
        color: var(--positive);
        border-color: rgba(23, 165, 137, 0.14);
    }

    .sentiment-neutral {
        background: rgba(138, 155, 168, 0.12);
        color: var(--neutral);
        border-color: rgba(138, 155, 168, 0.16);
    }

    .sentiment-negative {
        background: rgba(228, 87, 46, 0.12);
        color: var(--negative);
        border-color: rgba(228, 87, 46, 0.14);
    }

    .result-grid {
        display: grid;
        grid-template-columns: 150px 1fr;
        align-items: center;
        gap: 1.1rem;
        margin-top: 0.2rem;
    }

    .donut-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .donut {
        position: relative;
        width: 118px;
        height: 118px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(var(--positive) 0% 82%, #E7E3DA 82% 100%);
    }

    .donut::before {
        content: "";
        position: absolute;
        inset: 15px;
        background: white;
        border-radius: 50%;
        border: 1px solid var(--border);
    }

    .donut-inner {
        position: absolute;
        z-index: 1;
        font-weight: 800;
        font-size: 1.05rem;
        color: var(--ink);
        letter-spacing: -0.04em;
    }

    .score-bars {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }

    .score-row {
        display: grid;
        grid-template-columns: 66px 1fr 44px;
        align-items: center;
        column-gap: 0.7rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink);
    }

    .score-row span:first-child {
        text-transform: capitalize;
    }

    .bar-track {
        height: 10px;
        background: #F0EFEA;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid rgba(18, 48, 47, 0.04);
    }

    .bar-fill {
        height: 100%;
        border-radius: inherit;
    }

    .bar-positive { background: var(--positive); }
    .bar-neutral { background: var(--neutral); }
    .bar-negative { background: var(--negative); }

    .feature-grid {
        margin-top: 1.5rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }

    .feature-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.1rem 1rem 1rem;
    }

    .feature-score {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 3.1rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        background: rgba(14, 107, 102, 0.08);
        color: var(--primary);
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }

    .feature-card h4 {
        margin: 0 0 0.45rem;
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .feature-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    @media (max-width: 900px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }

        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='topbar'>
        <div class='brand'><span class='brand-mark'>S</span>Sentiment Lens</div>
        <div class='pill-tag'>AI-powered review analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
        <h1>Know how customers feel about your product</h1>
        <p>Turn product feedback into clear sentiment signals in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

