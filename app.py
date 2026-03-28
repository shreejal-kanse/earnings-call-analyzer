import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
from transformers import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")
st.title("Earnings Call Sentiment Analyzer")
st.caption("Powered by FinBERT — finance-specific NLP model")

# ── Load FinBERT model ─────────────────────────────────────────────────────────
# This downloads the model the first time (~400MB), then caches it locally.
# @st.cache_resource means it only loads once per session, not on every rerun.
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None
    )

with st.spinner("Loading FinBERT model (first run may take 1-2 minutes)..."):
    finbert = load_model()

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    ticker = st.text_input(
        "Stock Ticker",
        value="NVDA",
        help="e.g. NVDA, META, SBUX"
    ).upper()

    earnings_date = st.date_input(
        "Earnings Date",
        help="The date the earnings call took place"
    )

    quarter_label = st.text_input(
        "Quarter Label",
        value="Q3 2024",
        help="e.g. Q3 2024"
    )

    transcript = st.text_area(
        "Paste Earnings Call Transcript",
        height=300,
        placeholder="Paste the transcript here. Get it from Motley Fool or Seeking Alpha."
    )

    run_button = st.button("Analyze", type="primary", use_container_width=True)

# ── Helper: split transcript into sentences ────────────────────────────────────
# FinBERT works on individual sentences, not the whole transcript at once.
# We split by period and filter out very short fragments.
def split_sentences(text: str):
    raw = text.replace("\n", " ").split(".")
    return [s.strip() for s in raw if len(s.strip()) > 30]

# ── Helper: run FinBERT on transcript ─────────────────────────────────────────
# Returns overall scores and a per-sentence breakdown
def analyze_with_finbert(text: str):
    sentences = split_sentences(text)

    # FinBERT can only handle ~512 tokens per sentence.
    # We truncate long sentences to be safe.
    sentences = [s[:400] for s in sentences[:100]]  # max 100 sentences

    results = finbert(sentences)

    pos_scores, neg_scores, neu_scores = [], [], []
    sentence_data = []

    for sent, result in zip(sentences, results):
        scores = {r["label"]: r["score"] for r in result}
        pos = scores.get("positive", 0)
        neg = scores.get("negative", 0)
        neu = scores.get("neutral",  0)

        pos_scores.append(pos)
        neg_scores.append(neg)
        neu_scores.append(neu)

        # Dominant label for this sentence
        dominant = max(scores, key=scores.get)
        sentence_data.append({
            "Sentence": sent[:120] + "..." if len(sent) > 120 else sent,
            "Sentiment": dominant.capitalize(),
            "Positive":  round(pos * 100, 1),
            "Negative":  round(neg * 100, 1),
            "Neutral":   round(neu * 100, 1),
        })

    avg_pos = sum(pos_scores) / len(pos_scores) * 10  # scale to 0-10
    avg_neg = sum(neg_scores) / len(neg_scores) * 10
    avg_neu = sum(neu_scores) / len(neu_scores) * 10

    # Hedging: sentences with high neutral + some negative score
    hedging = [
        s["Sentence"] for s in sentence_data
        if s["Neutral"] > 60 and s["Negative"] > 15
    ][:5]

    # Top positive signals: most positive sentences
    top_signals = sorted(sentence_data, key=lambda x: x["Positive"], reverse=True)[:3]

    # Top risk flags: most negative sentences
    risk_flags = sorted(sentence_data, key=lambda x: x["Negative"], reverse=True)[:3]

    return {
        "avg_pos": round(avg_pos, 2),
        "avg_neg": round(avg_neg, 2),
        "avg_neu": round(avg_neu, 2),
        "confidence_score": round(avg_pos, 1),
        "sentence_data": sentence_data,
        "hedging_flags": hedging,
        "top_signals": [s["Sentence"] for s in top_signals],
        "risk_flags":  [s["Sentence"] for s in risk_flags],
    }

# ── Helper: fetch stock price data ────────────────────────────────────────────
def get_price_reaction(ticker: str, earnings_date):
    start = earnings_date - timedelta(days=3)
    end   = earnings_date + timedelta(days=7)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df

# ── Helper: calculate T+1 and T+5 returns ─────────────────────────────────────
def calc_returns(df: pd.DataFrame, earnings_date):
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    pre  = df_sorted[df_sorted["Date"] <= earnings_date]
    post = df_sorted[df_sorted["Date"] >  earnings_date]
    if pre.empty or post.empty:
        return None, None
    base = pre.iloc[-1]["Close"]
    t1 = ((post.iloc[0]["Close"] - base) / base * 100) if len(post) >= 1 else None
    t5 = ((post.iloc[4]["Close"] - base) / base * 100) if len(post) >= 5 else None
    return (round(float(t1), 2) if t1 is not None else None,
            round(float(t5), 2) if t5 is not None else None)

# ── Helper: gauge chart ────────────────────────────────────────────────────────
def make_gauge(value: float, title: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 10]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 4],  "color": "#f5c4b3"},
                {"range": [4, 7],  "color": "#faeeda"},
                {"range": [7, 10], "color": "#c0dd97"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ── Helper: sentiment breakdown bar chart ─────────────────────────────────────
def make_sentiment_bar(avg_pos, avg_neg, avg_neu):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positive", x=["Sentiment"], y=[avg_pos],
                         marker_color="#1D9E75"))
    fig.add_trace(go.Bar(name="Negative", x=["Sentiment"], y=[avg_neg],
                         marker_color="#D85A30"))
    fig.add_trace(go.Bar(name="Neutral",  x=["Sentiment"], y=[avg_neu],
                         marker_color="#888780"))
    fig.update_layout(
        barmode="group", height=250,
        title="Average Sentiment Scores (0–10 scale)",
        margin=dict(t=50, b=30, l=40, r=20)
    )
    return fig

# ── Helper: price chart ────────────────────────────────────────────────────────
def make_price_chart(df: pd.DataFrame, earnings_date, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"].astype(str), y=df["Close"],
        mode="lines+markers",
        line=dict(color="#378ADD", width=2),
        marker=dict(size=6), name=ticker
    ))
    fig.add_shape(
        type="line",
        x0=str(earnings_date),
        x1=str(earnings_date),
        y0=0, y1=1, yref="paper",
        line=dict(color="#D85A30", dash="dash", width=2)
    )
    fig.update_layout(
        title=f"{ticker} Price Around Earnings",
        xaxis_title="Date", yaxis_title="Close Price (USD)",
        height=320, margin=dict(t=50, b=40, l=50, r=20)
    )
    return fig

# ── Main analysis flow ─────────────────────────────────────────────────────────
if run_button:
    if not transcript.strip():
        st.error("Please paste a transcript before running.")
        st.stop()

    with st.spinner("Running FinBERT analysis and fetching price data..."):
        result   = analyze_with_finbert(transcript)
        price_df = get_price_reaction(ticker, earnings_date)
        t1_ret, t5_ret = calc_returns(price_df, earnings_date)

    # ── Row 1: Key metrics ─────────────────────────────────────────────────
    st.subheader(f"{ticker} — {quarter_label} Earnings Analysis")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentences Analyzed", len(result["sentence_data"]))
    col2.metric("Avg Positive Score", f"{result['avg_pos']:.1f} / 10")
    col3.metric("Avg Negative Score", f"{result['avg_neg']:.1f} / 10")
    if t1_ret is not None:
        col4.metric("T+1 Stock Return", f"{t1_ret:+.2f}%")

    # ── Row 2: Gauges + bar ────────────────────────────────────────────────
    st.subheader("Sentiment Scores")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(result["avg_pos"], "Positive Tone", "#1D9E75"),
                        use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(result["avg_neg"], "Negative Tone", "#D85A30"),
                        use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(result["avg_neu"], "Neutral Tone",  "#888780"),
                        use_container_width=True)

    st.plotly_chart(make_sentiment_bar(
        result["avg_pos"], result["avg_neg"], result["avg_neu"]
    ), use_container_width=True)

    # ── Row 3: Price chart ─────────────────────────────────────────────────
    st.subheader("Price Reaction")
    if not price_df.empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.plotly_chart(make_price_chart(price_df, earnings_date, ticker),
                            use_container_width=True)
        with c2:
            st.markdown("**Returns**")
            if t1_ret is not None:
                st.metric("T+1", f"{t1_ret:+.2f}%")
            if t5_ret is not None:
                st.metric("T+5", f"{t5_ret:+.2f}%")
    else:
        st.warning("Could not fetch price data — check the ticker and date.")

    # ── Row 4: Qualitative signals ─────────────────────────────────────────
    st.subheader("Qualitative Analysis")
    qa1, qa2, qa3 = st.columns(3)

    with qa1:
        st.markdown("**Top positive signals**")
        for s in result["top_signals"]:
            st.markdown(f"- {s}")

    with qa2:
        st.markdown("**Hedging language detected**")
        if result["hedging_flags"]:
            for h in result["hedging_flags"]:
                st.markdown(f"- _{h}_")
        else:
            st.markdown("_No strong hedging detected_")

    with qa3:
        st.markdown("**Risk flags**")
        for r in result["risk_flags"]:
            st.markdown(f"- {r}")

    # ── Row 5: Full sentence table ─────────────────────────────────────────
    st.subheader("Sentence-by-Sentence Breakdown")
    df_display = pd.DataFrame(result["sentence_data"])
    st.dataframe(df_display, use_container_width=True, height=400)