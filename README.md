# Earnings Call Sentiment Analyzer

A web application that analyzes earnings call transcripts using FinBERT — a finance-specific NLP model, and correlates management sentiment with post-earnings stock price reaction.

🔗 **[Live App](https://earnings-call-analyzer-zjszhewofdjlpk7nfx9ycg.streamlit.app)**

---

## What it does

- Scores management language from any earnings call transcript across three dimensions: positive, negative, and neutral tone
- Detects hedging language (e.g. *"subject to significant risks and uncertainties"*) that may signal management caution
- Extracts the top forward-looking signals and risk flags from the transcript
- Pulls historical stock price data automatically and calculates T+1 and T+5 post-earnings returns
- Generates a sentence-by-sentence sentiment breakdown table

---

## Why it matters

Buy-side analysts routinely read earnings call transcripts to assess management confidence beyond what the financial statements show. This tool systematizes that qualitative process, turning linguistic patterns into quantifiable scores that can be tracked across quarters and compared across companies.

---

## Tech stack

| Component | Tool |
|---|---|
| Sentiment model | FinBERT (ProsusAI/finbert via HuggingFace) |
| Stock price data | yfinance (Yahoo Finance API) |
| Visualization | Plotly |
| App framework | Streamlit |
| Language | Python 3.13 |

---

## Sample output — NVDA Q4 2026

- **Sentences analyzed:** 100
- **Avg positive score:** 4.5 / 10
- **Avg negative score:** 0.36 / 10
- **Key risk flag:** Chinese competitors bolstered by recent IPOs disrupting market structure
- **Hedging language detected:** *"subject to a number of significant risks and uncertainties, and our actual results may differ materially"*

---

## How to run locally

```bash
pip install streamlit yfinance plotly pandas transformers torch
streamlit run app.py
```

---

## About

Built by [Shreejal Kanse](https://www.linkedin.com/in/shreejal-kanse) — MS Quantitative Finance, Northeastern University. 

Focused on the intersection of finance and data — FP&A, investment research, and applied AI in financial analysis.
