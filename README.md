# Forward Matrix Scanner

A Streamlit app that scores any NSE stock on Dr Shantanu Samanta's **Forward Sentiment Matrix**: valuation, forward growth, earnings quality, balance sheet, ownership and market positioning. It gives each metric +1 / 0 / −1, six pillar scores and one composite score from −100 to +100, for a single company or a whole watchlist.

## What you need before you start
- A GitHub account (free): github.com
- A Streamlit Community Cloud account (free): share.streamlit.io. Sign in with GitHub.
- No API keys. Data comes from Yahoo Finance through the free `yfinance` library.
- Optional: your own weekly Trendlyne screener export (CSV/XLSX) for ownership, pledge, insider and delivery data.

## How to deploy (about 5 minutes)

### Step 1: Put the code on GitHub
1. On github.com click **New** (green button), name the repository `forward-matrix-app`, choose **Private** or Public, click **Create repository**.
2. Click **uploading an existing file**.
3. Drag in **all** files from this folder: `app.py`, `engine.py`, `requirements.txt`, `README.md`, `.gitignore` and the `.streamlit` folder (with `config.toml` inside). Click **Commit changes**.
   *Tip:* if the `.streamlit` folder doesn't upload by drag-and-drop, click **Add file → Create new file**, type `.streamlit/config.toml` as the name and paste its contents.

### Step 2: Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and click **Create app → Deploy a public app from GitHub**.
2. Pick your repository, branch `main`, main file path `app.py`.
3. Click **Deploy**. The first build takes 2–3 minutes.

## Data sources
- **Yahoo Finance (yfinance):** prices, statements, consensus EPS and revenue estimates, 90-day EPS trend, analyst ratings, earnings surprises, target range.
- **NSE daily reports (optional, you upload them):** NSE's Terms of Use prohibit automated data collection, so the app does not download from NSE. Download the equity "Full Bhavcopy and Security Deliverable data" CSVs and F&O UDiFF bhavcopy files yourself from nseindia.com (All Reports) and upload them in the sidebar to enable delivery % (M1), F&O OI build-up (M4) and the price cross-check.
- **Google News RSS (optional toggle):** last 14 days of headlines with a simple tone score. Shown for context only.

## How to use
1. **Sidebar:** type the current India 10-year G-Sec yield (Yahoo has no symbol for it; check TVC:IN10Y on TradingView). Optionally upload your Trendlyne export.
2. **Single company:** type an NSE symbol (e.g. `TCS`, `HDFCBANK`) and press **Scan company**. You get the composite, pillar chart, a full metric scorecard with the reason for every score, valuation-history, EPS-estimate-trend and 200-DMA charts, and a CSV download.
3. **Watchlist ranking:** paste up to 300 symbols, or upload a CSV with symbols in the first column, and press **Scan watchlist**. Download the ranked table as CSV.

**Trendlyne file tip:** include the **NSE code** column in your Trendlyne view. Without it the app matches by company name, which misses short names such as "L&T".

## Running locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to update the app
Upload the changed file(s) to the same GitHub repository. Streamlit Cloud redeploys automatically within a minute.

## Good to know
- Results are cached for 6 hours. Use **Clear cached data** in the sidebar for fresh numbers.
- Yahoo sometimes rate-limits shared cloud servers. If scans start failing, wait a few minutes or scan smaller lists.
- yfinance is an unofficial library meant for personal use; Yahoo can change or remove fields without notice.

---
© 2026 Dr Shantanu Samanta. For educational and research purposes only. This is not investment advice. Investments in securities market are subject to market risks. Read all the related documents carefully before investing.
